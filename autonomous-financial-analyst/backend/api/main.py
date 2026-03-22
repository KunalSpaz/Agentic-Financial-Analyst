"""
main.py
-------
FastAPI application factory.

Creates the FastAPI app, registers all routers, sets up CORS,
runs database migrations on startup, and starts the APScheduler.
"""
from __future__ import annotations

import os
import threading
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.api.routes.stock_routes import router as stock_router
from backend.api.routes.market_routes import router as market_router
from backend.api.routes.portfolio_routes import router as portfolio_router
from backend.api.routes.backtest_routes import router as backtest_router
from backend.api.routes.optimization_routes import router as optimization_router
from backend.api.routes.rag_routes import router as rag_router
from backend.api.routes.chat_routes import router as chat_router
from backend.database.migrations import run_migrations
from backend.utils.config import settings
from backend.utils.logger import get_logger, request_id_ctx
from backend.utils.scheduler import start_scheduler, shutdown_scheduler

logger = get_logger(__name__)

# ── Rate limiter (shared instance imported by route modules) ──────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Public paths that bypass API key auth ────────────────────────────────────
_PUBLIC_PATHS = {"/", "/docs", "/redoc", "/openapi.json"}


def _prewarm_finbert() -> None:
    """Load FinBERT in a background thread at startup so the first request is fast."""
    try:
        from backend.services.sentiment_service import SentimentService
        SentimentService()._get_pipeline()
        logger.info("FinBERT pre-warm complete.")
    except Exception as exc:
        logger.warning("FinBERT pre-warm failed (sentiment will load on first use): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Startup:  creates data/ dir, runs DB migrations, pre-warms FinBERT, starts scheduler.
    Shutdown: gracefully stops the scheduler.
    """
    logger.info("Starting Autonomous Financial Analyst API (env=%s)…", settings.app_env)

    # ── Fix 5: Ensure data/ directory exists before DB / FAISS use ────────────
    os.makedirs("data", exist_ok=True)
    logger.info("data/ directory ensured.")

    # Run database schema migrations
    run_migrations()

    # ── Fix 6: Pre-warm FinBERT in background so startup isn't blocked ────────
    threading.Thread(target=_prewarm_finbert, daemon=True, name="finbert-prewarm").start()

    # Start background scheduler
    if settings.scheduler_enabled:
        start_scheduler()
        logger.info("APScheduler started.")

    yield

    # Graceful shutdown
    if settings.scheduler_enabled:
        shutdown_scheduler()
        logger.info("APScheduler stopped.")

    logger.info("API shutdown complete.")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured :class:`fastapi.FastAPI` instance.
    """
    app = FastAPI(
        title="Autonomous Financial Analyst",
        description=(
            "AI-powered financial intelligence platform with multi-agent analysis, "
            "backtesting, portfolio risk management, and daily market briefings."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Rate limiter ──────────────────────────────────────────────────────────
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Generic 500 handler — never leak internal exception details ───────────
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method, request.url.path, exc,
        )
        body: dict = {"detail": "Internal server error."}
        if settings.app_env == "development":
            body["debug"] = str(exc)
        return JSONResponse(status_code=500, content=body)

    # ── Fix 1: CORS — restrict to configured origins ──────────────────────────
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*", "X-API-Key"],
    )

    # ── Request correlation ID ────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Propagate or generate X-Request-ID; bind it to the log context."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_ctx.reset(token)

    # ── Security response headers ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add standard security headers to every response."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    # ── Fix 2: API key authentication middleware ──────────────────────────────
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        """Enforce X-API-Key header when API_SECRET_KEY is configured."""
        if settings.api_secret_key:
            if request.url.path not in _PUBLIC_PATHS:
                provided = request.headers.get("X-API-Key", "")
                if provided != settings.api_secret_key:
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Invalid or missing X-API-Key header."},
                    )
        return await call_next(request)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(stock_router)
    app.include_router(market_router)
    app.include_router(portfolio_router)
    app.include_router(backtest_router)
    app.include_router(optimization_router)
    app.include_router(rag_router)
    app.include_router(chat_router)

    @app.get("/", tags=["Health"])
    async def health_check() -> dict:
        """API health check endpoint."""
        return {
            "status": "ok",
            "service": "Autonomous Financial Analyst",
            "version": "1.0.0",
            "env": settings.app_env,
        }

    logger.info("FastAPI app created with %d routes.", len(app.routes))
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
