"""
optimization_routes.py
-----------------------
FastAPI routes for strategy optimization.

Endpoints:
    POST /optimize-strategy — Grid-search optimize strategy parameters
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.strategy_optimization import StrategyOptimization
from backend.services.strategy_optimization_service import StrategyOptimizationService
from backend.utils.logger import get_logger

router = APIRouter(tags=["Optimization"])
logger = get_logger(__name__)

_opt = StrategyOptimizationService()


class OptimizationRequest(BaseModel):
    """Request body for /optimize-strategy."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    objective: str = Field(
        default="maximize_return",
        pattern="^(maximize_return|maximize_sharpe|minimize_drawdown)$",
        description="Optimization objective",
    )
    period: str = Field(default="2y", description="Historical data window for backtests")
    sentiment_label: str = Field(default="NEUTRAL", pattern="^(POSITIVE|NEUTRAL|NEGATIVE)$")


@router.post("/optimize-strategy", response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def optimize_strategy(
    request: Request,
    body: OptimizationRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run an exhaustive grid-search parameter optimization for the given ticker.

    Searches RSI thresholds, MACD confirmation, and MA filter combinations
    to find the configuration that best satisfies the chosen objective.

    Returns best parameters, best metrics, iteration count, and top-20 results.
    """
    ticker = body.ticker.upper()

    try:
        result = await asyncio.to_thread(
            _opt.optimize,
            ticker=ticker,
            objective=body.objective,
            period=body.period,
            sentiment_label=body.sentiment_label,
        )
    except Exception as exc:
        logger.error("Strategy optimization failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Strategy optimization failed. Please try again later.") from exc

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Persist
    try:
        db_opt = StrategyOptimization(
            ticker=ticker,
            objective=body.objective,
            best_parameters=result.get("best_parameters"),
            best_return=result.get("best_return"),
            best_sharpe=result.get("best_sharpe"),
            best_drawdown=result.get("best_drawdown"),
            iterations=result.get("iterations"),
            all_results=result.get("all_results"),
        )
        db.add(db_opt)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to persist optimization result: %s", exc)

    return result
