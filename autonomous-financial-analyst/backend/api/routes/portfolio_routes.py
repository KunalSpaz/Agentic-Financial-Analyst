"""
portfolio_routes.py
--------------------
FastAPI routes for portfolio risk analysis.

Endpoints:
    POST /portfolio-analysis — Compute full risk metrics for a portfolio
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.portfolio_risk_report import PortfolioRiskReport
from backend.services.portfolio_risk_service import PortfolioRiskService
from backend.utils.logger import get_logger

router = APIRouter(tags=["Portfolio"])
logger = get_logger(__name__)

_prs = PortfolioRiskService()


class PortfolioRequest(BaseModel):
    """Request body for /portfolio-analysis."""
    holdings: Dict[str, float] = Field(
        ...,
        description="Dict mapping ticker symbols to portfolio weights, e.g. {'AAPL': 0.4, 'MSFT': 0.6}",
        example={"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.3},
    )
    period: str = Field(default="1y", description="Historical look-back window")


@router.post("/portfolio-analysis", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def analyse_portfolio(
    request: Request,
    body: PortfolioRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Compute comprehensive risk metrics for the submitted portfolio.

    Returns volatility, beta, Sharpe ratio, max drawdown, VaR, correlation
    matrix, and sector exposure.
    """
    if not body.holdings:
        raise HTTPException(status_code=400, detail="Holdings dict cannot be empty.")

    try:
        result = await asyncio.to_thread(_prs.analyse, body.holdings, period=body.period)
    except Exception as exc:
        logger.error("Portfolio analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Portfolio analysis failed. Please try again later.") from exc

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    # Persist report
    try:
        portfolio_id = str(uuid.uuid4())[:8]
        report = PortfolioRiskReport(
            portfolio_id=portfolio_id,
            holdings=body.holdings,
            portfolio_volatility=result.get("portfolio_volatility"),
            portfolio_beta=result.get("portfolio_beta"),
            max_drawdown=result.get("max_drawdown"),
            sharpe_ratio=result.get("sharpe_ratio"),
            correlation_matrix=result.get("correlation_matrix"),
            sector_exposure=result.get("sector_exposure"),
            var_95=result.get("var_95_daily"),
        )
        db.add(report)
        db.commit()
        result["portfolio_id"] = portfolio_id
    except Exception as exc:
        logger.warning("Failed to persist portfolio report: %s", exc)

    return result
