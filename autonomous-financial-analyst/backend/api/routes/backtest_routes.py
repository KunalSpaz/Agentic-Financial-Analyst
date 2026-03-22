"""
backtest_routes.py
------------------
FastAPI routes for strategy backtesting.

Endpoints:
    POST /backtest — Run historical strategy simulation
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.models.backtest_result import BacktestResult as BacktestResultModel
from backend.services.backtesting_service import BacktestParameters, BacktestingService
from backend.utils.logger import get_logger

router = APIRouter(tags=["Backtesting"])
logger = get_logger(__name__)

_bt = BacktestingService()


class BacktestRequest(BaseModel):
    """Request body for the /backtest endpoint."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol")
    period: str = Field(default="2y", description="Historical data window")
    rsi_buy_threshold: float = Field(default=35.0, ge=5, le=50)
    rsi_sell_threshold: float = Field(default=70.0, ge=50, le=95)
    macd_confirmation: bool = Field(default=True)
    ma_filter: bool = Field(default=True)
    initial_capital: float = Field(default=10_000.0, ge=100)
    sentiment_label: str = Field(default="NEUTRAL", pattern="^(POSITIVE|NEUTRAL|NEGATIVE)$")


@router.post("/backtest", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def run_backtest(
    request: Request,
    body: BacktestRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run a historical strategy backtest for the given ticker and parameters.

    Returns total return, Sharpe ratio, max drawdown, win rate, number of trades,
    equity curve, and full trade log.
    """
    ticker = body.ticker.upper()

    params = BacktestParameters(
        rsi_buy_threshold=body.rsi_buy_threshold,
        rsi_sell_threshold=body.rsi_sell_threshold,
        macd_confirmation=body.macd_confirmation,
        ma_filter=body.ma_filter,
        initial_capital=body.initial_capital,
    )

    try:
        result = await asyncio.to_thread(
            _bt.run_backtest, ticker,
            params=params, period=body.period, sentiment_label=body.sentiment_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Backtest failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Backtest failed. Please try again later.") from exc

    # Persist
    try:
        db_result = BacktestResultModel(
            ticker=result.ticker,
            strategy_name=result.strategy_name,
            start_date=result.start_date,
            end_date=result.end_date,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            num_trades=result.num_trades,
            parameters=result.parameters,
            equity_curve=result.equity_curve,
            trade_log=result.trade_log,
        )
        db.add(db_result)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to persist backtest result: %s", exc)

    return {
        "ticker": result.ticker,
        "strategy_name": result.strategy_name,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "parameters": result.parameters,
        "metrics": {
            "total_return": result.total_return,
            "total_return_pct": f"{result.total_return:.2%}",
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_pct": f"{result.max_drawdown:.2%}",
            "win_rate": result.win_rate,
            "win_rate_pct": f"{result.win_rate:.2%}",
            "num_trades": result.num_trades,
        },
        "equity_curve": result.equity_curve,
        "trade_log": result.trade_log,
    }
