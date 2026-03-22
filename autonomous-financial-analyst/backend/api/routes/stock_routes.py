"""
stock_routes.py
---------------
FastAPI routes for individual stock analysis.

Endpoints:
    POST /analyze-stock   — Run full CrewAI analysis pipeline for a ticker
    GET  /stock/{ticker}  — Get quote + cached analysis for a ticker
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

limiter = Limiter(key_func=get_remote_address)

from backend.agents.crew_orchestrator import StockAnalysisCrew
from backend.database.connection import get_db
from backend.models.analysis_report import AnalysisReport
from backend.models.user_query import UserQuery
from backend.services.market_data_service import MarketDataService
from backend.utils.logger import get_logger

router = APIRouter(tags=["Stock Analysis"])
logger = get_logger(__name__)

_mds = MarketDataService()
_crew = StockAnalysisCrew()


class AnalyzeStockRequest(BaseModel):
    """Request body for the /analyze-stock endpoint."""
    ticker: str = Field(..., min_length=1, max_length=10, description="Stock ticker symbol, e.g. 'AAPL'")
    period: str = Field(default="1y", description="Historical data period")


@router.post("/analyze-stock", response_model=Dict[str, Any])
@limiter.limit("5/minute")
async def analyze_stock(
    request: Request,
    body: AnalyzeStockRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run the complete 8-agent CrewAI analysis pipeline for the given ticker.

    Returns a full analysis including recommendation, confidence score,
    technical signals, sentiment, and a narrative report.
    """
    ticker = body.ticker.upper()
    start_ts = time.time()

    try:
        result = await asyncio.to_thread(_crew.run, ticker)
    except Exception as exc:
        logger.error("Analysis failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again later.") from exc

    latency_ms = int((time.time() - start_ts) * 1000)

    # Persist report
    try:
        tech = result.get("technical_signals", {})
        report = AnalysisReport(
            ticker=ticker,
            recommendation=result.get("recommendation", "HOLD"),
            confidence_score=result.get("confidence_score", 50.0),
            technical_summary=str(result.get("score_breakdown", {}).get("technical", "")),
            sentiment_summary=str(result.get("sentiment", {})),
            narrative=result.get("narrative", ""),
            rsi=tech.get("rsi"),
            macd=tech.get("macd"),
            ma_50=tech.get("sma_50"),
            ma_200=tech.get("sma_200"),
        )
        db.add(report)

        query_log = UserQuery(
            endpoint="/analyze-stock",
            ticker=ticker,
            query_payload=body.model_dump_json(),
            response_summary=f"{result.get('recommendation')} score={result.get('confidence_score')}",
            latency_ms=latency_ms,
        )
        db.add(query_log)
        db.commit()
    except Exception as exc:
        logger.warning("Failed to persist analysis report: %s", exc)

    return result


@router.get("/stock/{ticker}", response_model=Dict[str, Any])
@limiter.limit("60/minute")
async def get_stock(request: Request, ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Return the latest cached analysis report + current market quote for *ticker*.

    If no cached report exists, returns the live quote data only.
    """
    ticker = ticker.upper()
    quote = await asyncio.to_thread(_mds.get_quote, ticker)

    # Look up latest persisted report
    report = (
        db.query(AnalysisReport)
        .filter(AnalysisReport.ticker == ticker)
        .order_by(AnalysisReport.created_at.desc())
        .first()
    )

    result: Dict[str, Any] = {"ticker": ticker, "quote": quote}

    if report:
        result["latest_report"] = {
            "recommendation": report.recommendation,
            "confidence_score": report.confidence_score,
            "rsi": report.rsi,
            "macd": report.macd,
            "ma_50": report.ma_50,
            "ma_200": report.ma_200,
            "narrative": report.narrative,
            "created_at": str(report.created_at),
        }

    return result
