"""
market_routes.py
----------------
FastAPI routes for market-wide data.

Endpoints:
    GET /market-report          — Latest daily AI market briefing
    GET /top-news               — Top financial headlines
    GET /market-opportunities   — Ranked investment opportunities
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.agents.analysis_graph import MarketScanGraph
from backend.api.rate_limit import limiter
from backend.database.connection import get_db
from backend.models.market_opportunity import MarketOpportunity
from backend.services.market_data_service import MarketDataService
from backend.services.news_service import NewsService
from backend.services.report_service import ReportService
from backend.services.sentiment_service import SentimentService
from backend.utils.config import settings
from backend.utils.logger import get_logger

router = APIRouter(tags=["Market"])
logger = get_logger(__name__)

_mds = MarketDataService()
_ns = NewsService()
_rs = ReportService()
_ss = SentimentService()


@router.get("/market-report", response_model=Dict[str, Any])
@limiter.limit("10/minute")
async def get_market_report(request: Request, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Return the latest AI-generated daily market briefing.

    Generates a fresh briefing using current opportunity data, market indices,
    top news, and overall sentiment.
    """
    # Gather inputs concurrently — both are blocking I/O calls
    indices, news = await asyncio.gather(
        asyncio.to_thread(_mds.get_market_index_data),
        asyncio.to_thread(_ns.get_top_financial_news, max_articles=10),
    )

    # Get top opportunities from DB (latest scan)
    db_opps = (
        db.query(MarketOpportunity)
        .order_by(MarketOpportunity.scan_date.desc(), MarketOpportunity.rank)
        .limit(10)
        .all()
    )
    opportunities = [
        {
            "ticker": o.ticker,
            "recommendation": o.recommendation,
            "confidence_score": o.confidence_score,
            "sector": o.sector,
        }
        for o in db_opps
    ]

    # Aggregate sentiment
    if news:
        agg_sentiment = await asyncio.to_thread(_ss.analyse_articles, news[:20])
        overall_sentiment = agg_sentiment.get("label", "NEUTRAL")
    else:
        overall_sentiment = "NEUTRAL"

    report = await asyncio.to_thread(
        _rs.generate_daily_briefing,
        opportunities=opportunities,
        market_indices=indices,
        top_news=news,
        overall_sentiment=overall_sentiment,
    )
    return report


@router.get("/top-news", response_model=List[Dict[str, Any]])
@limiter.limit("30/minute")
async def get_top_news(request: Request, limit: int = 20) -> List[Dict[str, Any]]:
    """Return the latest top financial headlines (up to *limit* articles)."""
    articles = await asyncio.to_thread(_ns.get_top_financial_news, max_articles=min(limit, 50))
    return articles


@router.get("/market-opportunities", response_model=Dict[str, Any])
@limiter.limit("3/minute")  # refresh=true can trigger up to 10 full LLM analysis pipelines
async def get_market_opportunities(
    request: Request,
    refresh: bool = False,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Return ranked investment opportunities from the last market scan.

    Args:
        refresh: If ``true``, re-run the full opportunity scan before returning.
                 This can take several minutes and triggers up to
                 ``len(stock_universe)`` full LLM analysis pipelines, so it is
                 rate-limited more tightly than a cached read (see the
                 ``@limiter.limit`` decorator below).

    Returns:
        Dict with ``opportunities`` (ranked list) and ``market_narrative``
        (thematic summary — only populated on a fresh ``refresh=true`` scan;
        cached reads don't persist the narrative, so it's empty there).
    """
    if refresh:
        logger.info("Refreshing market opportunities scan…")
        graph = MarketScanGraph()
        try:
            scan_result = await asyncio.to_thread(graph.scan)
        except Exception as exc:
            logger.error("Market scan failed: %s", exc)
            raise HTTPException(status_code=500, detail="Market scan failed. Please try again later.") from exc

        opportunities = scan_result["opportunities"]
        market_narrative = scan_result.get("market_narrative", "")

        # Persist
        try:
            for opp in opportunities:
                db_opp = MarketOpportunity(
                    ticker=opp["ticker"],
                    rank=opp["rank"],
                    recommendation=opp["recommendation"],
                    confidence_score=opp["confidence_score"],
                    current_price=opp.get("current_price"),
                    sector=opp.get("sector"),
                    rationale=opp.get("rationale", ""),
                )
                db.add(db_opp)
            db.flush()
        except Exception as exc:
            db.rollback()
            logger.warning("Failed to persist opportunities: %s", exc)

        return {"opportunities": opportunities, "market_narrative": market_narrative}

    # Return cached results
    db_opps = (
        db.query(MarketOpportunity)
        .order_by(MarketOpportunity.scan_date.desc(), MarketOpportunity.rank)
        .limit(20)
        .all()
    )

    return {
        "opportunities": [
            {
                "rank": o.rank,
                "ticker": o.ticker,
                "recommendation": o.recommendation,
                "confidence_score": o.confidence_score,
                "current_price": o.current_price,
                "sector": o.sector,
                "rationale": o.rationale,
                "scan_date": str(o.scan_date),
            }
            for o in db_opps
        ],
        "market_narrative": "",
    }
