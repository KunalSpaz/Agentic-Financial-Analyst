"""
scheduler.py
------------
APScheduler configuration and daily job definitions.

Jobs:
    1. daily_market_scan         — Scans the stock universe for opportunities
    2. daily_opportunity_update  — Persists opportunity rankings to the database
    3. daily_market_briefing     — Generates the AI daily market report

All jobs run daily at the configured hour (default: 8:00 AM local time).
"""
from __future__ import annotations

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_scheduler: BackgroundScheduler | None = None


def _job_daily_market_scan() -> None:
    """
    Daily job: Scan the stock universe and persist market opportunities.
    """
    logger.info("Scheduler: Starting daily market scan…")
    try:
        from backend.agents.crew_orchestrator import MarketScanCrew
        from backend.database.connection import db_session
        from backend.models.market_opportunity import MarketOpportunity

        crew = MarketScanCrew()
        opportunities = crew.scan()

        with db_session() as db:
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

        logger.info("Scheduler: Daily market scan complete. %d opportunities found.", len(opportunities))
    except Exception as exc:
        logger.error("Scheduler: Daily market scan failed: %s", exc)


def _job_daily_market_briefing() -> None:
    """
    Daily job: Generate and persist the AI daily market briefing.
    """
    logger.info("Scheduler: Generating daily AI market briefing…")
    try:
        from backend.database.connection import db_session
        from backend.models.market_opportunity import MarketOpportunity
        from backend.services.market_data_service import MarketDataService
        from backend.services.news_service import NewsService
        from backend.services.report_service import ReportService
        from backend.services.sentiment_service import SentimentService

        mds = MarketDataService()
        ns = NewsService()
        rs = ReportService()
        ss = SentimentService()

        indices = mds.get_market_index_data()
        news = ns.get_top_financial_news(max_articles=15)

        with db_session() as db:
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
                }
                for o in db_opps
            ]

        if news:
            sentiment = ss.analyse_articles(news[:20])
            overall_sentiment = sentiment.get("label", "NEUTRAL")
        else:
            overall_sentiment = "NEUTRAL"

        report = rs.generate_daily_briefing(
            opportunities=opportunities,
            market_indices=indices,
            top_news=news,
            overall_sentiment=overall_sentiment,
        )
        logger.info("Scheduler: Daily briefing generated for %s", report.get("date"))
    except Exception as exc:
        logger.error("Scheduler: Daily briefing failed: %s", exc)


def start_scheduler() -> None:
    """
    Create and start the background APScheduler with all daily jobs.

    Jobs fire at ``settings.daily_report_hour:settings.daily_report_minute`` local time.
    """
    global _scheduler

    if _scheduler is not None and _scheduler.running:
        logger.warning("Scheduler already running; skipping start.")
        return

    _scheduler = BackgroundScheduler(timezone="America/New_York")

    trigger = CronTrigger(
        hour=settings.daily_report_hour,
        minute=settings.daily_report_minute,
        timezone="America/New_York",
    )

    _scheduler.add_job(
        _job_daily_market_scan,
        trigger=trigger,
        id="daily_market_scan",
        name="Daily Market Scan",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _briefing_minute = (settings.daily_report_minute + 30) % 60
    _briefing_hour = (settings.daily_report_hour + (1 if settings.daily_report_minute >= 30 else 0)) % 24

    _scheduler.add_job(
        _job_daily_market_briefing,
        trigger=CronTrigger(
            hour=_briefing_hour,
            minute=_briefing_minute,
            timezone="America/New_York",
        ),
        id="daily_market_briefing",
        name="Daily AI Market Briefing",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    _scheduler.start()
    logger.info(
        "Scheduler started. Jobs will fire at %02d:%02d ET.",
        settings.daily_report_hour,
        settings.daily_report_minute,
    )


def shutdown_scheduler() -> None:
    """Gracefully shut down the APScheduler."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shut down.")
