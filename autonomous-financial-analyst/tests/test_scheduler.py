"""
test_scheduler.py
-----------------
Unit tests for APScheduler configuration and daily job logic.

Covers:
  start_scheduler():
    - Initialises BackgroundScheduler
    - Adds two jobs (market scan + briefing)
    - Does not start twice if already running

  shutdown_scheduler():
    - Calls scheduler.shutdown()
    - Handles gracefully when not running

  _job_daily_market_scan():
    - Calls MarketScanCrew.scan()
    - Persists each opportunity to DB
    - Logs and continues on exception

  _job_daily_market_briefing():
    - Fetches top opportunities from DB
    - Calls generate_daily_briefing
    - Logs and continues on exception
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())
sys.modules.setdefault("crewai",       MagicMock())

import backend.utils.scheduler as scheduler_mod  # noqa: E402


@pytest.fixture(autouse=True)
def reset_scheduler():
    """Ensure no scheduler leaks between tests."""
    scheduler_mod._scheduler = None
    yield
    if scheduler_mod._scheduler and scheduler_mod._scheduler.running:
        scheduler_mod._scheduler.shutdown(wait=False)
    scheduler_mod._scheduler = None


# ─────────────────────────────────────────────────────────────────────────────
# start_scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestStartScheduler:
    def test_creates_background_scheduler(self):
        with patch("backend.utils.scheduler.BackgroundScheduler") as mock_cls:
            mock_instance = MagicMock(); mock_instance.running = False
            mock_cls.return_value = mock_instance
            scheduler_mod.start_scheduler()
        mock_cls.assert_called_once_with(timezone="America/New_York")

    def test_starts_the_scheduler(self):
        with patch("backend.utils.scheduler.BackgroundScheduler") as mock_cls:
            mock_instance = MagicMock(); mock_instance.running = False
            mock_cls.return_value = mock_instance
            scheduler_mod.start_scheduler()
        mock_instance.start.assert_called_once()

    def test_adds_two_jobs(self):
        with patch("backend.utils.scheduler.BackgroundScheduler") as mock_cls:
            mock_instance = MagicMock(); mock_instance.running = False
            mock_cls.return_value = mock_instance
            scheduler_mod.start_scheduler()
        assert mock_instance.add_job.call_count == 2

    def test_jobs_have_misfire_grace_time(self):
        with patch("backend.utils.scheduler.BackgroundScheduler") as mock_cls:
            mock_instance = MagicMock(); mock_instance.running = False
            mock_cls.return_value = mock_instance
            scheduler_mod.start_scheduler()
        for call_args in mock_instance.add_job.call_args_list:
            kwargs = call_args[1]
            assert kwargs.get("misfire_grace_time") == 3600

    def test_does_not_start_twice_when_already_running(self):
        with patch("backend.utils.scheduler.BackgroundScheduler") as mock_cls:
            mock_instance = MagicMock(); mock_instance.running = True
            mock_cls.return_value = mock_instance
            scheduler_mod._scheduler = mock_instance  # simulate already running
            scheduler_mod.start_scheduler()
        mock_instance.start.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# shutdown_scheduler
# ─────────────────────────────────────────────────────────────────────────────

class TestShutdownScheduler:
    def test_calls_shutdown_on_running_scheduler(self):
        mock_instance = MagicMock(); mock_instance.running = True
        scheduler_mod._scheduler = mock_instance
        scheduler_mod.shutdown_scheduler()
        mock_instance.shutdown.assert_called_once_with(wait=False)

    def test_sets_scheduler_to_none_after_shutdown(self):
        mock_instance = MagicMock(); mock_instance.running = True
        scheduler_mod._scheduler = mock_instance
        scheduler_mod.shutdown_scheduler()
        assert scheduler_mod._scheduler is None

    def test_no_error_when_scheduler_not_running(self):
        scheduler_mod._scheduler = None
        scheduler_mod.shutdown_scheduler()  # should not raise

    def test_no_error_when_scheduler_stopped(self):
        mock_instance = MagicMock(); mock_instance.running = False
        scheduler_mod._scheduler = mock_instance
        scheduler_mod.shutdown_scheduler()  # should not raise
        mock_instance.shutdown.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# _job_daily_market_scan
# ─────────────────────────────────────────────────────────────────────────────

_MOCK_OPPORTUNITIES = [
    {"ticker": "AAPL", "rank": 1, "recommendation": "BUY",
     "confidence_score": 78.0, "current_price": 175.0, "sector": "Technology", "rationale": "Strong"},
    {"ticker": "MSFT", "rank": 2, "recommendation": "HOLD",
     "confidence_score": 65.0, "current_price": 380.0, "sector": "Technology", "rationale": "Steady"},
]


class TestJobDailyMarketScan:
    def test_calls_market_scan_crew(self):
        with patch("backend.agents.crew_orchestrator.MarketScanCrew") as mock_crew_cls, \
             patch("backend.database.connection.db_session") as mock_db_ctx, \
             patch("backend.models.market_opportunity.MarketOpportunity"):
            mock_crew = MagicMock()
            mock_crew.scan.return_value = []
            mock_crew_cls.return_value = mock_crew
            mock_db_ctx.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)
            scheduler_mod._job_daily_market_scan()
        mock_crew.scan.assert_called_once()

    def test_persists_each_opportunity(self):
        mock_db = MagicMock()
        with patch("backend.agents.crew_orchestrator.MarketScanCrew") as mock_crew_cls, \
             patch("backend.database.connection.db_session") as mock_db_ctx, \
             patch("backend.models.market_opportunity.MarketOpportunity") as mock_opp_cls:
            mock_crew = MagicMock()
            mock_crew.scan.return_value = _MOCK_OPPORTUNITIES
            mock_crew_cls.return_value = mock_crew
            mock_db_ctx.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)
            scheduler_mod._job_daily_market_scan()
        assert mock_db.add.call_count == len(_MOCK_OPPORTUNITIES)

    def test_exception_logged_not_raised(self):
        with patch("backend.agents.crew_orchestrator.MarketScanCrew") as mock_crew_cls:
            mock_crew_cls.return_value.scan.side_effect = RuntimeError("network down")
            scheduler_mod._job_daily_market_scan()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# _job_daily_market_briefing
# ─────────────────────────────────────────────────────────────────────────────

class TestJobDailyMarketBriefing:
    def test_generates_briefing(self):
        mock_db = MagicMock()
        mock_db.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
        with patch("backend.database.connection.db_session") as mock_db_ctx, \
             patch("backend.services.market_data_service.MarketDataService") as mock_mds_cls, \
             patch("backend.services.news_service.NewsService") as mock_ns_cls, \
             patch("backend.services.report_service.ReportService") as mock_rs_cls, \
             patch("backend.services.sentiment_service.SentimentService") as mock_ss_cls:
            mock_db_ctx.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_db_ctx.return_value.__exit__ = MagicMock(return_value=False)
            mock_mds_cls.return_value.get_market_index_data.return_value = {}
            mock_ns_cls.return_value.get_top_financial_news.return_value = []
            mock_rs_cls.return_value.generate_daily_briefing.return_value = {
                "date": "2026-03-21", "narrative": "Positive day."
            }
            mock_ss_cls.return_value.analyse_articles.return_value = {"label": "NEUTRAL"}
            scheduler_mod._job_daily_market_briefing()
        mock_rs_cls.return_value.generate_daily_briefing.assert_called_once()

    def test_exception_logged_not_raised(self):
        with patch("backend.database.connection.db_session") as mock_db_ctx:
            mock_db_ctx.side_effect = RuntimeError("db down")
            scheduler_mod._job_daily_market_briefing()  # must not raise
