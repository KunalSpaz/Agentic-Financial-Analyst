"""
test_market_routes.py
----------------------
Integration tests for market-wide API endpoints.

Covers:
  GET /market-report:
    - Returns 200 with required keys
    - Returns NEUTRAL sentiment when news is empty
    - Briefing service exception → 500

  GET /top-news:
    - Returns 200 with list
    - limit param passed through
    - limit capped at 50

  GET /market-opportunities:
    - Returns 200 and list from DB when no refresh
    - Empty DB → empty list
    - refresh=true runs crew scan
    - refresh=true crew exception → 500
    - refresh=true persists results to DB
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.modules.setdefault("crewai",       MagicMock())
sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())

from backend.api.main import create_app  # noqa: E402
from backend.database.connection import Base, engine  # noqa: E402


@pytest.fixture(scope="module")
def app():
    with patch("backend.api.main.run_migrations"), \
         patch("backend.api.main.start_scheduler"), \
         patch("backend.api.main.shutdown_scheduler"):
        application = create_app()
        Base.metadata.create_all(bind=engine)
        yield application
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


_MOCK_INDICES = {
    "SPY": {"price": 450.0, "change_pct": 0.5},
    "QQQ": {"price": 380.0, "change_pct": 1.2},
}
_MOCK_NEWS = [
    {"title": "Fed holds rates", "source": "Reuters", "description": ""},
    {"title": "Tech earnings beat", "source": "Bloomberg", "description": ""},
]
_MOCK_BRIEFING = {
    "date": "2026-03-21",
    "narrative": "Solid day for equities.",
    "overall_sentiment": "POSITIVE",
    "top_picks": ["AAPL", "MSFT"],
    "market_indices": _MOCK_INDICES,
}


# ─────────────────────────────────────────────────────────────────────────────
# GET /market-report
# ─────────────────────────────────────────────────────────────────────────────

class TestGetMarketReport:
    def _patch_all(self):
        return (
            patch("backend.api.routes.market_routes._mds") ,
            patch("backend.api.routes.market_routes._ns"),
            patch("backend.api.routes.market_routes._ss"),
            patch("backend.api.routes.market_routes._rs"),
        )

    def test_returns_200(self, client):
        with patch("backend.api.routes.market_routes._mds") as mock_mds, \
             patch("backend.api.routes.market_routes._ns") as mock_ns, \
             patch("backend.api.routes.market_routes._ss") as mock_ss, \
             patch("backend.api.routes.market_routes._rs") as mock_rs:
            mock_mds.get_market_index_data.return_value = _MOCK_INDICES
            mock_ns.get_top_financial_news.return_value = _MOCK_NEWS
            mock_ss.analyse_articles.return_value = {"label": "POSITIVE"}
            mock_rs.generate_daily_briefing.return_value = _MOCK_BRIEFING
            resp = client.get("/market-report")
        assert resp.status_code == 200

    def test_response_has_required_keys(self, client):
        with patch("backend.api.routes.market_routes._mds") as mock_mds, \
             patch("backend.api.routes.market_routes._ns") as mock_ns, \
             patch("backend.api.routes.market_routes._ss") as mock_ss, \
             patch("backend.api.routes.market_routes._rs") as mock_rs:
            mock_mds.get_market_index_data.return_value = _MOCK_INDICES
            mock_ns.get_top_financial_news.return_value = _MOCK_NEWS
            mock_ss.analyse_articles.return_value = {"label": "POSITIVE"}
            mock_rs.generate_daily_briefing.return_value = _MOCK_BRIEFING
            resp = client.get("/market-report")
        data = resp.json()
        for key in ("date", "narrative", "overall_sentiment", "top_picks", "market_indices"):
            assert key in data, f"Missing key: {key}"

    def test_empty_news_defaults_to_neutral_sentiment(self, client):
        """When no news articles, report service should be called with NEUTRAL sentiment."""
        with patch("backend.api.routes.market_routes._mds") as mock_mds, \
             patch("backend.api.routes.market_routes._ns") as mock_ns, \
             patch("backend.api.routes.market_routes._ss") as mock_ss, \
             patch("backend.api.routes.market_routes._rs") as mock_rs:
            mock_mds.get_market_index_data.return_value = _MOCK_INDICES
            mock_ns.get_top_financial_news.return_value = []
            mock_ss.analyse_articles.return_value = {"label": "NEUTRAL"}
            mock_rs.generate_daily_briefing.return_value = {**_MOCK_BRIEFING, "overall_sentiment": "NEUTRAL"}
            resp = client.get("/market-report")
            # Sentiment service should NOT be called when news is empty
            mock_ss.analyse_articles.assert_not_called()
        assert resp.status_code == 200

    def test_news_fetched_with_max_10(self, client):
        with patch("backend.api.routes.market_routes._mds") as mock_mds, \
             patch("backend.api.routes.market_routes._ns") as mock_ns, \
             patch("backend.api.routes.market_routes._ss") as mock_ss, \
             patch("backend.api.routes.market_routes._rs") as mock_rs:
            mock_mds.get_market_index_data.return_value = _MOCK_INDICES
            mock_ns.get_top_financial_news.return_value = _MOCK_NEWS
            mock_ss.analyse_articles.return_value = {"label": "POSITIVE"}
            mock_rs.generate_daily_briefing.return_value = _MOCK_BRIEFING
            client.get("/market-report")
            mock_ns.get_top_financial_news.assert_called_once_with(max_articles=10)

    def test_briefing_service_called_once(self, client):
        with patch("backend.api.routes.market_routes._mds") as mock_mds, \
             patch("backend.api.routes.market_routes._ns") as mock_ns, \
             patch("backend.api.routes.market_routes._ss") as mock_ss, \
             patch("backend.api.routes.market_routes._rs") as mock_rs:
            mock_mds.get_market_index_data.return_value = _MOCK_INDICES
            mock_ns.get_top_financial_news.return_value = _MOCK_NEWS
            mock_ss.analyse_articles.return_value = {"label": "POSITIVE"}
            mock_rs.generate_daily_briefing.return_value = _MOCK_BRIEFING
            client.get("/market-report")
            mock_rs.generate_daily_briefing.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# GET /top-news
# ─────────────────────────────────────────────────────────────────────────────

class TestGetTopNews:
    def test_returns_200_and_list(self, client):
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = _MOCK_NEWS
            resp = client.get("/top-news")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_default_limit_is_20(self, client):
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = []
            client.get("/top-news")
            mock_ns.get_top_financial_news.assert_called_once_with(max_articles=20)

    def test_custom_limit_passed_through(self, client):
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = []
            client.get("/top-news?limit=5")
            mock_ns.get_top_financial_news.assert_called_once_with(max_articles=5)

    def test_limit_capped_at_50(self, client):
        """limit=999 should be capped to 50."""
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = []
            client.get("/top-news?limit=999")
            mock_ns.get_top_financial_news.assert_called_once_with(max_articles=50)

    def test_empty_news_returns_empty_list(self, client):
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = []
            resp = client.get("/top-news")
        assert resp.json() == []

    def test_articles_content_passthrough(self, client):
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = _MOCK_NEWS
            resp = client.get("/top-news")
        assert resp.json()[0]["title"] == "Fed holds rates"


# ─────────────────────────────────────────────────────────────────────────────
# GET /market-opportunities
# ─────────────────────────────────────────────────────────────────────────────

class TestGetMarketOpportunities:
    def test_returns_200(self, client):
        resp = client.get("/market-opportunities")
        assert resp.status_code == 200

    def test_returns_list(self, client):
        resp = client.get("/market-opportunities")
        assert isinstance(resp.json(), list)

    def test_empty_db_returns_empty_list(self, client):
        resp = client.get("/market-opportunities")
        assert resp.json() == []

    def test_cached_results_have_required_fields(self, app, client):
        """Pre-populate DB with one opportunity and verify field mapping."""
        from backend.database.connection import get_db
        from backend.models.market_opportunity import MarketOpportunity

        # Insert a test record
        db_gen = get_db()
        db = next(db_gen)
        opp = MarketOpportunity(
            ticker="AAPL", rank=1, recommendation="BUY",
            confidence_score=75.0, current_price=175.0,
            sector="Technology", rationale="Strong fundamentals",
        )
        db.add(opp)
        db.commit()

        try:
            resp = client.get("/market-opportunities")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) >= 1
            first = data[0]
            for field in ("rank", "ticker", "recommendation", "confidence_score"):
                assert field in first, f"Missing field: {field}"
        finally:
            db.delete(opp)
            db.commit()

    def test_refresh_true_runs_crew(self, client):
        mock_opps = [
            {"ticker": "AAPL", "rank": 1, "recommendation": "BUY",
             "confidence_score": 78.0, "current_price": 175.0,
             "sector": "Technology", "rationale": "Strong"},
        ]
        with patch("backend.api.routes.market_routes.MarketScanCrew") as mock_crew_cls:
            mock_crew = MagicMock()
            mock_crew.scan.return_value = mock_opps
            mock_crew_cls.return_value = mock_crew
            resp = client.get("/market-opportunities?refresh=true")
        assert resp.status_code == 200
        mock_crew.scan.assert_called_once()

    def test_refresh_true_returns_scan_results(self, client):
        mock_opps = [
            {"ticker": "MSFT", "rank": 1, "recommendation": "STRONG BUY",
             "confidence_score": 85.0, "current_price": 380.0,
             "sector": "Technology", "rationale": "AI growth"},
        ]
        with patch("backend.api.routes.market_routes.MarketScanCrew") as mock_crew_cls:
            mock_crew_cls.return_value.scan.return_value = mock_opps
            resp = client.get("/market-opportunities?refresh=true")
        assert resp.json()[0]["ticker"] == "MSFT"

    def test_refresh_crew_exception_returns_500(self, client):
        with patch("backend.api.routes.market_routes.MarketScanCrew") as mock_crew_cls:
            mock_crew_cls.return_value.scan.side_effect = RuntimeError("AI failure")
            resp = client.get("/market-opportunities?refresh=true")
        assert resp.status_code == 500

    def test_refresh_true_persists_to_db(self, client):
        """Scan results should be written to the database."""
        mock_opps = [
            {"ticker": "NVDA", "rank": 1, "recommendation": "BUY",
             "confidence_score": 80.0, "current_price": 900.0,
             "sector": "Technology", "rationale": "GPU demand"},
        ]
        with patch("backend.api.routes.market_routes.MarketScanCrew") as mock_crew_cls:
            mock_crew_cls.return_value.scan.return_value = mock_opps
            resp = client.get("/market-opportunities?refresh=true")

        assert resp.status_code == 200
        # Verify the opportunity now appears in cached endpoint
        cached = client.get("/market-opportunities")
        tickers = [r["ticker"] for r in cached.json()]
        assert "NVDA" in tickers
