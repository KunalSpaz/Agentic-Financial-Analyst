"""
test_api_routes.py
-------------------
Integration tests for all FastAPI routes using the HTTPX TestClient.

All external dependencies (CrewAI, yfinance, NewsAPI, FinBERT, SQLAlchemy)
are mocked so these tests run without network access or API keys.

Covers:
  GET  /                      — health check
  POST /analyze-stock         — returns 200 with required fields or 422 on bad input
  GET  /stock/{ticker}        — returns quote + optional cached report
  GET  /market-report         — returns report dict
  GET  /top-news              — returns list
  GET  /market-opportunities  — returns list from DB
  POST /backtest              — returns metrics + equity curve
  POST /portfolio-analysis    — returns risk metrics
  POST /optimize-strategy     — returns best_parameters
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── Patch heavy dependencies BEFORE importing the app ──────────────────────
# This prevents FinBERT from downloading on import and OpenAI from initialising.
import sys
sys.modules.setdefault("crewai",         MagicMock())
sys.modules.setdefault("crewai.agent",   MagicMock())
sys.modules.setdefault("crewai.task",    MagicMock())
sys.modules.setdefault("crewai.crew",    MagicMock())
sys.modules.setdefault("faiss",          MagicMock())
sys.modules.setdefault("transformers",   MagicMock())
sys.modules.setdefault("torch",          MagicMock())
sys.modules.setdefault("newsapi",        MagicMock())

from backend.api.main import create_app  # noqa: E402  (must come after mocks)
from backend.database.connection import Base, engine  # noqa: E402


@pytest.fixture(scope="module")
def app():
    """Create a fresh app instance with an in-memory SQLite DB."""
    with patch("backend.api.main.run_migrations"):
        with patch("backend.api.main.start_scheduler"):
            with patch("backend.api.main.shutdown_scheduler"):
                application = create_app()
                # Create tables in the test database
                Base.metadata.create_all(bind=engine)
                yield application
                Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def client(app):
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# GET / — health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthCheck:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        resp = client.get("/")
        assert resp.json()["status"] == "ok"

    def test_returns_service_name(self, client):
        resp = client.get("/")
        assert "Autonomous Financial Analyst" in resp.json()["service"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-stock
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeStock:
    def test_returns_200_with_mocked_crew(self, client):
        mock_result = {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "current_price": 175.0,
            "recommendation": "BUY",
            "confidence_score": 72.5,
            "confidence_color": "#69F0AE",
            "technical_signals": {"rsi": 42.0},
            "sentiment": {"label": "POSITIVE", "compound": 0.3, "article_count": 10},
            "score_breakdown": {},
            "narrative": "AI narrative here.",
            "quote": {"price": 175.0, "sector": "Technology"},
            "rag_context_used": False,
        }
        with patch("backend.api.routes.stock_routes._crew") as mock_crew:
            mock_crew.run.return_value = mock_result
            resp = client.post("/analyze-stock", json={"ticker": "AAPL"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ticker"] == "AAPL"
        assert data["recommendation"] == "BUY"
        assert data["confidence_score"] == 72.5

    def test_missing_ticker_returns_422(self, client):
        resp = client.post("/analyze-stock", json={})
        assert resp.status_code == 422

    def test_crew_error_returns_500(self, client):
        with patch("backend.api.routes.stock_routes._crew") as mock_crew:
            mock_crew.run.side_effect = RuntimeError("CrewAI failed")
            resp = client.post("/analyze-stock", json={"ticker": "AAPL"})
        assert resp.status_code == 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /stock/{ticker}
# ─────────────────────────────────────────────────────────────────────────────

class TestGetStock:
    def test_returns_200_with_quote(self, client):
        mock_quote = {"ticker": "MSFT", "price": 380.0, "sector": "Technology"}
        with patch("backend.api.routes.stock_routes._mds") as mock_mds:
            mock_mds.get_quote.return_value = mock_quote
            resp = client.get("/stock/MSFT")
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "MSFT"

    def test_ticker_is_uppercased(self, client):
        mock_quote = {"ticker": "NVDA", "price": 500.0}
        with patch("backend.api.routes.stock_routes._mds") as mock_mds:
            mock_mds.get_quote.return_value = mock_quote
            resp = client.get("/stock/nvda")
        assert resp.status_code == 200
        assert resp.json()["ticker"] == "NVDA"


# ─────────────────────────────────────────────────────────────────────────────
# GET /top-news
# ─────────────────────────────────────────────────────────────────────────────

class TestTopNews:
    def test_returns_200_and_list(self, client):
        mock_articles = [
            {"ticker": None, "title": "Market rallies", "description": "Up 2%",
             "content": "", "url": "", "source": "Reuters", "author": "", "published_at": ""}
        ]
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = mock_articles
            resp = client.get("/top-news")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_limit_param_passed(self, client):
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = []
            resp = client.get("/top-news?limit=5")
        assert resp.status_code == 200
        mock_ns.get_top_financial_news.assert_called_once_with(max_articles=5)


# ─────────────────────────────────────────────────────────────────────────────
# GET /market-opportunities
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketOpportunities:
    def test_returns_200_and_list(self, client):
        resp = client.get("/market-opportunities")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ─────────────────────────────────────────────────────────────────────────────
# POST /backtest
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktest:
    def _mock_backtest_result(self):
        from backend.services.backtesting_service import BacktestResult
        return BacktestResult(
            ticker="AAPL", strategy_name="RSI_MACD_Sentiment",
            start_date="2022-01-01", end_date="2024-01-01",
            parameters={"rsi_buy_threshold": 35, "rsi_sell_threshold": 70,
                        "macd_confirmation": True, "ma_filter": True, "initial_capital": 10000},
            total_return=0.15, sharpe_ratio=1.3, max_drawdown=0.08,
            win_rate=0.55, num_trades=12,
            equity_curve=[10000, 10500, 11000, 11500],
            trade_log=[{"date": "2022-06-01", "action": "BUY", "price": 150.0, "shares": 10, "pnl": 0, "capital": 8500}],
        )

    def test_returns_200_with_metrics(self, client):
        with patch("backend.api.routes.backtest_routes._bt") as mock_bt:
            mock_bt.run_backtest.return_value = self._mock_backtest_result()
            resp = client.post("/backtest", json={"ticker": "AAPL"})
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        assert "equity_curve" in data
        assert "trade_log" in data

    def test_metrics_contain_required_fields(self, client):
        with patch("backend.api.routes.backtest_routes._bt") as mock_bt:
            mock_bt.run_backtest.return_value = self._mock_backtest_result()
            resp = client.post("/backtest", json={"ticker": "AAPL"})
        metrics = resp.json()["metrics"]
        for key in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate", "num_trades"):
            assert key in metrics

    def test_missing_ticker_returns_422(self, client):
        resp = client.post("/backtest", json={})
        assert resp.status_code == 422

    def test_invalid_sentiment_returns_422(self, client):
        resp = client.post("/backtest", json={"ticker": "AAPL", "sentiment_label": "VERY_BULLISH"})
        assert resp.status_code == 422

    def test_value_error_returns_422(self, client):
        with patch("backend.api.routes.backtest_routes._bt") as mock_bt:
            mock_bt.run_backtest.side_effect = ValueError("insufficient data")
            resp = client.post("/backtest", json={"ticker": "AAPL"})
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /portfolio-analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioAnalysis:
    MOCK_RISK_RESULT = {
        "holdings": {"AAPL": 0.6, "MSFT": 0.4},
        "portfolio_volatility": 0.18,
        "portfolio_annual_return": 0.12,
        "portfolio_beta": 1.1,
        "sharpe_ratio": 0.56,
        "max_drawdown": 0.22,
        "var_95_daily": -0.021,
        "correlation_matrix": {"AAPL": {"AAPL": 1.0, "MSFT": 0.7}, "MSFT": {"AAPL": 0.7, "MSFT": 1.0}},
        "sector_exposure": {"Technology": 1.0},
    }

    def test_returns_200_with_risk_metrics(self, client):
        with patch("backend.api.routes.portfolio_routes._prs") as mock_prs:
            mock_prs.analyse.return_value = self.MOCK_RISK_RESULT
            resp = client.post(
                "/portfolio-analysis",
                json={"holdings": {"AAPL": 0.6, "MSFT": 0.4}},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "portfolio_volatility" in data
        assert "sharpe_ratio" in data

    def test_empty_holdings_returns_400(self, client):
        resp = client.post("/portfolio-analysis", json={"holdings": {}})
        assert resp.status_code == 400

    def test_error_from_service_returns_422(self, client):
        with patch("backend.api.routes.portfolio_routes._prs") as mock_prs:
            mock_prs.analyse.return_value = {"error": "No data"}
            resp = client.post("/portfolio-analysis", json={"holdings": {"AAPL": 1.0}})
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# POST /optimize-strategy
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizeStrategy:
    MOCK_OPT_RESULT = {
        "ticker": "AAPL",
        "objective": "maximize_return",
        "best_parameters": {"rsi_buy_threshold": 35, "rsi_sell_threshold": 70, "macd_confirmation": True, "ma_filter": True},
        "best_return": 0.22,
        "best_sharpe": 1.4,
        "best_drawdown": 0.12,
        "iterations": 40,
        "all_results": [],
    }

    def test_returns_200_with_best_parameters(self, client):
        with patch("backend.api.routes.optimization_routes._opt") as mock_opt:
            mock_opt.optimize.return_value = self.MOCK_OPT_RESULT
            resp = client.post(
                "/optimize-strategy",
                json={"ticker": "AAPL", "objective": "maximize_return"},
            )
        assert resp.status_code == 200
        assert "best_parameters" in resp.json()

    def test_invalid_objective_returns_422(self, client):
        resp = client.post(
            "/optimize-strategy",
            json={"ticker": "AAPL", "objective": "maximize_alpha"},
        )
        assert resp.status_code == 422

    def test_missing_ticker_returns_422(self, client):
        resp = client.post("/optimize-strategy", json={"objective": "maximize_return"})
        assert resp.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# API key middleware
# ─────────────────────────────────────────────────────────────────────────────

class TestAPIKeyMiddleware:
    """Tests for the X-API-Key authentication middleware (Fix 2)."""

    def test_no_key_configured_allows_all_requests(self, client):
        """When API_SECRET_KEY is blank, all requests pass through."""
        with patch("backend.api.routes.stock_routes._mds") as mock_mds:
            mock_mds.get_quote.return_value = {"ticker": "AAPL", "price": 175.0}
            resp = client.get("/stock/AAPL")
        assert resp.status_code == 200

    def test_correct_key_allows_request(self, app):
        """When API_SECRET_KEY is set and correct X-API-Key is provided, request passes."""
        with patch("backend.api.main.settings") as mock_settings:
            mock_settings.api_secret_key = "my-secret"
            with TestClient(app) as c:
                with patch("backend.api.routes.stock_routes._mds") as mock_mds:
                    mock_mds.get_quote.return_value = {"ticker": "AAPL", "price": 175.0}
                    resp = c.get("/stock/AAPL", headers={"X-API-Key": "my-secret"})
            assert resp.status_code == 200

    def test_wrong_key_returns_401(self, app):
        """When API_SECRET_KEY is set and wrong key provided, returns 401."""
        with patch("backend.api.main.settings") as mock_settings:
            mock_settings.api_secret_key = "correct-key"
            with TestClient(app) as c:
                resp = c.get("/stock/AAPL", headers={"X-API-Key": "wrong-key"})
            assert resp.status_code == 401

    def test_missing_key_header_returns_401(self, app):
        """When API_SECRET_KEY is set and no header provided, returns 401."""
        with patch("backend.api.main.settings") as mock_settings:
            mock_settings.api_secret_key = "correct-key"
            with TestClient(app) as c:
                resp = c.get("/stock/AAPL")
            assert resp.status_code == 401

    def test_health_check_bypasses_auth(self, app):
        """GET / is in public paths and never requires a key."""
        with patch("backend.api.main.settings") as mock_settings:
            mock_settings.api_secret_key = "super-secret"
            with TestClient(app) as c:
                resp = c.get("/")
            assert resp.status_code == 200

    def test_docs_bypass_auth(self, app):
        """GET /docs is public — should not require X-API-Key."""
        with patch("backend.api.main.settings") as mock_settings:
            mock_settings.api_secret_key = "super-secret"
            with TestClient(app) as c:
                resp = c.get("/docs")
            assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# Security response headers
# ─────────────────────────────────────────────────────────────────────────────

class TestSecurityHeaders:
    """Every response from the app should carry standard security headers."""

    def test_x_content_type_options_present(self, client):
        resp = client.get("/")
        assert resp.headers.get("x-content-type-options") == "nosniff"

    def test_x_frame_options_present(self, client):
        resp = client.get("/")
        assert resp.headers.get("x-frame-options") == "DENY"

    def test_referrer_policy_present(self, client):
        resp = client.get("/")
        assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"

    def test_security_headers_on_json_endpoint(self, client):
        """Security headers should be added to non-health endpoints too."""
        resp = client.get("/top-news")
        with patch("backend.api.routes.market_routes._ns") as mock_ns:
            mock_ns.get_top_financial_news.return_value = []
            resp = client.get("/top-news")
        assert "x-content-type-options" in resp.headers
        assert "x-frame-options" in resp.headers

    def test_x_frame_options_is_deny_not_sameorigin(self, client):
        """Stricter DENY value expected — not SAMEORIGIN."""
        resp = client.get("/")
        assert resp.headers.get("x-frame-options").upper() == "DENY"
