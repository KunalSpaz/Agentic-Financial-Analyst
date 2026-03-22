"""
test_portfolio_risk_service.py
--------------------------------
Unit tests for PortfolioRiskService.

Covers:
  - analyse returns all required keys
  - portfolio_volatility is a positive float
  - weights are normalised correctly (sum-to-1 safe)
  - single-ticker portfolio still works
  - error returned when no market data available
  - correlation_matrix contains expected tickers
  - sector_exposure weights sum to approximately 1
  - max_drawdown is non-negative
  - beta computation falls back to 1.0 on failure
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.services.portfolio_risk_service import PortfolioRiskService


def _make_returns_df(tickers: list[str], n: int = 252) -> pd.DataFrame:
    """Generate synthetic daily return DataFrame."""
    np.random.seed(0)
    data = {t: np.random.normal(0.0005, 0.01, n) for t in tickers}
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(data, index=idx)


def _returns_to_prices(returns: pd.Series) -> pd.DataFrame:
    """Convert a returns series back to a Close price DataFrame."""
    prices = (1 + returns).cumprod() * 100
    return pd.DataFrame({"Close": prices.values}, index=returns.index)


@pytest.fixture
def service() -> PortfolioRiskService:
    return PortfolioRiskService()


@pytest.fixture
def two_ticker_setup(monkeypatch, service):
    """
    Patches MarketDataService so get_historical_data returns synthetic data
    and get_quote returns sector info.
    """
    returns_df = _make_returns_df(["AAPL", "MSFT"])

    def mock_history(ticker, period="1y"):
        if ticker in ("AAPL", "MSFT"):
            return _returns_to_prices(returns_df[ticker])
        if ticker == "SPY":
            spy_returns = pd.Series(np.random.normal(0.0004, 0.008, len(returns_df)))
            return _returns_to_prices(spy_returns)
        return pd.DataFrame()

    mock_quote = MagicMock(return_value={"sector": "Technology"})
    monkeypatch.setattr(service._mds, "get_historical_data", mock_history)
    monkeypatch.setattr(service._mds, "get_quote", mock_quote)
    return service


# ─────────────────────────────────────────────────────────────────────────────
# Result structure
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyseResultStructure:
    REQUIRED_KEYS = {
        "holdings", "portfolio_volatility", "portfolio_annual_return",
        "portfolio_beta", "sharpe_ratio", "max_drawdown",
        "var_95_daily", "correlation_matrix", "sector_exposure",
    }

    def test_all_required_keys_present(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.6, "MSFT": 0.4})
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_no_error_key_in_normal_run(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.6, "MSFT": 0.4})
        assert "error" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Metric sanity checks
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricValues:
    def test_portfolio_volatility_is_positive(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.6, "MSFT": 0.4})
        assert result["portfolio_volatility"] > 0

    def test_max_drawdown_is_non_negative(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.5, "MSFT": 0.5})
        assert result["max_drawdown"] >= 0

    def test_var_95_is_negative_float(self, two_ticker_setup):
        """Daily VaR at 95% should be negative (a loss)."""
        result = two_ticker_setup.analyse({"AAPL": 0.5, "MSFT": 0.5})
        assert result["var_95_daily"] < 0

    def test_correlation_matrix_contains_tickers(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.5, "MSFT": 0.5})
        corr = result["correlation_matrix"]
        assert "AAPL" in corr
        assert "MSFT" in corr

    def test_self_correlation_is_one(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.5, "MSFT": 0.5})
        corr = result["correlation_matrix"]
        assert abs(corr["AAPL"]["AAPL"] - 1.0) < 0.001
        assert abs(corr["MSFT"]["MSFT"] - 1.0) < 0.001

    def test_sector_exposure_sums_to_one(self, two_ticker_setup):
        result = two_ticker_setup.analyse({"AAPL": 0.5, "MSFT": 0.5})
        total = sum(result["sector_exposure"].values())
        assert abs(total - 1.0) < 0.001

    def test_holdings_preserved_in_result(self, two_ticker_setup):
        holdings = {"AAPL": 0.7, "MSFT": 0.3}
        result = two_ticker_setup.analyse(holdings)
        assert "AAPL" in result["holdings"]
        assert "MSFT" in result["holdings"]


# ─────────────────────────────────────────────────────────────────────────────
# Weight normalisation
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightNormalisation:
    def test_unnormalised_weights_still_produce_valid_result(self, two_ticker_setup):
        """Weights that don't sum to 1 should be normalised internally."""
        result = two_ticker_setup.analyse({"AAPL": 3.0, "MSFT": 1.0})
        # If normalised, sector exposure should still sum to 1
        total = sum(result["sector_exposure"].values())
        assert abs(total - 1.0) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_returns_error_when_no_data_available(self, service, monkeypatch):
        monkeypatch.setattr(service._mds, "get_historical_data", lambda *a, **kw: pd.DataFrame())
        monkeypatch.setattr(service._mds, "get_quote", MagicMock(return_value={}))
        result = service.analyse({"AAPL": 1.0})
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Beta
# ─────────────────────────────────────────────────────────────────────────────

class TestBetaFallback:
    def test_beta_falls_back_to_one_when_spy_unavailable(self, service, monkeypatch):
        returns_df = _make_returns_df(["AAPL"], n=252)

        def mock_history(ticker, period="1y"):
            if ticker == "AAPL":
                return _returns_to_prices(returns_df["AAPL"])
            return pd.DataFrame()  # SPY returns nothing

        monkeypatch.setattr(service._mds, "get_historical_data", mock_history)
        monkeypatch.setattr(service._mds, "get_quote", MagicMock(return_value={"sector": "Technology"}))
        result = service.analyse({"AAPL": 1.0})
        assert result["portfolio_beta"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Static helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestMaxDrawdownStatic:
    def test_always_declining_gives_large_drawdown(self):
        returns = np.array([-0.05] * 20)
        dd = PortfolioRiskService._max_drawdown(returns)
        assert dd > 0.6

    def test_always_rising_gives_zero_drawdown(self):
        returns = np.array([0.01] * 100)
        dd = PortfolioRiskService._max_drawdown(returns)
        assert math.isclose(dd, 0.0, abs_tol=1e-9)
