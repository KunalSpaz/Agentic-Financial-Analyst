"""
test_market_data_service.py
---------------------------
Unit tests for MarketDataService.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import backend.services.market_data_service as _mds_module
from backend.services.market_data_service import MarketDataService


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the module-level TTL cache before every test to prevent cross-test pollution."""
    _mds_module._cache.clear()
    yield
    _mds_module._cache.clear()


@pytest.fixture
def service() -> MarketDataService:
    return MarketDataService()


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Minimal OHLCV DataFrame for testing."""
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open":  np.random.uniform(100, 200, 10),
            "High":  np.random.uniform(150, 250, 10),
            "Low":   np.random.uniform(80, 150, 10),
            "Close": np.random.uniform(100, 200, 10),
            "Volume": np.random.randint(1_000_000, 10_000_000, 10),
        },
        index=dates,
    )


class TestGetHistoricalData:
    def test_returns_dataframe_on_success(self, service, sample_ohlcv):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_ohlcv
            df = service.get_historical_data("AAPL", period="1y")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_returns_empty_df_on_error(self, service):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.side_effect = Exception("network error")
            df = service.get_historical_data("INVALID_TICKER")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_uses_cache_on_second_call(self, service, sample_ohlcv):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_ohlcv
            service.get_historical_data("AAPL", period="1y")
            service.get_historical_data("AAPL", period="1y")
            # Should only call yfinance once due to caching
            assert mock_ticker.return_value.history.call_count == 1


class TestGetQuote:
    def test_returns_dict_with_expected_keys(self, service):
        mock_info = {
            "longName": "Apple Inc.", "sector": "Technology",
            "currentPrice": 175.0, "marketCap": 2.7e12,
            "regularMarketChangePercent": 0.012,
            "volume": 50_000_000, "trailingPE": 28.5,
            "fiftyTwoWeekHigh": 199.0, "fiftyTwoWeekLow": 130.0,
            "beta": 1.2,
        }
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = mock_info
            quote = service.get_quote("AAPL")

        assert quote["ticker"] == "AAPL"
        assert quote["company_name"] == "Apple Inc."
        assert quote["price"] == 175.0
        assert quote["sector"] == "Technology"

    def test_returns_error_key_on_failure(self, service):
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = MagicMock(side_effect=Exception("API error"))
            # info is a property, accessing it raises; simulate gracefully
            type(mock_ticker.return_value).info = property(lambda self: (_ for _ in ()).throw(Exception("API error")))
            quote = service.get_quote("BADTICKER")
        assert "error" in quote


class TestGetBulkQuotes:
    def test_returns_list_of_quotes(self, service):
        mock_info = {"currentPrice": 100.0, "longName": "Test Corp"}
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.info = mock_info
            quotes = service.get_bulk_quotes(["AAPL", "MSFT"])
        assert isinstance(quotes, list)
        assert len(quotes) == 2
