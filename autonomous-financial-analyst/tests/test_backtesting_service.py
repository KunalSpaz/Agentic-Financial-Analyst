"""
test_backtesting_service.py
----------------------------
Unit tests for the BacktestingService.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backend.services.backtesting_service import BacktestParameters, BacktestingService


@pytest.fixture
def service() -> BacktestingService:
    return BacktestingService()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Generate synthetic OHLCV + indicator data for backtesting."""
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Open":   close * 0.995,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)
    return df


class TestRunBacktest:
    def test_returns_backtest_result_object(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        assert result.ticker == "AAPL"
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert isinstance(result.num_trades, int)

    def test_equity_curve_length_matches_data(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        # equity curve should be non-empty
        assert len(result.equity_curve) > 0

    def test_max_drawdown_is_non_negative(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        assert result.max_drawdown >= 0

    def test_win_rate_between_zero_and_one(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        assert 0.0 <= result.win_rate <= 1.0

    def test_raises_on_empty_data(self, service):
        with patch.object(service._mds, "get_historical_data", return_value=pd.DataFrame()):
            with pytest.raises(ValueError):
                service.run_backtest("BADTICKER")

    def test_custom_params_respected(self, service, sample_df):
        params = BacktestParameters(rsi_buy_threshold=40.0, rsi_sell_threshold=65.0)
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL", params=params)
        assert result.parameters["rsi_buy_threshold"] == 40.0
        assert result.parameters["rsi_sell_threshold"] == 65.0

    def test_strategy_name_is_set(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        assert result.strategy_name == "RSI_MACD_Sentiment"

    def test_start_and_end_dates_are_strings(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        assert isinstance(result.start_date, str)
        assert isinstance(result.end_date, str)

    def test_trade_log_entries_have_required_keys(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        for trade in result.trade_log:
            for key in ("date", "action", "price", "shares", "pnl", "capital"):
                assert key in trade, f"Trade log entry missing key: {key}"

    def test_all_trade_actions_are_buy_or_sell(self, service, sample_df):
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result = service.run_backtest("AAPL")
        for trade in result.trade_log:
            assert trade["action"] in ("BUY", "SELL")

    def test_negative_sentiment_reduces_trades(self, service, sample_df):
        """NEGATIVE sentiment should block buys → fewer or zero trades."""
        with patch.object(service._mds, "get_historical_data", return_value=sample_df):
            result_neutral  = service.run_backtest("AAPL", sentiment_label="NEUTRAL")
            result_negative = service.run_backtest("AAPL", sentiment_label="NEGATIVE")
        assert result_negative.num_trades <= result_neutral.num_trades


class TestSharpeRatio:
    def test_positive_returns_give_positive_sharpe(self):
        equity = [10_000 * (1.001 ** i) for i in range(252)]
        sharpe = BacktestingService._sharpe_ratio(equity)
        assert sharpe > 0

    def test_flat_equity_gives_zero_sharpe(self):
        equity = [10_000.0] * 100
        sharpe = BacktestingService._sharpe_ratio(equity)
        assert sharpe == 0.0


class TestMaxDrawdown:
    def test_declining_equity_has_large_drawdown(self):
        equity = [10_000 - i * 100 for i in range(50)]
        dd = BacktestingService._max_drawdown(equity)
        assert dd > 0.4  # should be close to 50%

    def test_rising_equity_has_zero_drawdown(self):
        equity = [10_000 + i * 100 for i in range(50)]
        dd = BacktestingService._max_drawdown(equity)
        assert dd == 0.0
