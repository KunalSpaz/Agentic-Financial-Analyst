"""
test_technical_analysis_service.py
------------------------------------
Unit tests for TechnicalAnalysisService.

Covers:
  - compute_indicators appends correct columns
  - compute_indicators is safe with insufficient data
  - get_latest_signals extracts scalar values correctly
  - get_latest_signals derives boolean signals correctly
  - get_latest_signals handles empty DataFrame gracefully
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.services.technical_analysis_service import TechnicalAnalysisService


@pytest.fixture
def service() -> TechnicalAnalysisService:
    return TechnicalAnalysisService()


@pytest.fixture
def ohlcv_300() -> pd.DataFrame:
    """300 days of synthetic OHLCV — enough for SMA-200 to stabilise."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.cumprod(1 + np.random.normal(0.0005, 0.01, n)) * 150
    df = pd.DataFrame(
        {
            "Open":   close * 0.998,
            "High":   close * 1.012,
            "Low":    close * 0.988,
            "Close":  close,
            "Volume": np.random.randint(5_000_000, 20_000_000, n).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def ohlcv_short() -> pd.DataFrame:
    """Only 10 rows — below the minimum threshold for indicators."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    close = np.linspace(100, 110, 10)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": np.ones(10) * 1_000_000},
        index=dates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# compute_indicators
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIndicators:
    def test_rsi_column_appended(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        assert "RSI_14" in df.columns

    def test_macd_columns_appended(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        assert "MACD_12_26_9" in df.columns
        assert "MACDs_12_26_9" in df.columns
        assert "MACDh_12_26_9" in df.columns

    def test_sma_columns_appended(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        assert "SMA_50" in df.columns
        assert "SMA_200" in df.columns

    def test_bollinger_band_columns_appended(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        assert "BBU_20_2.0_2.0" in df.columns
        assert "BBL_20_2.0_2.0" in df.columns
        assert "BBM_20_2.0_2.0" in df.columns

    def test_volume_ratio_appended(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        assert "VOLUME_RATIO" in df.columns

    def test_rsi_values_in_valid_range(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        rsi = df["RSI_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_sma200_value_matches_manual_calculation(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        last_sma200 = df["SMA_200"].iloc[-1]
        manual = ohlcv_300["Close"].iloc[-200:].mean()
        assert abs(last_sma200 - manual) < 0.01

    def test_short_df_returns_without_crash(self, service, ohlcv_short):
        """Fewer than 30 rows: should return df unchanged without raising."""
        df = service.compute_indicators(ohlcv_short.copy())
        assert isinstance(df, pd.DataFrame)
        # No indicator columns should be present
        assert "RSI_14" not in df.columns

    def test_empty_df_returns_empty(self, service):
        df = service.compute_indicators(pd.DataFrame())
        assert df.empty

    def test_original_ohlcv_columns_preserved(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# get_latest_signals
# ─────────────────────────────────────────────────────────────────────────────

class TestGetLatestSignals:
    def test_returns_dict_with_scalar_rsi(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        signals = service.get_latest_signals(df)
        assert isinstance(signals.get("rsi"), float)

    def test_returns_empty_dict_for_empty_df(self, service):
        signals = service.get_latest_signals(pd.DataFrame())
        assert signals == {}

    def test_close_price_matches_last_row(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        signals = service.get_latest_signals(df)
        expected_close = float(ohlcv_300["Close"].iloc[-1])
        assert abs(signals["close"] - expected_close) < 0.001

    def test_golden_cross_when_sma50_above_sma200(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        # Force golden cross by overwriting the last row
        df.loc[df.index[-1], "SMA_50"] = 200.0
        df.loc[df.index[-1], "SMA_200"] = 150.0
        signals = service.get_latest_signals(df)
        assert signals["golden_cross"] is True
        assert signals["death_cross"] is False

    def test_death_cross_when_sma50_below_sma200(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        df.loc[df.index[-1], "SMA_50"] = 140.0
        df.loc[df.index[-1], "SMA_200"] = 170.0
        signals = service.get_latest_signals(df)
        assert signals["death_cross"] is True
        assert signals["golden_cross"] is False

    def test_rsi_oversold_flag_set_when_rsi_below_35(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        df.loc[df.index[-1], "RSI_14"] = 28.0
        signals = service.get_latest_signals(df)
        assert signals["rsi_oversold"] is True
        assert signals["rsi_overbought"] is False

    def test_rsi_overbought_flag_set_when_rsi_above_70(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        df.loc[df.index[-1], "RSI_14"] = 75.0
        signals = service.get_latest_signals(df)
        assert signals["rsi_overbought"] is True
        assert signals["rsi_oversold"] is False

    def test_high_volume_flag_when_ratio_above_1_5(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        df.loc[df.index[-1], "VOLUME_RATIO"] = 2.0
        signals = service.get_latest_signals(df)
        assert signals["high_volume"] is True

    def test_high_volume_flag_false_when_ratio_below_1_5(self, service, ohlcv_300):
        df = service.compute_indicators(ohlcv_300.copy())
        df.loc[df.index[-1], "VOLUME_RATIO"] = 0.9
        signals = service.get_latest_signals(df)
        assert signals["high_volume"] is False
