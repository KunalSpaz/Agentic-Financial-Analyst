"""
test_frontend_components.py
----------------------------
Unit tests for frontend Plotly chart factories and the confidence gauge.

All functions return go.Figure objects — no Streamlit calls are made —
so these tests run without a browser, display, or mocked Streamlit state.

Covers:
  charts.py:
    candlestick_chart  — traces, overlays, volume colours, layout
    rsi_chart          — RSI trace, reference hlines, y-axis range
    macd_chart         — MACD/signal/histogram traces, bar colours, zero hline
    equity_curve_chart — trace, initial-capital hline, title, y-axis label
    opportunity_bar_chart — colour map, empty guard, height, orientation
    correlation_heatmap   — axes, colorscale, zmid, empty guard
    sector_pie_chart      — hole, labels/values, empty guard

  metrics.py:
    confidence_gauge   — score value, recommendation title, 5 colour steps,
                         correct step ranges and colours, threshold value

  sidebar.py (module-level constant):
    API_BASE_URL default and BACKEND_URL env-var override
"""
from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# ── Mock Streamlit before importing anything that touches it ──────────────────
sys.modules.setdefault("streamlit", MagicMock())

from frontend.components.charts import (   # noqa: E402
    candlestick_chart,
    correlation_heatmap,
    equity_curve_chart,
    macd_chart,
    opportunity_bar_chart,
    rsi_chart,
    sector_pie_chart,
)
from frontend.components.metrics import confidence_gauge  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """Minimal OHLCV DataFrame with indicator columns."""
    rng = np.random.default_rng(0)
    n = 30
    idx = pd.date_range("2024-01-01", periods=n)
    close = rng.uniform(150, 200, n)
    open_ = close * rng.uniform(0.98, 1.02, n)
    return pd.DataFrame({
        "Open":             open_,
        "High":             close * 1.02,
        "Low":              close * 0.98,
        "Close":            close,
        "Volume":           rng.integers(1_000_000, 5_000_000, n),
        "SMA_50":           close * 0.97,
        "SMA_200":          close * 0.93,
        "RSI_14":           rng.uniform(20, 80, n),
        "MACD_12_26_9":     rng.uniform(-2, 2, n),
        "MACDs_12_26_9":    rng.uniform(-1, 1, n),
        "MACDh_12_26_9":    rng.uniform(-1, 1, n),
        "BBU_20_2.0_2.0":   close * 1.05,
        "BBM_20_2.0_2.0":   close,
        "BBL_20_2.0_2.0":   close * 0.95,
    }, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# candlestick_chart
# ─────────────────────────────────────────────────────────────────────────────

class TestCandlestickChart:

    def test_returns_figure(self, ohlcv_df):
        assert isinstance(candlestick_chart(ohlcv_df, "AAPL"), go.Figure)

    def test_title_contains_ticker(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "TSLA")
        assert "TSLA" in fig.layout.title.text

    def test_height_is_600(self, ohlcv_df):
        assert candlestick_chart(ohlcv_df, "X").layout.height == 600

    def test_rangeslider_hidden(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "X")
        assert fig.layout.xaxis.rangeslider.visible is False

    def test_with_volume_includes_bar_trace(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "X", show_volume=True)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 1

    def test_without_volume_no_bar_trace(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "X", show_volume=False)
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 0

    def test_with_ma_adds_sma_traces(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "X", show_ma=True, show_volume=False)
        scatter_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert "SMA 50" in scatter_names
        assert "SMA 200" in scatter_names

    def test_without_ma_no_sma_traces(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "X", show_ma=False, show_volume=False)
        scatter_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert "SMA 50" not in scatter_names
        assert "SMA 200" not in scatter_names

    def test_no_sma_traces_when_columns_absent(self):
        """show_ma=True but DataFrame lacks SMA columns → no SMA traces."""
        bare = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [98.0],
            "Close": [102.0], "Volume": [1_000_000],
        }, index=pd.date_range("2024-01-01", periods=1))
        fig = candlestick_chart(bare, "X", show_ma=True, show_volume=False)
        scatter_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert "SMA 50" not in scatter_names

    def test_with_bb_adds_bb_traces(self, ohlcv_df):
        fig = candlestick_chart(ohlcv_df, "X", show_bb=True, show_volume=False)
        scatter_names = [t.name for t in fig.data if isinstance(t, go.Scatter)]
        assert "BB Upper" in scatter_names
        assert "BB Mid" in scatter_names
        assert "BB Lower" in scatter_names

    def test_volume_color_green_when_close_ge_open(self):
        """Single bar where Close >= Open → green (#00C853)."""
        df = pd.DataFrame({
            "Open": [100.0], "High": [105.0], "Low": [98.0],
            "Close": [103.0], "Volume": [1_000_000],
        }, index=pd.date_range("2024-01-01", periods=1))
        fig = candlestick_chart(df, "X", show_volume=True, show_ma=False)
        bar_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert bar_trace.marker.color[0] == "#00C853"

    def test_volume_color_red_when_close_lt_open(self):
        """Single bar where Close < Open → red (#D50000)."""
        df = pd.DataFrame({
            "Open": [105.0], "High": [106.0], "Low": [98.0],
            "Close": [100.0], "Volume": [1_000_000],
        }, index=pd.date_range("2024-01-01", periods=1))
        fig = candlestick_chart(df, "X", show_volume=True, show_ma=False)
        bar_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert bar_trace.marker.color[0] == "#D50000"


# ─────────────────────────────────────────────────────────────────────────────
# rsi_chart
# ─────────────────────────────────────────────────────────────────────────────

class TestRsiChart:

    def test_returns_figure(self, ohlcv_df):
        assert isinstance(rsi_chart(ohlcv_df), go.Figure)

    def test_rsi_trace_added_when_column_present(self, ohlcv_df):
        fig = rsi_chart(ohlcv_df)
        names = [t.name for t in fig.data]
        assert "RSI(14)" in names

    def test_no_rsi_trace_when_column_absent(self):
        fig = rsi_chart(pd.DataFrame({"Close": [100.0]}))
        assert len(fig.data) == 0

    def test_yaxis_range_0_to_100(self, ohlcv_df):
        fig = rsi_chart(ohlcv_df)
        assert fig.layout.yaxis.range == (0, 100)

    def test_overbought_hline_at_70(self, ohlcv_df):
        fig = rsi_chart(ohlcv_df)
        y_levels = [s.y0 for s in fig.layout.shapes]
        assert 70 in y_levels

    def test_oversold_hline_at_30(self, ohlcv_df):
        fig = rsi_chart(ohlcv_df)
        y_levels = [s.y0 for s in fig.layout.shapes]
        assert 30 in y_levels


# ─────────────────────────────────────────────────────────────────────────────
# macd_chart
# ─────────────────────────────────────────────────────────────────────────────

class TestMacdChart:

    def test_returns_figure(self, ohlcv_df):
        assert isinstance(macd_chart(ohlcv_df), go.Figure)

    def test_macd_line_trace_added(self, ohlcv_df):
        fig = macd_chart(ohlcv_df)
        names = [t.name for t in fig.data]
        assert "MACD" in names

    def test_signal_line_trace_added(self, ohlcv_df):
        fig = macd_chart(ohlcv_df)
        names = [t.name for t in fig.data]
        assert "Signal" in names

    def test_histogram_bar_trace_added(self, ohlcv_df):
        fig = macd_chart(ohlcv_df)
        names = [t.name for t in fig.data]
        assert "Histogram" in names

    def test_histogram_positive_values_are_green(self):
        df = pd.DataFrame({"MACDh_12_26_9": [1.0, 2.0]},
                          index=pd.date_range("2024-01-01", periods=2))
        fig = macd_chart(df)
        hist_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert all(c == "#00C853" for c in hist_trace.marker.color)

    def test_histogram_negative_values_are_red(self):
        df = pd.DataFrame({"MACDh_12_26_9": [-1.0, -2.0]},
                          index=pd.date_range("2024-01-01", periods=2))
        fig = macd_chart(df)
        hist_trace = next(t for t in fig.data if isinstance(t, go.Bar))
        assert all(c == "#D50000" for c in hist_trace.marker.color)

    def test_zero_hline_present(self, ohlcv_df):
        fig = macd_chart(ohlcv_df)
        y_levels = [s.y0 for s in fig.layout.shapes]
        assert 0 in y_levels

    def test_no_traces_when_columns_absent(self):
        fig = macd_chart(pd.DataFrame({"Close": [100.0]}))
        assert len(fig.data) == 0


# ─────────────────────────────────────────────────────────────────────────────
# equity_curve_chart
# ─────────────────────────────────────────────────────────────────────────────

class TestEquityCurveChart:

    def test_returns_figure(self):
        assert isinstance(equity_curve_chart([10000.0, 10200.0, 9800.0], "AAPL"), go.Figure)

    def test_title_contains_ticker(self):
        fig = equity_curve_chart([10000.0, 11000.0], "NVDA")
        assert "NVDA" in fig.layout.title.text

    def test_title_contains_strategy_name(self):
        fig = equity_curve_chart([10000.0, 11000.0], "X", strategy_name="RSI_Test")
        assert "RSI_Test" in fig.layout.title.text

    def test_initial_capital_hline_at_first_value(self):
        curve = [10000.0, 10500.0, 9800.0]
        fig = equity_curve_chart(curve, "X")
        y_levels = [s.y0 for s in fig.layout.shapes]
        assert 10000.0 in y_levels

    def test_yaxis_title_is_portfolio_value(self):
        fig = equity_curve_chart([10000.0], "X")
        assert "Portfolio Value" in fig.layout.yaxis.title.text

    def test_scatter_trace_uses_supplied_equity_curve(self):
        curve = [10000.0, 11000.0, 10500.0]
        fig = equity_curve_chart(curve, "X")
        scatter = fig.data[0]
        assert list(scatter.y) == curve


# ─────────────────────────────────────────────────────────────────────────────
# opportunity_bar_chart
# ─────────────────────────────────────────────────────────────────────────────

class TestOpportunityBarChart:

    _COLOR_MAP = {
        "STRONG BUY":  "#00C853",
        "BUY":         "#69F0AE",
        "HOLD":        "#FFD740",
        "SELL":        "#FF6D00",
        "STRONG SELL": "#D50000",
    }

    def _make_opps(self, recs: list[str]) -> list[dict]:
        return [{"ticker": f"T{i}", "confidence_score": 60.0, "recommendation": r}
                for i, r in enumerate(recs)]

    def test_empty_list_returns_empty_figure(self):
        fig = opportunity_bar_chart([])
        assert len(fig.data) == 0

    @pytest.mark.parametrize("rec,expected_color", list(_COLOR_MAP.items()))
    def test_color_map(self, rec, expected_color):
        fig = opportunity_bar_chart(self._make_opps([rec]))
        bar = fig.data[0]
        assert bar.marker.color[0] == expected_color

    def test_unknown_recommendation_uses_white(self):
        fig = opportunity_bar_chart(self._make_opps(["UNKNOWN"]))
        assert fig.data[0].marker.color[0] == "#FFFFFF"

    def test_horizontal_orientation(self):
        fig = opportunity_bar_chart(self._make_opps(["BUY"]))
        assert fig.data[0].orientation == "h"

    def test_height_scales_with_ticker_count(self):
        opps = self._make_opps(["BUY"] * 10)
        fig = opportunity_bar_chart(opps)
        assert fig.layout.height == max(300, 10 * 45)

    def test_height_minimum_300(self):
        fig = opportunity_bar_chart(self._make_opps(["BUY"]))
        assert fig.layout.height >= 300

    def test_xaxis_range_0_to_105(self):
        fig = opportunity_bar_chart(self._make_opps(["HOLD"]))
        assert fig.layout.xaxis.range == (0, 105)


# ─────────────────────────────────────────────────────────────────────────────
# correlation_heatmap
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelationHeatmap:

    _MATRIX = {
        "AAPL": {"AAPL": 1.0, "MSFT": 0.72},
        "MSFT": {"AAPL": 0.72, "MSFT": 1.0},
    }

    def test_empty_dict_returns_empty_figure(self):
        fig = correlation_heatmap({})
        assert len(fig.data) == 0

    def test_returns_figure(self):
        assert isinstance(correlation_heatmap(self._MATRIX), go.Figure)

    def test_x_labels_are_tickers(self):
        fig = correlation_heatmap(self._MATRIX)
        assert set(fig.data[0].x) == {"AAPL", "MSFT"}

    def test_y_labels_are_tickers(self):
        fig = correlation_heatmap(self._MATRIX)
        assert set(fig.data[0].y) == {"AAPL", "MSFT"}

    def test_colorscale_is_rdbu(self):
        # Plotly resolves "RdBu" to its RGB tuple list; verify via the known midpoint
        fig = correlation_heatmap(self._MATRIX)
        colorscale = fig.data[0].colorscale
        # RdBu midpoint at position 0.5 is rgb(247,247,247) (neutral white)
        midpoint_color = dict(colorscale).get(0.5, "")
        assert "247" in midpoint_color  # white midpoint of the RdBu diverging scale

    def test_zmid_is_zero(self):
        fig = correlation_heatmap(self._MATRIX)
        assert fig.data[0].zmid == 0


# ─────────────────────────────────────────────────────────────────────────────
# sector_pie_chart
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorPieChart:

    _EXPOSURE = {"Technology": 0.5, "Healthcare": 0.3, "Finance": 0.2}

    def test_empty_dict_returns_empty_figure(self):
        fig = sector_pie_chart({})
        assert len(fig.data) == 0

    def test_returns_figure(self):
        assert isinstance(sector_pie_chart(self._EXPOSURE), go.Figure)

    def test_hole_is_0_4_donut(self):
        fig = sector_pie_chart(self._EXPOSURE)
        assert fig.data[0].hole == 0.4

    def test_labels_match_keys(self):
        fig = sector_pie_chart(self._EXPOSURE)
        assert set(fig.data[0].labels) == set(self._EXPOSURE.keys())

    def test_values_match_weights(self):
        fig = sector_pie_chart(self._EXPOSURE)
        assert set(fig.data[0].values) == set(self._EXPOSURE.values())


# ─────────────────────────────────────────────────────────────────────────────
# confidence_gauge
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceGauge:

    def test_returns_figure(self):
        assert isinstance(confidence_gauge(75.0, "BUY"), go.Figure)

    def test_score_value_set_correctly(self):
        fig = confidence_gauge(82.5, "STRONG BUY")
        assert fig.data[0].value == 82.5

    def test_recommendation_in_title(self):
        fig = confidence_gauge(50.0, "HOLD")
        assert "HOLD" in fig.data[0].title.text

    def test_five_color_steps_defined(self):
        fig = confidence_gauge(50.0, "HOLD")
        assert len(fig.data[0].gauge.steps) == 5

    def test_step_ranges_correct(self):
        steps = confidence_gauge(50.0, "X").data[0].gauge.steps
        ranges = [tuple(s.range) for s in steps]
        assert (0, 34) in ranges
        assert (34, 49) in ranges
        assert (49, 64) in ranges
        assert (64, 79) in ranges
        assert (79, 100) in ranges

    def test_step_colors_correct(self):
        steps = confidence_gauge(50.0, "X").data[0].gauge.steps
        color_by_range = {tuple(s.range): s.color for s in steps}
        assert color_by_range[(0, 34)] == "#D50000"
        assert color_by_range[(34, 49)] == "#FF6D00"
        assert color_by_range[(49, 64)] == "#FFD740"
        assert color_by_range[(64, 79)] == "#69F0AE"
        assert color_by_range[(79, 100)] == "#00C853"

    def test_threshold_value_equals_score(self):
        fig = confidence_gauge(63.0, "HOLD")
        assert fig.data[0].gauge.threshold.value == 63.0

    def test_custom_color_applied(self):
        fig = confidence_gauge(70.0, "BUY", color="#FF0000")
        assert fig.data[0].gauge.bar.color == "#FF0000"

    def test_height_is_300(self):
        assert confidence_gauge(50.0, "HOLD").layout.height == 300


# ─────────────────────────────────────────────────────────────────────────────
# sidebar — module-level API_BASE_URL constant
# ─────────────────────────────────────────────────────────────────────────────

class TestSidebarApiUrl:

    def test_default_url_is_localhost_8000(self, monkeypatch):
        monkeypatch.delenv("BACKEND_URL", raising=False)
        import frontend.components.sidebar as sidebar_mod
        importlib.reload(sidebar_mod)
        assert sidebar_mod.API_BASE_URL == "http://localhost:8000"

    def test_url_reads_from_backend_url_env_var(self, monkeypatch):
        monkeypatch.setenv("BACKEND_URL", "http://prod-server:9000")
        import frontend.components.sidebar as sidebar_mod
        importlib.reload(sidebar_mod)
        assert sidebar_mod.API_BASE_URL == "http://prod-server:9000"
