"""
test_backtest_graph.py
-----------------------
Unit tests for BacktestNarrativeGraph.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())

from backend.agents.backtest_graph import BacktestNarrativeGraph  # noqa: E402
from backend.services.backtesting_service import BacktestResult   # noqa: E402


def _make_result(**overrides) -> BacktestResult:
    base = dict(
        ticker="AAPL", strategy_name="RSI_MACD_Sentiment", start_date="2023-01-01",
        end_date="2023-12-31", parameters={}, total_return=0.15, sharpe_ratio=1.2,
        max_drawdown=0.1, win_rate=0.6, num_trades=10, equity_curve=[10000, 11500],
        trade_log=[],
    )
    base.update(overrides)
    return BacktestResult(**base)


class TestBacktestNarrativeGraph:
    def test_run_returns_narrative_from_graph(self):
        graph = BacktestNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "Solid risk-adjusted return."}
        result = graph.run(_make_result())
        assert result == "Solid risk-adjusted return."

    def test_run_falls_back_on_exception(self):
        graph = BacktestNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.side_effect = RuntimeError("LLM down")
        result = graph.run(_make_result())
        assert "AAPL" in result

    def test_run_falls_back_when_narrative_empty(self):
        graph = BacktestNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": ""}
        result = graph.run(_make_result())
        assert "AAPL" in result

    def test_run_accepts_dict_input(self):
        graph = BacktestNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "ok"}
        result = graph.run({"ticker": "MSFT", "total_return": 0.1})
        assert result == "ok"

    def test_graph_invoked_with_ticker_and_result(self):
        graph = BacktestNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "ok"}
        graph.run(_make_result(ticker="NVDA"))
        submitted_state = graph._graph.invoke.call_args.args[0]
        assert submitted_state["ticker"] == "NVDA"
        assert submitted_state["backtest_result"]["ticker"] == "NVDA"
