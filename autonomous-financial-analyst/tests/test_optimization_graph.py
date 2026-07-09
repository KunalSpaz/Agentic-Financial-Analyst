"""
test_optimization_graph.py
----------------------------
Unit tests for StrategyOptimizationNarrativeGraph.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())

from backend.agents.optimization_graph import StrategyOptimizationNarrativeGraph  # noqa: E402

_OPT_RESULT = {
    "ticker": "AAPL",
    "objective": "maximize_return",
    "best_parameters": {"rsi_buy_threshold": 30, "rsi_sell_threshold": 70},
    "best_return": 0.22,
    "best_sharpe": 1.5,
    "best_drawdown": 0.12,
    "iterations": 48,
}


class TestStrategyOptimizationNarrativeGraph:
    def test_run_returns_narrative_from_graph(self):
        graph = StrategyOptimizationNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "Aggressive RSI thresholds won out."}
        result = graph.run("AAPL", "maximize_return", _OPT_RESULT)
        assert result == "Aggressive RSI thresholds won out."

    def test_run_falls_back_on_exception(self):
        graph = StrategyOptimizationNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.side_effect = RuntimeError("LLM down")
        result = graph.run("AAPL", "maximize_return", _OPT_RESULT)
        assert "AAPL" in result

    def test_run_falls_back_when_narrative_empty(self):
        graph = StrategyOptimizationNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": ""}
        result = graph.run("AAPL", "maximize_return", _OPT_RESULT)
        assert "AAPL" in result

    def test_graph_invoked_with_ticker_objective_and_result(self):
        graph = StrategyOptimizationNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "ok"}
        graph.run("NVDA", "maximize_sharpe", _OPT_RESULT)
        submitted_state = graph._graph.invoke.call_args.args[0]
        assert submitted_state["ticker"] == "NVDA"
        assert submitted_state["objective"] == "maximize_sharpe"
        assert submitted_state["optimization_result"] == _OPT_RESULT
