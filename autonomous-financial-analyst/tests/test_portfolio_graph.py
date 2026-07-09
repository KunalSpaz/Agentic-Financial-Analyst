"""
test_portfolio_graph.py
------------------------
Unit tests for PortfolioRiskNarrativeGraph.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())

from backend.agents.portfolio_graph import PortfolioRiskNarrativeGraph  # noqa: E402

_RISK_RESULT = {
    "portfolio_volatility": 0.22,
    "portfolio_beta": 1.15,
    "sharpe_ratio": 1.4,
    "max_drawdown": 0.18,
    "var_95_daily": -0.03,
    "correlation_matrix": {},
    "sector_exposure": {"Technology": 0.6, "Healthcare": 0.4},
}


class TestPortfolioRiskNarrativeGraph:
    def test_run_returns_narrative_from_graph(self):
        graph = PortfolioRiskNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "Concentrated in tech."}
        result = graph.run(_RISK_RESULT)
        assert result == "Concentrated in tech."

    def test_run_falls_back_on_exception(self):
        graph = PortfolioRiskNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.side_effect = RuntimeError("LLM down")
        result = graph.run(_RISK_RESULT)
        assert isinstance(result, str) and len(result) > 0

    def test_run_falls_back_when_narrative_empty(self):
        graph = PortfolioRiskNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": ""}
        result = graph.run(_RISK_RESULT)
        assert isinstance(result, str) and len(result) > 0

    def test_graph_invoked_with_risk_result(self):
        graph = PortfolioRiskNarrativeGraph()
        graph._graph = MagicMock()
        graph._graph.invoke.return_value = {"narrative": "ok"}
        graph.run(_RISK_RESULT)
        submitted_state = graph._graph.invoke.call_args.args[0]
        assert submitted_state["risk_result"] == _RISK_RESULT
