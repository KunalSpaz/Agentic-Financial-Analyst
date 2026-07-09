"""
portfolio_graph.py
-------------------
Single-node LangGraph wrapping the Portfolio Risk reasoning agent. Wired
into ``POST /portfolio-analysis`` to interpret a completed
PortfolioRiskService run.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from backend.agents.portfolio_risk_agent import create_portfolio_risk_node
from backend.agents.state import PortfolioRiskNarrativeState
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _build_graph():
    graph = StateGraph(PortfolioRiskNarrativeState)
    graph.add_node("portfolio_risk", create_portfolio_risk_node())
    graph.add_edge(START, "portfolio_risk")
    graph.add_edge("portfolio_risk", END)
    return graph.compile()


class PortfolioRiskNarrativeGraph:
    """Interprets a :class:`PortfolioRiskService.analyse` result into a narrative."""

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, risk_result: Dict[str, Any]) -> str:
        try:
            final_state = self._graph.invoke({"risk_result": risk_result})
            return final_state.get("narrative") or self._fallback(risk_result)
        except Exception as exc:
            logger.error("Portfolio risk narrative generation failed: %s", exc)
            return self._fallback(risk_result)

    @staticmethod
    def _fallback(result: Dict[str, Any]) -> str:
        return (
            f"Portfolio volatility={result.get('portfolio_volatility', 0):.2%}, "
            f"beta={result.get('portfolio_beta', 0):.2f}, "
            f"Sharpe={result.get('sharpe_ratio', 0):.2f}, "
            f"VaR(95%)={result.get('var_95_daily', 0):.2%}."
        )
