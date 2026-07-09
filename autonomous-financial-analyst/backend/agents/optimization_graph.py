"""
optimization_graph.py
----------------------
Single-node LangGraph wrapping the Strategy Optimization reasoning agent.
Wired into ``POST /optimize-strategy`` to interpret a completed
StrategyOptimizationService grid-search run.
"""
from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from backend.agents.state import StrategyOptimizationNarrativeState
from backend.agents.strategy_optimization_agent import create_strategy_optimization_node
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _build_graph():
    graph = StateGraph(StrategyOptimizationNarrativeState)
    graph.add_node("strategy_optimization", create_strategy_optimization_node())
    graph.add_edge(START, "strategy_optimization")
    graph.add_edge("strategy_optimization", END)
    return graph.compile()


class StrategyOptimizationNarrativeGraph:
    """Interprets a :class:`StrategyOptimizationService.optimize` result into a narrative."""

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, ticker: str, objective: str, optimization_result: Dict[str, Any]) -> str:
        try:
            final_state = self._graph.invoke({
                "ticker": ticker,
                "objective": objective,
                "optimization_result": optimization_result,
            })
            return final_state.get("narrative") or self._fallback(ticker, optimization_result)
        except Exception as exc:
            logger.error("Strategy optimization narrative generation failed: %s", exc)
            return self._fallback(ticker, optimization_result)

    @staticmethod
    def _fallback(ticker: str, result: Dict[str, Any]) -> str:
        return (
            f"Best parameters for {ticker}: {result.get('best_parameters', {})} "
            f"(return={result.get('best_return', 0):.2%}, "
            f"sharpe={result.get('best_sharpe', 0):.2f}) "
            f"over {result.get('iterations', 0)} combinations tested."
        )
