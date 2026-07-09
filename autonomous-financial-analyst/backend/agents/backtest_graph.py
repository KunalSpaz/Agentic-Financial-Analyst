"""
backtest_graph.py
------------------
Single-node LangGraph wrapping the Backtesting reasoning agent. Wired into
``POST /backtest`` to interpret a completed BacktestingService run.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.agents.backtesting_agent import create_backtesting_node
from backend.agents.state import BacktestNarrativeState
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def _build_graph():
    graph = StateGraph(BacktestNarrativeState)
    graph.add_node("backtesting", create_backtesting_node())
    graph.add_edge(START, "backtesting")
    graph.add_edge("backtesting", END)
    return graph.compile()


class BacktestNarrativeGraph:
    """Interprets a :class:`BacktestResult` into a plain-language narrative."""

    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, backtest_result: Any) -> str:
        """
        Args:
            backtest_result: A ``BacktestResult`` dataclass instance (or dict)
                from :class:`BacktestingService`.

        Returns:
            Narrative string, or a short fallback message on failure.
        """
        result_dict = asdict(backtest_result) if is_dataclass(backtest_result) else dict(backtest_result)
        try:
            final_state = self._graph.invoke({
                "ticker": result_dict.get("ticker", ""),
                "backtest_result": result_dict,
            })
            return final_state.get("narrative") or self._fallback(result_dict)
        except Exception as exc:
            logger.error("Backtest narrative generation failed: %s", exc)
            return self._fallback(result_dict)

    @staticmethod
    def _fallback(result: dict) -> str:
        return (
            f"Backtest for {result.get('ticker', 'N/A')}: "
            f"return={result.get('total_return', 0):.2%}, "
            f"sharpe={result.get('sharpe_ratio', 0):.2f}, "
            f"max drawdown={result.get('max_drawdown', 0):.2%} "
            f"over {result.get('num_trades', 0)} trades."
        )
