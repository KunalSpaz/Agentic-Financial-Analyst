"""
strategy_optimization_agent.py
-------------------------------
LangGraph Strategy Optimization node — interprets a completed grid-search
run from StrategyOptimizationService into a narrative explaining parameter
tradeoffs and overfitting caution. Wired into POST /optimize-strategy via
backend/agents/optimization_graph.py.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StrategyOptimizationNarrativeState

ROLE = "Strategy Optimization Engineer"
GOAL = (
    "Automatically optimise trading strategy parameters through grid search. "
    "Identify the parameter configuration that best achieves the chosen "
    "objective (maximize return, maximize Sharpe, minimize drawdown) while "
    "avoiding overfitting."
)
BACKSTORY = (
    "You are a machine learning engineer who specialises in financial strategy "
    "optimisation. You understand the risk of data snooping bias and always "
    "validate results out-of-sample before recommending parameter changes."
)


def create_strategy_optimization_node() -> Callable[[StrategyOptimizationNarrativeState], Dict[str, str]]:
    """Build the LangGraph node function for the Strategy Optimization agent."""
    llm = get_chat_model()

    def node(state: StrategyOptimizationNarrativeState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        objective = state.get("objective", "maximize_return")
        result = state.get("optimization_result") or {}
        metrics_lines = "\n".join(
            f"- {k}: {v}" for k, v in result.items() if k != "all_results"
        )
        iterations = result.get("iterations", 0)
        task = (
            f"Interpret the following strategy optimization results for {ticker} "
            f"(objective: {objective}, {iterations} parameter combinations tested). "
            "Explain why the best parameters won, what tradeoffs they imply versus other "
            "objectives, and caution on overfitting/data-snooping risk given the iteration "
            f"count.\n\n{metrics_lines}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"narrative": response.content}

    return node
