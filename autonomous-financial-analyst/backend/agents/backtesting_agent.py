"""
backtesting_agent.py
---------------------
LangGraph Backtesting node — interprets a completed BacktestingService run
(return, Sharpe, drawdown, trade log) into a narrative assessment. Wired
into POST /backtest via backend/agents/backtest_graph.py.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import BacktestNarrativeState

ROLE = "Quantitative Backtesting Specialist"
GOAL = (
    "Run rigorous historical backtests of trading strategies. Accurately "
    "compute performance metrics (return, Sharpe, drawdown, win rate) and "
    "provide clear interpretation of strategy strengths and weaknesses."
)
BACKSTORY = (
    "You are a quant researcher specialising in strategy simulation. You "
    "have backtested hundreds of strategies and understand the common pitfalls "
    "of overfitting, look-ahead bias, and transaction cost underestimation."
)


def create_backtesting_node() -> Callable[[BacktestNarrativeState], Dict[str, str]]:
    """Build the LangGraph node function for the Backtesting agent."""
    llm = get_chat_model()

    def node(state: BacktestNarrativeState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        result = state.get("backtest_result") or {}
        num_trades = result.get("num_trades", 0)
        metrics_lines = "\n".join(
            f"- {k}: {v}" for k, v in result.items() if k not in ("equity_curve", "trade_log")
        )
        task = (
            f"Interpret the following backtest results for {ticker} ({num_trades} trades). "
            "Explain what the return/Sharpe/drawdown/win-rate combination reveals about "
            "strategy quality, flag overfitting or look-ahead-bias risk given the trade "
            f"count and parameters, and note any transaction-cost caveats.\n\n{metrics_lines}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"narrative": response.content}

    return node
