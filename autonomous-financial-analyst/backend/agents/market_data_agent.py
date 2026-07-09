"""
market_data_agent.py
--------------------
LangGraph Market Data node — interprets quote/quality context (price,
market cap, 52-week range, data completeness) as part of the stock-analysis
fan-out. Actual data fetching stays in MarketDataService; this node only
reasons over the already-fetched quote.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StockAnalysisState

ROLE = "Market Data Specialist"
GOAL = (
    "Fetch accurate, up-to-date OHLCV market data and fundamental "
    "metrics for any given stock ticker. Provide clean, structured "
    "data suitable for technical analysis."
)
BACKSTORY = (
    "You are a quantitative data engineer with 10 years of experience "
    "fetching and validating financial market data. You ensure data quality "
    "and completeness before it reaches any analysis pipeline."
)


def create_market_data_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Market Data agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        quote = state.get("quote") or {}
        quote_lines = "\n".join(f"- {k}: {v}" for k, v in quote.items()) or "No quote data available."
        task = (
            f"Review the following live market data snapshot for {ticker}. "
            "Flag any missing, stale, or suspicious fields, note where the price "
            "sits relative to its 52-week range, and summarise what this data context "
            f"means for downstream analysis.\n\n{quote_lines}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"market_data_insight": response.content}

    return node
