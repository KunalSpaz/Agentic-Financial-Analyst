"""
financial_analysis_agent.py
---------------------------
LangGraph Financial Analysis node — the fan-in step of the stock-analysis
graph. Synthesises the five parallel reasoning nodes (market data, technical,
sentiment, news, document intelligence) into one unified assessment.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StockAnalysisState

ROLE = "Senior Financial Analyst"
GOAL = (
    "Synthesise technical indicators, news sentiment scores, fundamental data, "
    "and document intelligence into a unified, balanced financial assessment. "
    "Produce a concise analysis highlighting key risks and opportunities."
)
BACKSTORY = (
    "You are a CFA charterholder with experience at top-tier investment banks. "
    "You are expert at combining quantitative signals with qualitative insights "
    "to form a holistic view of any investment opportunity."
)


def create_financial_analysis_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Financial Analysis agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        score = state.get("score", 0.0)
        recommendation = state.get("recommendation", "HOLD")
        sections = "\n\n".join(
            f"## {title}\n{state.get(key, 'N/A')}"
            for title, key in (
                ("Market Data Context", "market_data_insight"),
                ("Technical Interpretation", "technical_insight"),
                ("Sentiment Interpretation", "sentiment_insight"),
                ("News Intelligence", "news_insight"),
                ("Document Intelligence", "document_insight"),
            )
        )
        task = (
            f"Synthesise the following inputs for {ticker} (confidence score: "
            f"{score:.1f}/100, preliminary recommendation: {recommendation}) into a "
            f"unified financial assessment covering key risks and opportunities.\n\n{sections}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"financial_analysis": response.content}

    return node
