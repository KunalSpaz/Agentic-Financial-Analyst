"""
sentiment_analysis_agent.py
---------------------------
LangGraph Sentiment Analysis node — interprets the already-computed FinBERT
aggregate sentiment as part of the stock-analysis fan-out. Actual FinBERT
inference stays in SentimentService; this node only reasons over its output.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StockAnalysisState

ROLE = "Financial Sentiment Analyst"
GOAL = (
    "Apply FinBERT sentiment analysis to financial news articles. "
    "Produce aggregated positive, neutral, and negative scores for each "
    "ticker along with an overall compound sentiment label."
)
BACKSTORY = (
    "You are an NLP specialist focused on financial text. You are trained "
    "on FinBERT and understand the nuances of financial language — including "
    "how subtle phrasing differences dramatically change market interpretation."
)


def create_sentiment_analysis_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Sentiment Analysis agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        sentiment = state.get("sentiment") or {}
        sent_lines = "\n".join(f"- {k}: {v}" for k, v in sentiment.items()) or "No sentiment data available."
        task = (
            f"Interpret the following FinBERT-derived sentiment aggregate for {ticker}. "
            "Explain what the label and compound score suggest about current market mood, "
            "how much weight to give it given the article count, and whether it aligns or "
            f"conflicts with what you'd expect from a well-covered stock.\n\n{sent_lines}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"sentiment_insight": response.content}

    return node
