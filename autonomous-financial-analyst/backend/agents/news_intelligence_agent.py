"""
news_intelligence_agent.py
--------------------------
LangGraph News Intelligence node — synthesises raw news headlines into
structured intelligence as part of the stock-analysis fan-out.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt, wrap_untrusted
from backend.agents.state import StockAnalysisState

ROLE = "Financial News Intelligence Analyst"
GOAL = (
    "Retrieve the latest relevant financial news articles for a given stock "
    "or market event. Summarise key developments, identify material events "
    "(earnings, M&A, regulatory actions), and flag sentiment-relevant headlines."
)
BACKSTORY = (
    "You are a veteran financial journalist turned AI analyst. You have "
    "an expert eye for distinguishing material news from noise, and you "
    "can rapidly synthesise dozens of articles into actionable intelligence."
)


def create_news_intelligence_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the News Intelligence agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        articles = state.get("articles") or []
        articles_text = "\n".join(
            f"- {a.get('title', 'N/A')} [{a.get('source', 'Unknown')}]"
            for a in articles[:15]
        ) or "No news articles available."
        task = (
            f"Analyse the following news headlines for {ticker}. "
            "Identify material events (earnings beats/misses, M&A, regulatory actions, "
            "executive changes), any conflicting signals, and overall sentiment direction. "
            "Rank headlines by investment relevance.\n\n"
            f"{wrap_untrusted('news_headlines', articles_text)}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"news_insight": response.content}

    return node
