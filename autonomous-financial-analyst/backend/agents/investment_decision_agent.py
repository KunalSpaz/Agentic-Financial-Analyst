"""
investment_decision_agent.py
-----------------------------
LangGraph Investment Decision node — confirms or refines the deterministic
recommendation based on the synthesised financial analysis.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StockAnalysisState

ROLE = "Investment Decision Strategist"
GOAL = (
    "Generate precise buy/hold/sell investment recommendations with a "
    "0–100 confidence score. Base decisions on weighted signals from "
    "technical analysis (50%), news sentiment (30%), and market momentum (20%). "
    "Always provide clear, evidence-based rationale."
)
BACKSTORY = (
    "You are a portfolio manager who has generated alpha for institutional "
    "investors for over a decade. You make conviction-driven investment "
    "decisions by rigorously weighing quantitative and qualitative factors."
)


def create_investment_decision_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Investment Decision agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        score = state.get("score", 0.0)
        recommendation = state.get("recommendation", "HOLD")
        analysis = state.get("financial_analysis", "")
        task = (
            f"Based on the following financial analysis for {ticker}, confirm or refine "
            f"the preliminary recommendation of {recommendation} (confidence: {score:.1f}/100). "
            f"Provide clear rationale and key risk factors.\n\n{analysis}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"investment_decision": response.content}

    return node
