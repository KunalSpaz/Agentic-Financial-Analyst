"""
technical_analysis_agent.py
---------------------------
LangGraph Technical Analysis node — interprets pre-computed technical
indicators as part of the stock-analysis fan-out.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StockAnalysisState

ROLE = "Technical Analysis Expert"
GOAL = (
    "Compute and interpret technical indicators (RSI, MACD, SMA 50/200, "
    "Bollinger Bands, Volume) from OHLCV data. Identify key signals such as "
    "oversold/overbought conditions, crossovers, and trend direction."
)
BACKSTORY = (
    "You are a chartered market technician (CMT) with 15 years of experience "
    "in quantitative trading. You interpret chart patterns and indicators with "
    "precision and can explain complex technical signals in plain language."
)


def create_technical_analysis_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Technical Analysis agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        signals = state.get("tech_signals") or {}
        tech_text = "\n".join(f"- {k}: {v}" for k, v in signals.items()) or "No technical signals available."
        task = (
            f"Interpret the following pre-computed technical indicators for {ticker}. "
            "Explain what the combination of signals means for near-term and medium-term "
            "price action. Identify the most important signal and flag any indicator "
            f"contradictions or divergences.\n\n{tech_text}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"technical_insight": response.content}

    return node
