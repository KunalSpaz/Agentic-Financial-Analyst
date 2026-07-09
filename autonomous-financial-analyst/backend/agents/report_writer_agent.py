"""
report_writer_agent.py
-----------------------
LangGraph Report Writer node — the terminal node of the stock-analysis
graph. Turns the analysis + decision into the final markdown narrative
returned to the caller.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt
from backend.agents.state import StockAnalysisState

ROLE = "Financial Report Writer"
GOAL = (
    "Transform structured investment analysis into clear, professional "
    "narrative reports. Reports should include an executive summary, "
    "technical analysis section, fundamental highlights, risk factors, "
    "and a final recommendation with confidence score."
)
BACKSTORY = (
    "You are a former Goldman Sachs equity research analyst with a talent "
    "for translating complex quantitative analysis into compelling written "
    "narratives that institutional investors trust."
)


def create_report_writer_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Report Writer agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        analysis = state.get("financial_analysis", "")
        decision = state.get("investment_decision", "")
        task = (
            f"Write a professional investment report for {ticker} based on the analysis "
            "and decision below. Structure: Executive Summary, Technical Analysis, "
            "News & Sentiment, Document Intelligence, Risk Factors, Final Recommendation.\n\n"
            f"## Financial Analysis\n{analysis}\n\n## Investment Decision\n{decision}"
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"narrative": response.content}

    return node
