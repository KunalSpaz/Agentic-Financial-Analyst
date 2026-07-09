"""
document_intelligence_agent.py
-------------------------------
LangGraph Document Intelligence node — reasons over RAG-retrieved financial
document context (earnings transcripts, reports, analyst notes) as part of
the stock-analysis fan-out.
"""
from __future__ import annotations

from typing import Callable, Dict

from backend.agents.llm import get_chat_model
from backend.agents.prompts import build_prompt, wrap_untrusted
from backend.agents.state import StockAnalysisState

ROLE = "Financial Document Intelligence Analyst"
GOAL = (
    "Retrieve and reason over relevant passages from earnings transcripts, "
    "financial reports, and analyst notes stored in the vector knowledge base. "
    "Provide management commentary context, forward guidance, and risk disclosures "
    "that enhance the overall investment analysis."
)
BACKSTORY = (
    "You are a fundamental research analyst specialising in earnings transcript "
    "analysis. You read every 10-K, 10-Q, and earnings call transcript and have "
    "an extraordinary ability to identify management tone shifts and forward-looking "
    "statements that others miss."
)


def create_document_intelligence_node() -> Callable[[StockAnalysisState], Dict[str, str]]:
    """Build the LangGraph node function for the Document Intelligence agent."""
    llm = get_chat_model()

    def node(state: StockAnalysisState) -> Dict[str, str]:
        ticker = state.get("ticker", "")
        rag_context = state.get("rag_context", "")
        task = (
            f"Extract key financial insights from the following document context for {ticker}.\n\n"
            f"{wrap_untrusted('retrieved_documents', rag_context)}\n\n"
            "Respond with a bullet-point list of key insights relevant to this investment."
        )
        response = llm.invoke(build_prompt(ROLE, GOAL, BACKSTORY, task))
        return {"document_insight": response.content}

    return node
