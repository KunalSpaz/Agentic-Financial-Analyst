"""
document_intelligence_agent.py
-------------------------------
CrewAI Document Intelligence Agent — retrieves relevant financial document
context via RAG before contributing to analysis.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_document_intelligence_agent() -> Agent:
    """
    Instantiate the Document Intelligence Agent.

    Uses FAISS-based RAG to retrieve relevant passages from earnings call
    transcripts, financial reports, and analyst commentary.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Financial Document Intelligence Analyst",
        goal=(
            "Retrieve and reason over relevant passages from earnings transcripts, "
            "financial reports, and analyst notes stored in the vector knowledge base. "
            "Provide management commentary context, forward guidance, and risk disclosures "
            "that enhance the overall investment analysis."
        ),
        backstory=(
            "You are a fundamental research analyst specialising in earnings transcript "
            "analysis. You read every 10-K, 10-Q, and earnings call transcript and have "
            "an extraordinary ability to identify management tone shifts and forward-looking "
            "statements that others miss."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
