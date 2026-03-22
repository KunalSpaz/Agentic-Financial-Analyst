"""
financial_analysis_agent.py
---------------------------
CrewAI Financial Analysis Agent — synthesises all signals into a unified view.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_financial_analysis_agent() -> Agent:
    """
    Instantiate the Financial Analysis Agent.

    Synthesises technical indicators, news sentiment, and RAG-retrieved
    document context into a comprehensive financial view.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Senior Financial Analyst",
        goal=(
            "Synthesise technical indicators, news sentiment scores, fundamental data, "
            "and document intelligence into a unified, balanced financial assessment. "
            "Produce a concise analysis highlighting key risks and opportunities."
        ),
        backstory=(
            "You are a CFA charterholder with experience at top-tier investment banks. "
            "You are expert at combining quantitative signals with qualitative insights "
            "to form a holistic view of any investment opportunity."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
