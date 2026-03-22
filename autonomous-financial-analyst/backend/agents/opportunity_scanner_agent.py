"""
opportunity_scanner_agent.py
-----------------------------
CrewAI Opportunity Scanner Agent — scans the stock universe for top
investment opportunities ranked by confidence score.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_opportunity_scanner_agent() -> Agent:
    """
    Instantiate the Opportunity Scanner Agent.

    Scans AAPL, MSFT, NVDA, TSLA, AMZN, META, GOOGL, AMD, NFLX, INTC and
    ranks them by confidence score to surface the highest-conviction plays.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Market Opportunity Scanner",
        goal=(
            "Systematically scan the predefined stock universe (AAPL, MSFT, NVDA, "
            "TSLA, AMZN, META, GOOGL, AMD, NFLX, INTC), compute confidence scores "
            "for each, and produce a ranked list of the best current investment "
            "opportunities with clear rationale."
        ),
        backstory=(
            "You are a quantitative research analyst who built systematic screening "
            "tools at a hedge fund. You excel at filtering the universe of stocks "
            "to surface high-conviction opportunities efficiently."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
