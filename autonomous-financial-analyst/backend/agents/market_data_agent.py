"""
market_data_agent.py
--------------------
CrewAI Market Data Agent — responsible for fetching and structuring
OHLCV data and stock metadata for downstream agents.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_market_data_agent() -> Agent:
    """
    Instantiate the Market Data Agent.

    This agent's role is to retrieve, validate, and structure stock market
    data including price history, volume, and fundamental metadata.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Market Data Specialist",
        goal=(
            "Fetch accurate, up-to-date OHLCV market data and fundamental "
            "metrics for any given stock ticker. Provide clean, structured "
            "data suitable for technical analysis."
        ),
        backstory=(
            "You are a quantitative data engineer with 10 years of experience "
            "fetching and validating financial market data. You ensure data quality "
            "and completeness before it reaches any analysis pipeline."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
