"""
technical_analysis_agent.py
---------------------------
CrewAI Technical Analysis Agent — computes and interprets technical indicators.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_technical_analysis_agent() -> Agent:
    """
    Instantiate the Technical Analysis Agent.

    Computes RSI, MACD, Moving Averages (50/200), Bollinger Bands, and
    Volume Trend. Interprets signals and produces a structured technical summary.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Technical Analysis Expert",
        goal=(
            "Compute and interpret technical indicators (RSI, MACD, SMA 50/200, "
            "Bollinger Bands, Volume) from OHLCV data. Identify key signals such as "
            "oversold/overbought conditions, crossovers, and trend direction."
        ),
        backstory=(
            "You are a chartered market technician (CMT) with 15 years of experience "
            "in quantitative trading. You interpret chart patterns and indicators with "
            "precision and can explain complex technical signals in plain language."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
