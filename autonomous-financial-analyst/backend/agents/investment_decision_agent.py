"""
investment_decision_agent.py
-----------------------------
CrewAI Investment Decision Agent — generates buy/hold/sell recommendations
with confidence scores.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_investment_decision_agent() -> Agent:
    """
    Instantiate the Investment Decision Agent.

    Takes the synthesised financial analysis and produces a structured
    recommendation (STRONG BUY/BUY/HOLD/SELL/STRONG SELL) with a
    0–100 confidence score and clear rationale.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Investment Decision Strategist",
        goal=(
            "Generate precise buy/hold/sell investment recommendations with a "
            "0–100 confidence score. Base decisions on weighted signals from "
            "technical analysis (50%), news sentiment (30%), and market momentum (20%). "
            "Always provide clear, evidence-based rationale."
        ),
        backstory=(
            "You are a portfolio manager who has generated alpha for institutional "
            "investors for over a decade. You make conviction-driven investment "
            "decisions by rigorously weighing quantitative and qualitative factors."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
