"""
portfolio_risk_agent.py
-----------------------
CrewAI Portfolio Risk Agent — analyses portfolio risk metrics.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_portfolio_risk_agent() -> Agent:
    """
    Instantiate the Portfolio Risk Agent.

    Computes and interprets portfolio volatility, beta, correlation,
    sector exposure, drawdown, and VaR metrics.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Portfolio Risk Manager",
        goal=(
            "Analyse the risk profile of any submitted portfolio. Compute "
            "volatility, beta, Sharpe ratio, max drawdown, VaR(95%), correlation "
            "matrix, and sector exposure. Provide plain-language risk interpretation "
            "and actionable diversification recommendations."
        ),
        backstory=(
            "You are a chief risk officer with 20 years of experience managing "
            "multi-billion dollar portfolio risk. You specialise in identifying "
            "hidden concentrations, tail risks, and correlation breakdown scenarios."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
