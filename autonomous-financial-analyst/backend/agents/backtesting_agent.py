"""
backtesting_agent.py
---------------------
CrewAI Backtesting Agent — orchestrates historical strategy simulations.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_backtesting_agent() -> Agent:
    """
    Instantiate the Backtesting Agent.

    Runs historical strategy simulations and interprets performance metrics
    including total return, Sharpe ratio, max drawdown, and win rate.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Quantitative Backtesting Specialist",
        goal=(
            "Run rigorous historical backtests of trading strategies. Accurately "
            "compute performance metrics (return, Sharpe, drawdown, win rate) and "
            "provide clear interpretation of strategy strengths and weaknesses."
        ),
        backstory=(
            "You are a quant researcher specialising in strategy simulation. You "
            "have backtested hundreds of strategies and understand the common pitfalls "
            "of overfitting, look-ahead bias, and transaction cost underestimation."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
