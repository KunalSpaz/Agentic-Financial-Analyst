"""
strategy_optimization_agent.py
-------------------------------
CrewAI Strategy Optimization Agent — automatically tunes trading parameters.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_strategy_optimization_agent() -> Agent:
    """
    Instantiate the Strategy Optimization Agent.

    Uses grid search to tune RSI thresholds, MACD confirmation, and
    moving average windows to maximise return, Sharpe ratio, or minimise drawdown.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Strategy Optimization Engineer",
        goal=(
            "Automatically optimise trading strategy parameters through grid search. "
            "Identify the parameter configuration that best achieves the chosen "
            "objective (maximize return, maximize Sharpe, minimize drawdown) while "
            "avoiding overfitting."
        ),
        backstory=(
            "You are a machine learning engineer who specialises in financial strategy "
            "optimisation. You understand the risk of data snooping bias and always "
            "validate results out-of-sample before recommending parameter changes."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
