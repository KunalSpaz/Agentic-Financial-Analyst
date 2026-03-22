"""
sentiment_analysis_agent.py
---------------------------
CrewAI Sentiment Analysis Agent — classifies news using FinBERT and
produces aggregated sentiment scores per ticker.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_sentiment_analysis_agent() -> Agent:
    """
    Instantiate the Sentiment Analysis Agent.

    Applies FinBERT NLP to news articles and aggregates positive/neutral/negative
    scores into a compound sentiment signal for each stock.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Financial Sentiment Analyst",
        goal=(
            "Apply FinBERT sentiment analysis to financial news articles. "
            "Produce aggregated positive, neutral, and negative scores for each "
            "ticker along with an overall compound sentiment label."
        ),
        backstory=(
            "You are an NLP specialist focused on financial text. You are trained "
            "on FinBERT and understand the nuances of financial language — including "
            "how subtle phrasing differences dramatically change market interpretation."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
