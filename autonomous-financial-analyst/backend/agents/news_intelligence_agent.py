"""
news_intelligence_agent.py
--------------------------
CrewAI News Intelligence Agent — retrieves and summarises financial news.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_news_intelligence_agent() -> Agent:
    """
    Instantiate the News Intelligence Agent.

    Retrieves financial news from multiple sources and produces structured
    summaries of relevant articles per ticker.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Financial News Intelligence Analyst",
        goal=(
            "Retrieve the latest relevant financial news articles for a given stock "
            "or market event. Summarise key developments, identify material events "
            "(earnings, M&A, regulatory actions), and flag sentiment-relevant headlines."
        ),
        backstory=(
            "You are a veteran financial journalist turned AI analyst. You have "
            "an expert eye for distinguishing material news from noise, and you "
            "can rapidly synthesise dozens of articles into actionable intelligence."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
