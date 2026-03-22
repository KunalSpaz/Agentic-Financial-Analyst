"""
report_writer_agent.py
-----------------------
CrewAI Report Writer Agent — creates professional narrative analysis reports.
"""
from __future__ import annotations
from crewai import Agent
from backend.utils.config import settings


def create_report_writer_agent() -> Agent:
    """
    Instantiate the Report Writer Agent.

    Transforms structured analysis outputs into clear, professional
    investment research reports suitable for end-user consumption.

    Returns:
        Configured :class:`crewai.Agent` instance.
    """
    return Agent(
        role="Financial Report Writer",
        goal=(
            "Transform structured investment analysis into clear, professional "
            "narrative reports. Reports should include an executive summary, "
            "technical analysis section, fundamental highlights, risk factors, "
            "and a final recommendation with confidence score."
        ),
        backstory=(
            "You are a former Goldman Sachs equity research analyst with a talent "
            "for translating complex quantitative analysis into compelling written "
            "narratives that institutional investors trust."
        ),
        verbose=True,
        allow_delegation=False,
        llm=settings.openai_model,
    )
