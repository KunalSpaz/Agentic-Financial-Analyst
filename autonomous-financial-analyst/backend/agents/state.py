"""
state.py
--------
TypedDict state definitions shared by every LangGraph graph in this package.

A LangGraph node receives the full state and returns a partial dict of the
keys it updates; LangGraph merges that partial dict back into the running
state. Because each reasoning node below writes to its own dedicated key
(``tech_insight``, ``news_insight``, ...), no custom reducer is needed for
the fan-out/fan-in sections — the default "last write wins per key" merge
is exactly right since no two parallel nodes ever write the same key.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class StockAnalysisState(TypedDict, total=False):
    """State threaded through :class:`backend.agents.analysis_graph.StockAnalysisGraph`."""

    # Inputs / deterministic pipeline outputs (populated before the graph runs)
    ticker: str
    quote: Dict[str, Any]
    tech_signals: Dict[str, Any]
    articles: List[Dict[str, Any]]
    sentiment: Dict[str, Any]
    rag_context: str
    score: float
    breakdown: Dict[str, Any]
    recommendation: str

    # Fan-out reasoning node outputs (each node writes exactly one key)
    market_data_insight: str
    technical_insight: str
    sentiment_insight: str
    news_insight: str
    document_insight: str

    # Fan-in reasoning node outputs
    financial_analysis: str
    investment_decision: str
    narrative: str


class BacktestNarrativeState(TypedDict, total=False):
    ticker: str
    backtest_result: Dict[str, Any]
    narrative: str


class PortfolioRiskNarrativeState(TypedDict, total=False):
    risk_result: Dict[str, Any]
    narrative: str


class StrategyOptimizationNarrativeState(TypedDict, total=False):
    ticker: str
    objective: str
    optimization_result: Dict[str, Any]
    narrative: str


class OpportunityScanState(TypedDict, total=False):
    opportunities: List[Dict[str, Any]]
    market_narrative: str
