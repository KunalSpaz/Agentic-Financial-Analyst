"""
analysis_graph.py
------------------
Wires all reasoning agents into LangGraph state graphs and exposes
high-level analysis workflows. Replaces the former CrewAI-based
crew_orchestrator.py.

Two main graphs:
    1. StockAnalysisGraph — full analysis pipeline for a single ticker
    2. MarketScanGraph     — opportunity scanning across the stock universe
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from backend.agents.document_intelligence_agent import create_document_intelligence_node
from backend.agents.financial_analysis_agent import create_financial_analysis_node
from backend.agents.investment_decision_agent import create_investment_decision_node
from backend.agents.market_data_agent import create_market_data_node
from backend.agents.news_intelligence_agent import create_news_intelligence_node
from backend.agents.opportunity_scanner_agent import create_opportunity_scanner_node
from backend.agents.report_writer_agent import create_report_writer_node
from backend.agents.sentiment_analysis_agent import create_sentiment_analysis_node
from backend.agents.state import StockAnalysisState
from backend.agents.technical_analysis_agent import create_technical_analysis_node
from backend.services.confidence_score_service import ConfidenceScoreService
from backend.services.market_data_service import MarketDataService
from backend.services.news_service import NewsService
from backend.services.rag_service import get_rag_service
from backend.services.recommendation_service import RecommendationService
from backend.services.sentiment_service import SentimentService
from backend.services.technical_analysis_service import TechnicalAnalysisService
from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Dedicated executor purely to enforce a hard wall-clock timeout on graph LLM
# calls: there is no reliable way to cancel a running synchronous call from
# another thread, so we race it against a deadline in a worker thread and
# abandon it on timeout. Pool size is configurable via
# settings.graph_max_concurrency — this is NOT a general concurrency
# limiter, it exists solely for the timeout race.
_GRAPH_EXECUTOR = ThreadPoolExecutor(
    max_workers=settings.graph_max_concurrency, thread_name_prefix="langgraph",
)

# Node names are suffixed with "_step" — LangGraph rejects a node name that
# collides with a state key, and several of ours would otherwise (e.g. the
# "financial_analysis" node writes the "financial_analysis" state key).
_FAN_OUT_NODES = (
    "market_data_step",
    "technical_analysis_step",
    "sentiment_analysis_step",
    "news_intelligence_step",
    "document_intelligence_step",
)


def _build_analysis_graph():
    """Compile the 8-node fan-out/fan-in stock analysis reasoning graph."""
    graph = StateGraph(StockAnalysisState)
    graph.add_node("market_data_step", create_market_data_node())
    graph.add_node("technical_analysis_step", create_technical_analysis_node())
    graph.add_node("sentiment_analysis_step", create_sentiment_analysis_node())
    graph.add_node("news_intelligence_step", create_news_intelligence_node())
    graph.add_node("document_intelligence_step", create_document_intelligence_node())
    graph.add_node("financial_analysis_step", create_financial_analysis_node())
    graph.add_node("investment_decision_step", create_investment_decision_node())
    graph.add_node("report_writer_step", create_report_writer_node())

    for node_name in _FAN_OUT_NODES:
        graph.add_edge(START, node_name)
        graph.add_edge(node_name, "financial_analysis_step")

    graph.add_edge("financial_analysis_step", "investment_decision_step")
    graph.add_edge("investment_decision_step", "report_writer_step")
    graph.add_edge("report_writer_step", END)
    return graph.compile()


class StockAnalysisGraph:
    """
    Orchestrates the full stock analysis pipeline for a single ticker.

    Pipeline:
        Market Data → Technical Analysis → News → Sentiment → RAG →
        Confidence Score → Recommendation
                                   │
                                   ▼
        8-agent reasoning graph (fan-out: market data / technical / sentiment /
        news / document insight → fan-in: financial analysis → investment
        decision → report writer)
    """

    def __init__(self) -> None:
        self._mds = MarketDataService()
        self._ns = NewsService()
        self._ss = SentimentService()
        self._tas = TechnicalAnalysisService()
        self._css = ConfidenceScoreService()
        self._res = RecommendationService()
        self._rag = get_rag_service()
        self._graph = _build_analysis_graph()

    def run(self, ticker: str) -> Dict[str, Any]:
        """
        Execute the complete stock analysis workflow for *ticker*.

        Runs the deterministic service pipeline, then uses the LangGraph
        reasoning graph to synthesise a narrative report.

        Args:
            ticker: Stock symbol, e.g. ``"AAPL"``.

        Returns:
            Dict containing recommendation, confidence_score, technical signals,
            sentiment, narrative report, and all intermediate data.
        """
        ticker = ticker.upper()
        logger.info("Starting StockAnalysisGraph for %s", ticker)

        # ── Step 1: Market Data ──────────────────────────────────────────
        quote = self._mds.get_quote(ticker)
        df = self._mds.get_historical_data(ticker, period="1y")

        # ── Step 2: Technical Analysis ───────────────────────────────────
        if not df.empty:
            df = self._tas.compute_indicators(df)
            tech_signals = self._tas.get_latest_signals(df)
        else:
            tech_signals = {}

        # ── Step 3: News + Sentiment ─────────────────────────────────────
        articles = self._ns.get_stock_news(
            ticker, company_name=quote.get("company_name"), max_articles=20
        )
        sentiment = self._ss.analyse_articles(articles, ticker=ticker)

        # ── Step 4: RAG Document Context ─────────────────────────────────
        rag_context = self._rag.retrieve(
            f"Financial performance and outlook for {ticker} {quote.get('company_name', '')}"
        )

        # ── Step 5: Confidence Score ──────────────────────────────────────
        score, breakdown = self._css.compute(tech_signals, sentiment)

        # ── Step 6: Recommendation ────────────────────────────────────────
        recommendation, color = self._res.get_recommendation_with_color(score)

        # ── Step 7: LangGraph Reasoning ────────────────────────────────────
        initial_state: StockAnalysisState = {
            "ticker": ticker,
            "quote": quote,
            "tech_signals": tech_signals,
            "articles": articles,
            "sentiment": sentiment,
            "rag_context": rag_context,
            "score": score,
            "breakdown": breakdown,
            "recommendation": recommendation,
        }
        narrative = self._run_graph_with_timeout(initial_state)

        result = {
            "ticker": ticker,
            "company_name": quote.get("company_name", ticker),
            "current_price": quote.get("price"),
            "recommendation": recommendation,
            "confidence_score": round(score, 2),
            "confidence_color": color,
            "technical_signals": tech_signals,
            "sentiment": sentiment,
            "score_breakdown": breakdown,
            "narrative": narrative,
            "quote": quote,
            "rag_context_used": bool(rag_context and len(rag_context) > 50),
        }

        logger.info(
            "StockAnalysisGraph completed for %s: %s (score=%.1f)",
            ticker, recommendation, score,
        )
        return result

    def _run_graph_with_timeout(self, initial_state: StockAnalysisState) -> str:
        """Invoke the reasoning graph with a hard wall-clock timeout, falling
        back to a templated narrative on timeout or failure."""
        try:
            future = _GRAPH_EXECUTOR.submit(self._graph.invoke, initial_state)
            final_state = future.result(timeout=settings.graph_timeout_seconds)
            narrative = final_state.get("narrative")
            if narrative:
                return narrative
        except FuturesTimeoutError:
            logger.error(
                "LangGraph narrative timed out after %ds for %s",
                settings.graph_timeout_seconds, initial_state.get("ticker"),
            )
        except Exception as exc:
            logger.error("LangGraph narrative generation failed: %s", exc)
        return self._fallback_narrative(initial_state)

    @staticmethod
    def _fallback_narrative(state: StockAnalysisState) -> str:
        tech_signals = state.get("tech_signals") or {}
        sentiment = state.get("sentiment") or {}
        return (
            f"## {state.get('ticker')} Analysis\n\n"
            f"**Recommendation:** {state.get('recommendation')}\n"
            f"**Confidence Score:** {state.get('score', 0):.1f}/100\n\n"
            f"Technical signals indicate {'bullish trend' if tech_signals.get('golden_cross') else 'mixed signals'}. "
            f"Market sentiment is {sentiment.get('label', 'NEUTRAL')}."
        )


class MarketScanGraph:
    """
    Scans the full stock universe, ranks opportunities by confidence score,
    then runs the opportunity-scanner reasoning node once over the full
    ranked list to produce a thematic market narrative.
    """

    def __init__(self) -> None:
        self._analysis_graph = StockAnalysisGraph()
        self._opportunity_node = create_opportunity_scanner_node()

    def scan(self, tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run analysis on each ticker in the universe and rank by confidence score.

        Args:
            tickers: List of tickers to scan. Defaults to ``settings.stock_universe_list``.

        Returns:
            Dict with ``opportunities`` (list, sorted by confidence_score descending)
            and ``market_narrative`` (thematic summary of the ranked list).
        """
        if tickers is None:
            tickers = settings.stock_universe_list

        opportunities = []
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info("Scanning %s (%d/%d)…", ticker, i, len(tickers))
                result = self._analysis_graph.run(ticker)
                opportunities.append({
                    "ticker": ticker,
                    "rank": 0,  # assigned after sort
                    "recommendation": result["recommendation"],
                    "confidence_score": result["confidence_score"],
                    "current_price": result.get("current_price"),
                    "sector": result.get("quote", {}).get("sector"),
                    "rationale": result.get("narrative", "")[:300],
                })
            except Exception as exc:
                logger.error("Failed to scan %s: %s", ticker, exc)

        opportunities.sort(key=lambda x: x["confidence_score"], reverse=True)
        for i, opp in enumerate(opportunities, 1):
            opp["rank"] = i

        market_narrative = ""
        if opportunities:
            try:
                out = self._opportunity_node({"opportunities": opportunities})
                market_narrative = out.get("market_narrative", "")
            except Exception as exc:
                logger.error("Opportunity scanner narrative failed: %s", exc)

        return {"opportunities": opportunities, "market_narrative": market_narrative}
