"""
crew_orchestrator.py
---------------------
Wires all CrewAI agents into crews and exposes high-level analysis workflows.

Two main crews:
    1. StockAnalysisCrew  — full analysis pipeline for a single ticker
    2. MarketScanCrew     — opportunity scanning across the stock universe
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Dict, Optional

_CREW_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="crewai")
_CREW_TIMEOUT = 300  # seconds — 5 minutes max per narrative generation

from crewai import Crew, Task

from backend.agents.document_intelligence_agent import create_document_intelligence_agent
from backend.agents.financial_analysis_agent import create_financial_analysis_agent
from backend.agents.investment_decision_agent import create_investment_decision_agent
from backend.agents.news_intelligence_agent import create_news_intelligence_agent
from backend.agents.report_writer_agent import create_report_writer_agent
from backend.agents.technical_analysis_agent import create_technical_analysis_agent
from backend.services.market_data_service import MarketDataService
from backend.services.news_service import NewsService
from backend.services.sentiment_service import SentimentService
from backend.services.technical_analysis_service import TechnicalAnalysisService
from backend.services.confidence_score_service import ConfidenceScoreService
from backend.services.recommendation_service import RecommendationService
from backend.services.rag_service import RAGService
from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class StockAnalysisCrew:
    """
    Orchestrates the full stock analysis pipeline for a single ticker.

    Pipeline:
        Market Data → Technical Analysis → News → Sentiment →
        Document Intelligence → Financial Analysis → Investment Decision →
        Report Writer
    """

    def __init__(self) -> None:
        self._mds = MarketDataService()
        self._ns = NewsService()
        self._ss = SentimentService()
        self._tas = TechnicalAnalysisService()
        self._css = ConfidenceScoreService()
        self._res = RecommendationService()
        self._rag = RAGService()

    def run(self, ticker: str) -> Dict[str, Any]:
        """
        Execute the complete stock analysis workflow for *ticker*.

        This method runs the full service pipeline and uses GPT-4 via CrewAI
        to synthesise a narrative report.

        Args:
            ticker: Stock symbol, e.g. ``"AAPL"``.

        Returns:
            Dict containing recommendation, confidence_score, technical signals,
            sentiment, narrative report, and all intermediate data.
        """
        ticker = ticker.upper()
        logger.info("Starting StockAnalysisCrew for %s", ticker)

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

        # ── Step 7: CrewAI Narrative Report ──────────────────────────────
        narrative = self._generate_narrative(
            ticker, quote, tech_signals, sentiment, articles, score, recommendation, rag_context
        )

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
            "StockAnalysisCrew completed for %s: %s (score=%.1f)",
            ticker, recommendation, score,
        )
        return result

    def _generate_narrative(
        self,
        ticker: str,
        quote: Dict[str, Any],
        tech_signals: Dict[str, Any],
        sentiment: Dict[str, Any],
        articles: list,
        score: float,
        recommendation: str,
        rag_context: str,
    ) -> str:
        """
        Use 6 CrewAI agents to generate a narrative investment report.

        Pipeline (parallel inputs → synthesis → decision → report):
            doc_agent     → extracts insights from RAG-retrieved documents
            news_agent    → synthesises raw articles into structured intelligence
            tech_agent    → interprets pre-computed technical indicators
              ↓ context from all three above
            analyst       → synthesises everything into a unified financial view
              ↓
            decision_maker→ confirms/refines the investment recommendation
              ↓
            writer        → produces the final markdown report
        """
        # ── Agents ────────────────────────────────────────────────────────────
        doc_agent = create_document_intelligence_agent()
        news_agent = create_news_intelligence_agent()
        tech_agent = create_technical_analysis_agent()
        financial_analyst = create_financial_analysis_agent()
        decision_maker = create_investment_decision_agent()
        writer = create_report_writer_agent()

        # ── Format inputs ─────────────────────────────────────────────────────
        articles_text = "\n".join(
            f"- {a.get('title', 'N/A')} [{a.get('source', 'Unknown')}]"
            for a in (articles or [])[:15]
        ) or "No news articles available."

        tech_text = "\n".join(
            f"- {k}: {v}" for k, v in tech_signals.items()
        ) or "No technical signals available."

        meta = (
            f"Ticker: {ticker} | Price: ${quote.get('price', 'N/A')} | "
            f"Sector: {quote.get('sector', 'N/A')} | "
            f"Sentiment: {sentiment.get('label', 'NEUTRAL')} "
            f"(compound={sentiment.get('compound', 0):.3f}) | "
            f"Confidence: {score:.1f}/100 | Recommendation: {recommendation}"
        )

        # ── Tasks ─────────────────────────────────────────────────────────────
        task_doc = Task(
            description=(
                f"Extract key financial insights from the following document context for {ticker}.\n\n"
                f"{rag_context[:800] if rag_context else 'No documents available.'}"
            ),
            agent=doc_agent,
            expected_output="Bullet-point list of key insights from financial documents relevant to this investment.",
        )

        task_news = Task(
            description=(
                f"Analyse the following news headlines for {ticker}. "
                "Identify material events (earnings beats/misses, M&A, regulatory actions, "
                "executive changes), any conflicting signals, and overall sentiment direction. "
                "Rank headlines by investment relevance.\n\n"
                f"{articles_text}"
            ),
            agent=news_agent,
            expected_output=(
                "Structured news intelligence: key themes, material events flagged, "
                "sentiment direction, and any contradictions between headlines."
            ),
        )

        task_tech = Task(
            description=(
                f"Interpret the following pre-computed technical indicators for {ticker}. "
                "Explain what the combination of signals means for near-term and medium-term "
                "price action. Identify the most important signal and flag any indicator "
                "contradictions or divergences.\n\n"
                f"{tech_text}\n\n{meta}"
            ),
            agent=tech_agent,
            expected_output=(
                "Technical narrative: interpretation of the indicator combination, "
                "key signal identified, trend direction, and any divergences noted."
            ),
        )

        task_analysis = Task(
            description=(
                f"Synthesise the document insights, news intelligence, and technical "
                f"interpretation for {ticker} into a unified financial assessment.\n\n{meta}"
            ),
            agent=financial_analyst,
            expected_output="Comprehensive financial analysis: balanced view of risks and opportunities.",
            context=[task_doc, task_news, task_tech],
        )

        task_decision = Task(
            description=f"Based on the full analysis, confirm or refine the investment decision for {ticker}.",
            agent=decision_maker,
            expected_output="Final investment recommendation with clear rationale and key risk factors.",
            context=[task_analysis],
        )

        task_report = Task(
            description=(
                f"Write a professional investment report for {ticker} based on all preceding analysis. "
                "Structure: Executive Summary, Technical Analysis, News & Sentiment, "
                "Document Intelligence, Risk Factors, Final Recommendation."
            ),
            agent=writer,
            expected_output="Professional investment report in markdown format.",
            context=[task_analysis, task_decision],
        )

        crew = Crew(
            agents=[doc_agent, news_agent, tech_agent, financial_analyst, decision_maker, writer],
            tasks=[task_doc, task_news, task_tech, task_analysis, task_decision, task_report],
            verbose=settings.app_env == "development",
        )

        try:
            future = _CREW_EXECUTOR.submit(crew.kickoff)
            result = future.result(timeout=_CREW_TIMEOUT)
            return str(result)
        except FuturesTimeoutError:
            logger.error("CrewAI narrative timed out after %ds for %s", _CREW_TIMEOUT, ticker)
        except Exception as exc:
            logger.error("CrewAI narrative generation failed: %s", exc)
        return (
            f"## {ticker} Analysis\n\n"
            f"**Recommendation:** {recommendation}\n"
            f"**Confidence Score:** {score:.1f}/100\n\n"
            f"Technical signals indicate {'bullish trend' if tech_signals.get('golden_cross') else 'mixed signals'}. "
            f"Market sentiment is {sentiment.get('label', 'NEUTRAL')}."
        )


class MarketScanCrew:
    """
    Scans the full stock universe and ranks opportunities by confidence score.
    """

    def __init__(self) -> None:
        self._analysis_crew = StockAnalysisCrew()

    def scan(self, tickers: Optional[list] = None) -> list:
        """
        Run analysis on each ticker in the universe and rank by confidence score.

        Args:
            tickers: List of tickers to scan. Defaults to ``settings.stock_universe_list``.

        Returns:
            List of opportunity dicts sorted by confidence_score descending.
        """
        if tickers is None:
            tickers = settings.stock_universe_list

        opportunities = []
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info("Scanning %s (%d/%d)…", ticker, i, len(tickers))
                result = self._analysis_crew.run(ticker)
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

        return opportunities
