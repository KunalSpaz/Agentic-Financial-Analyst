"""
test_crew_orchestrator.py
--------------------------
Unit tests for StockAnalysisCrew and MarketScanCrew.

All external services are mocked so tests run without network access.

Covers:
  StockAnalysisCrew.run():
    - All pipeline steps execute
    - Result dict has all required keys
    - Ticker is uppercased
    - Empty historical data → tech_signals = {}
    - CrewAI timeout → graceful fallback narrative
    - CrewAI exception → graceful fallback narrative
    - rag_context_used flag set correctly

  MarketScanCrew.scan():
    - Results sorted by confidence_score descending
    - Ranks assigned 1-indexed
    - Individual ticker failure doesn't abort the scan
    - Custom tickers list respected
    - Empty tickers → empty results
    - Default tickers from settings when none provided
"""
from __future__ import annotations

import sys
from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Mock heavy deps before any import ─────────────────────────────────────────
sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())
sys.modules.setdefault("crewai",       MagicMock())

from backend.agents.crew_orchestrator import MarketScanCrew, StockAnalysisCrew  # noqa: E402

# ── Shared test data ──────────────────────────────────────────────────────────
_MOCK_QUOTE = {
    "ticker": "AAPL", "company_name": "Apple Inc.", "price": 175.0,
    "sector": "Technology", "market_cap": 2_700_000_000_000,
}
_MOCK_TECH_SIGNALS = {
    "rsi": 42.0, "close": 175.0, "sma_50": 165.0, "sma_200": 150.0,
    "golden_cross": True, "rsi_oversold": False, "rsi_overbought": False,
}
_MOCK_SENTIMENT = {"label": "POSITIVE", "compound": 0.4, "article_count": 10}
_MOCK_SCORE = 72.5
_MOCK_BREAKDOWN = {
    "score": 72.5,
    "technical": {"score": 70.0, "detail": {}},
    "sentiment":  {"score": 75.0, "detail": {}},
    "momentum":   {"score": 68.0, "detail": {}},
}


def _make_df(rows: int = 60) -> pd.DataFrame:
    """Minimal OHLCV DataFrame."""
    dates = pd.date_range("2023-01-01", periods=rows)
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Open":   rng.uniform(160, 180, rows),
        "High":   rng.uniform(180, 200, rows),
        "Low":    rng.uniform(140, 160, rows),
        "Close":  rng.uniform(160, 180, rows),
        "Volume": rng.integers(1_000_000, 10_000_000, rows),
    }, index=dates)


@pytest.fixture
def patched_services(monkeypatch):
    """Replace every service inside crew_orchestrator with a MagicMock."""
    mds = MagicMock(); mds.get_quote.return_value = _MOCK_QUOTE
    mds.get_historical_data.return_value = _make_df()

    ns  = MagicMock(); ns.get_stock_news.return_value = []
    ss  = MagicMock(); ss.analyse_articles.return_value = _MOCK_SENTIMENT
    tas = MagicMock()
    tas.compute_indicators.side_effect = lambda df: df
    tas.get_latest_signals.return_value = _MOCK_TECH_SIGNALS

    css = MagicMock(); css.compute.return_value = (_MOCK_SCORE, _MOCK_BREAKDOWN)
    res = MagicMock(); res.get_recommendation_with_color.return_value = ("BUY", "#69F0AE")
    rag = MagicMock(); rag.retrieve.return_value = "Relevant docs here — revenue strong."
    rag._index = MagicMock(); rag._index.ntotal = 1

    monkeypatch.setattr("backend.agents.crew_orchestrator.MarketDataService",      lambda: mds)
    monkeypatch.setattr("backend.agents.crew_orchestrator.NewsService",            lambda: ns)
    monkeypatch.setattr("backend.agents.crew_orchestrator.SentimentService",       lambda: ss)
    monkeypatch.setattr("backend.agents.crew_orchestrator.TechnicalAnalysisService", lambda: tas)
    monkeypatch.setattr("backend.agents.crew_orchestrator.ConfidenceScoreService", lambda: css)
    monkeypatch.setattr("backend.agents.crew_orchestrator.RecommendationService",  lambda: res)
    monkeypatch.setattr("backend.agents.crew_orchestrator.RAGService",             lambda: rag)

    return {"mds": mds, "ns": ns, "ss": ss, "tas": tas, "css": css, "res": res, "rag": rag}


@pytest.fixture
def crew(patched_services) -> StockAnalysisCrew:
    return StockAnalysisCrew()


# ─────────────────────────────────────────────────────────────────────────────
# StockAnalysisCrew.run()
# ─────────────────────────────────────────────────────────────────────────────

class TestStockAnalysisCrewRun:
    def test_result_has_all_required_keys(self, crew):
        result = crew.run("AAPL")
        for key in (
            "ticker", "company_name", "current_price", "recommendation",
            "confidence_score", "confidence_color", "technical_signals",
            "sentiment", "score_breakdown", "narrative", "quote", "rag_context_used",
        ):
            assert key in result, f"Missing key: {key}"

    def test_ticker_uppercased(self, crew):
        result = crew.run("aapl")
        assert result["ticker"] == "AAPL"

    def test_recommendation_from_service(self, crew, patched_services):
        result = crew.run("AAPL")
        assert result["recommendation"] == "BUY"

    def test_confidence_score_from_service(self, crew, patched_services):
        result = crew.run("AAPL")
        assert result["confidence_score"] == round(_MOCK_SCORE, 2)

    def test_sentiment_from_service(self, crew, patched_services):
        result = crew.run("AAPL")
        assert result["sentiment"]["label"] == "POSITIVE"

    def test_technical_signals_from_service(self, crew, patched_services):
        result = crew.run("AAPL")
        assert result["technical_signals"] == _MOCK_TECH_SIGNALS

    def test_empty_df_skips_technical_analysis(self, crew, patched_services):
        patched_services["mds"].get_historical_data.return_value = pd.DataFrame()
        result = crew.run("AAPL")
        assert result["technical_signals"] == {}
        patched_services["tas"].compute_indicators.assert_not_called()

    def test_rag_context_used_flag_true_when_context_long_enough(self, crew, patched_services):
        # retrieve returns a 35+ char string → flag should be True
        patched_services["rag"].retrieve.return_value = "x" * 60
        result = crew.run("AAPL")
        assert result["rag_context_used"] is True

    def test_rag_context_used_flag_false_when_context_too_short(self, crew, patched_services):
        patched_services["rag"].retrieve.return_value = "short"
        result = crew.run("AAPL")
        assert result["rag_context_used"] is False

    def test_crew_timeout_returns_fallback_narrative(self, crew):
        with patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.side_effect = FuturesTimeoutError()
            mock_exec.submit.return_value = future
            result = crew.run("AAPL")
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_crew_exception_returns_fallback_narrative(self, crew):
        with patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.side_effect = RuntimeError("LLM unavailable")
            mock_exec.submit.return_value = future
            result = crew.run("AAPL")
        assert isinstance(result["narrative"], str)

    def test_fallback_narrative_contains_recommendation(self, crew, patched_services):
        with patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.side_effect = FuturesTimeoutError()
            mock_exec.submit.return_value = future
            result = crew.run("AAPL")
        assert "BUY" in result["narrative"]

    def test_all_pipeline_steps_called(self, crew, patched_services):
        crew.run("AAPL")
        patched_services["mds"].get_quote.assert_called_once()
        patched_services["mds"].get_historical_data.assert_called_once()
        patched_services["ns"].get_stock_news.assert_called_once()
        patched_services["ss"].analyse_articles.assert_called_once()
        patched_services["rag"].retrieve.assert_called_once()
        patched_services["css"].compute.assert_called_once()
        patched_services["res"].get_recommendation_with_color.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# MarketScanCrew.scan()
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketScanCrew:
    def _make_run_result(self, ticker: str, score: float) -> dict:
        return {
            "ticker": ticker,
            "recommendation": "BUY",
            "confidence_score": score,
            "current_price": 100.0,
            "quote": {"sector": "Technology"},
            "narrative": "narrative text",
        }

    def test_results_sorted_by_confidence_score_descending(self, patched_services):
        scores = {"AAPL": 72.0, "MSFT": 85.0, "NVDA": 60.0}
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.side_effect = (
            lambda t: self._make_run_result(t, scores[t])
        )
        results = scan_crew.scan(tickers=["AAPL", "MSFT", "NVDA"])
        scores_out = [r["confidence_score"] for r in results]
        assert scores_out == sorted(scores_out, reverse=True)

    def test_ranks_assigned_correctly(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.side_effect = (
            lambda t: self._make_run_result(t, 70.0)
        )
        results = scan_crew.scan(tickers=["AAPL", "MSFT", "NVDA"])
        assert [r["rank"] for r in results] == [1, 2, 3]

    def test_individual_ticker_failure_does_not_abort(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()

        def run_side_effect(ticker):
            if ticker == "MSFT":
                raise RuntimeError("data unavailable")
            return self._make_run_result(ticker, 70.0)

        scan_crew._analysis_crew.run.side_effect = run_side_effect
        results = scan_crew.scan(tickers=["AAPL", "MSFT", "NVDA"])
        tickers = [r["ticker"] for r in results]
        assert "AAPL" in tickers
        assert "NVDA" in tickers
        assert "MSFT" not in tickers

    def test_empty_tickers_returns_empty_list(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        results = scan_crew.scan(tickers=[])
        assert results == []

    def test_single_ticker_gets_rank_1(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = self._make_run_result("AAPL", 75.0)
        results = scan_crew.scan(tickers=["AAPL"])
        assert len(results) == 1
        assert results[0]["rank"] == 1

    def test_all_tickers_fail_returns_empty_list(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.side_effect = RuntimeError("all down")
        results = scan_crew.scan(tickers=["AAPL", "MSFT"])
        assert results == []

    def test_rationale_truncated_to_300_chars(self, patched_services):
        long_narrative = "x" * 1000
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = {
            **self._make_run_result("AAPL", 70.0),
            "narrative": long_narrative,
        }
        results = scan_crew.scan(tickers=["AAPL"])
        assert len(results[0]["rationale"]) <= 300
