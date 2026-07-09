"""
test_analysis_graph.py
-----------------------
Unit tests for StockAnalysisGraph and MarketScanGraph (LangGraph-based,
replaces the former CrewAI StockAnalysisCrew/MarketScanCrew).

All external services and the LangGraph reasoning graph itself are mocked
so tests run without network access or real LLM calls.

Covers:
  StockAnalysisGraph.run():
    - All deterministic pipeline steps execute
    - Result dict has all required keys
    - Ticker is uppercased
    - Empty historical data -> tech_signals = {}
    - Graph timeout -> graceful fallback narrative
    - Graph exception -> graceful fallback narrative
    - rag_context_used flag set correctly

  MarketScanGraph.scan():
    - Results sorted by confidence_score descending
    - Ranks assigned 1-indexed
    - Individual ticker failure doesn't abort the scan
    - Empty tickers -> empty opportunities, no narrative call
    - Default tickers from settings when none provided
    - market_narrative populated from the opportunity-scanner node
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

from backend.agents.analysis_graph import MarketScanGraph, StockAnalysisGraph  # noqa: E402
from backend.utils.config import settings                                       # noqa: E402

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
    """Replace every deterministic service inside analysis_graph with a MagicMock."""
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

    monkeypatch.setattr("backend.agents.analysis_graph.MarketDataService",      lambda: mds)
    monkeypatch.setattr("backend.agents.analysis_graph.NewsService",            lambda: ns)
    monkeypatch.setattr("backend.agents.analysis_graph.SentimentService",       lambda: ss)
    monkeypatch.setattr("backend.agents.analysis_graph.TechnicalAnalysisService", lambda: tas)
    monkeypatch.setattr("backend.agents.analysis_graph.ConfidenceScoreService", lambda: css)
    monkeypatch.setattr("backend.agents.analysis_graph.RecommendationService",  lambda: res)
    monkeypatch.setattr("backend.agents.analysis_graph.get_rag_service",        lambda: rag)

    return {"mds": mds, "ns": ns, "ss": ss, "tas": tas, "css": css, "res": res, "rag": rag}


@pytest.fixture
def graph(patched_services) -> StockAnalysisGraph:
    return StockAnalysisGraph()


def _mock_graph_result(narrative: str = "LLM narrative."):
    """Mimic the final LangGraph state dict returned by ``compiled_graph.invoke``."""
    return {"narrative": narrative}


# ─────────────────────────────────────────────────────────────────────────────
# StockAnalysisGraph.run()
# ─────────────────────────────────────────────────────────────────────────────

class TestStockAnalysisGraphRun:
    def test_result_has_all_required_keys(self, graph):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        for key in (
            "ticker", "company_name", "current_price", "recommendation",
            "confidence_score", "confidence_color", "technical_signals",
            "sentiment", "score_breakdown", "narrative", "quote", "rag_context_used",
        ):
            assert key in result, f"Missing key: {key}"

    def test_ticker_uppercased(self, graph):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("aapl")
        assert result["ticker"] == "AAPL"

    def test_recommendation_from_service(self, graph, patched_services):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert result["recommendation"] == "BUY"

    def test_confidence_score_from_service(self, graph, patched_services):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert result["confidence_score"] == round(_MOCK_SCORE, 2)

    def test_technical_signals_from_service(self, graph, patched_services):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert result["technical_signals"] == _MOCK_TECH_SIGNALS

    def test_empty_df_skips_technical_analysis(self, graph, patched_services):
        patched_services["mds"].get_historical_data.return_value = pd.DataFrame()
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert result["technical_signals"] == {}
        patched_services["tas"].compute_indicators.assert_not_called()

    def test_rag_context_used_flag_true_when_context_long_enough(self, graph, patched_services):
        patched_services["rag"].retrieve.return_value = "x" * 60
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert result["rag_context_used"] is True

    def test_rag_context_used_flag_false_when_context_too_short(self, graph, patched_services):
        patched_services["rag"].retrieve.return_value = "short"
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert result["rag_context_used"] is False

    def test_graph_timeout_returns_fallback_narrative(self, graph):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.side_effect = FuturesTimeoutError()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_graph_exception_returns_fallback_narrative(self, graph):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.side_effect = RuntimeError("LLM unavailable")
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert isinstance(result["narrative"], str)

    def test_empty_narrative_falls_back(self, graph):
        """A final state with no 'narrative' key must also trigger the fallback."""
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = {}
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert isinstance(result["narrative"], str) and len(result["narrative"]) > 0

    def test_fallback_narrative_contains_recommendation(self, graph, patched_services):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.side_effect = FuturesTimeoutError()
            mock_exec.submit.return_value = future
            result = graph.run("AAPL")
        assert "BUY" in result["narrative"]

    def test_all_pipeline_steps_called(self, graph, patched_services):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            graph.run("AAPL")
        patched_services["mds"].get_quote.assert_called_once()
        patched_services["mds"].get_historical_data.assert_called_once()
        patched_services["ns"].get_stock_news.assert_called_once()
        patched_services["ss"].analyse_articles.assert_called_once()
        patched_services["rag"].retrieve.assert_called_once()
        patched_services["css"].compute.assert_called_once()
        patched_services["res"].get_recommendation_with_color.assert_called_once()

    def test_graph_invoked_with_initial_state_containing_ticker(self, graph, patched_services):
        with patch("backend.agents.analysis_graph._GRAPH_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = _mock_graph_result()
            mock_exec.submit.return_value = future
            graph.run("AAPL")
        submitted_state = mock_exec.submit.call_args.args[1]
        assert submitted_state["ticker"] == "AAPL"
        assert submitted_state["recommendation"] == "BUY"


# ─────────────────────────────────────────────────────────────────────────────
# MarketScanGraph.scan()
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketScanGraph:
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
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.side_effect = (
            lambda t: self._make_run_result(t, scores[t])
        )
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=["AAPL", "MSFT", "NVDA"])
        scores_out = [r["confidence_score"] for r in result["opportunities"]]
        assert scores_out == sorted(scores_out, reverse=True)

    def test_ranks_assigned_correctly(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.side_effect = (
            lambda t: self._make_run_result(t, 70.0)
        )
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=["AAPL", "MSFT", "NVDA"])
        assert [r["rank"] for r in result["opportunities"]] == [1, 2, 3]

    def test_individual_ticker_failure_does_not_abort(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()

        def run_side_effect(ticker):
            if ticker == "MSFT":
                raise RuntimeError("data unavailable")
            return self._make_run_result(ticker, 70.0)

        scan_graph._analysis_graph.run.side_effect = run_side_effect
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=["AAPL", "MSFT", "NVDA"])
        tickers = [r["ticker"] for r in result["opportunities"]]
        assert "AAPL" in tickers
        assert "NVDA" in tickers
        assert "MSFT" not in tickers

    def test_empty_tickers_returns_empty_opportunities_and_skips_narrative(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": "should not be called"})
        result = scan_graph.scan(tickers=[])
        assert result["opportunities"] == []
        assert result["market_narrative"] == ""
        scan_graph._opportunity_node.assert_not_called()

    def test_single_ticker_gets_rank_1(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.return_value = self._make_run_result("AAPL", 75.0)
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=["AAPL"])
        assert len(result["opportunities"]) == 1
        assert result["opportunities"][0]["rank"] == 1

    def test_all_tickers_fail_returns_empty_list(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.side_effect = RuntimeError("all down")
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=["AAPL", "MSFT"])
        assert result["opportunities"] == []

    def test_rationale_truncated_to_300_chars(self, patched_services):
        long_narrative = "x" * 1000
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.return_value = {
            **self._make_run_result("AAPL", 70.0),
            "narrative": long_narrative,
        }
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=["AAPL"])
        assert len(result["opportunities"][0]["rationale"]) <= 300

    def test_default_tickers_from_settings(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.side_effect = lambda t: self._make_run_result(t, 70.0)
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": ""})
        result = scan_graph.scan(tickers=None)
        scanned = {r["ticker"] for r in result["opportunities"]}
        assert scanned == set(settings.stock_universe_list)

    def test_market_narrative_populated_from_opportunity_node(self, patched_services):
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.return_value = self._make_run_result("AAPL", 70.0)
        scan_graph._opportunity_node = MagicMock(return_value={"market_narrative": "Tech-heavy top picks."})
        result = scan_graph.scan(tickers=["AAPL"])
        assert result["market_narrative"] == "Tech-heavy top picks."

    def test_opportunity_node_failure_leaves_narrative_empty(self, patched_services):
        """A failing narrative synthesis step must not abort the whole scan."""
        scan_graph = MarketScanGraph()
        scan_graph._analysis_graph = MagicMock()
        scan_graph._analysis_graph.run.return_value = self._make_run_result("AAPL", 70.0)
        scan_graph._opportunity_node = MagicMock(side_effect=RuntimeError("LLM down"))
        result = scan_graph.scan(tickers=["AAPL"])
        assert result["market_narrative"] == ""
        assert len(result["opportunities"]) == 1
