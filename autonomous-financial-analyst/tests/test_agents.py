"""
test_agents.py
--------------
Unit tests for all CrewAI agent factory functions and crew orchestrator
task wiring / narrative generation.

Covers:
  Agent factories (12 agents):
    - Correct role string
    - allow_delegation=False
    - verbose=True
    - llm=settings.openai_model
    - non-empty goal and backstory

  StockAnalysisCrew._generate_narrative() wiring:
    - Crew instantiated with exactly 6 agents and 6 tasks
    - task_doc/task_news/task_tech descriptions contain ticker
    - task_analysis.context = [task_doc, task_news, task_tech]
    - task_decision.context = [task_analysis]
    - task_report.context = [task_analysis, task_decision]
    - every task has a non-empty expected_output
    - successful kickoff result stored as narrative (str-cast)
    - crew.kickoff submitted to thread executor

  MarketScanCrew output format:
    - each result has all required fields
    - sector sourced from quote dict
    - confidence_score preserved
    - default tickers from settings.stock_universe_list
"""
from __future__ import annotations

import sys
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

from backend.agents.backtesting_agent import create_backtesting_agent               # noqa: E402
from backend.agents.document_intelligence_agent import create_document_intelligence_agent  # noqa: E402
from backend.agents.financial_analysis_agent import create_financial_analysis_agent   # noqa: E402
from backend.agents.investment_decision_agent import create_investment_decision_agent  # noqa: E402
from backend.agents.market_data_agent import create_market_data_agent               # noqa: E402
from backend.agents.news_intelligence_agent import create_news_intelligence_agent   # noqa: E402
from backend.agents.opportunity_scanner_agent import create_opportunity_scanner_agent  # noqa: E402
from backend.agents.portfolio_risk_agent import create_portfolio_risk_agent         # noqa: E402
from backend.agents.report_writer_agent import create_report_writer_agent           # noqa: E402
from backend.agents.sentiment_analysis_agent import create_sentiment_analysis_agent   # noqa: E402
from backend.agents.strategy_optimization_agent import create_strategy_optimization_agent  # noqa: E402
from backend.agents.technical_analysis_agent import create_technical_analysis_agent   # noqa: E402
from backend.agents.crew_orchestrator import MarketScanCrew, StockAnalysisCrew     # noqa: E402
from backend.utils.config import settings                                           # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agent_kwargs(module_path: str, factory_fn) -> dict:
    """Patch Agent in the given module, call factory_fn, return call kwargs."""
    with patch(f"{module_path}.Agent") as mock_cls:
        factory_fn()
        return mock_cls.call_args.kwargs


def _make_df(rows: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows)
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "Open":   rng.uniform(160, 180, rows),
        "High":   rng.uniform(180, 200, rows),
        "Low":    rng.uniform(140, 160, rows),
        "Close":  rng.uniform(160, 180, rows),
        "Volume": rng.integers(1_000_000, 10_000_000, rows),
    }, index=dates)


_MOCK_QUOTE = {
    "ticker": "TSLA", "company_name": "Tesla Inc.",
    "price": 250.0, "sector": "Consumer Cyclical", "market_cap": 800_000_000_000,
}

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def patched_services(monkeypatch):
    mds = MagicMock()
    mds.get_quote.return_value = _MOCK_QUOTE
    mds.get_historical_data.return_value = _make_df()
    ns = MagicMock(); ns.get_stock_news.return_value = []
    ss = MagicMock(); ss.analyse_articles.return_value = {"label": "NEUTRAL", "compound": 0.0, "article_count": 0}
    tas = MagicMock()
    tas.compute_indicators.side_effect = lambda df: df
    tas.get_latest_signals.return_value = {"rsi": 50.0, "golden_cross": False}
    css = MagicMock(); css.compute.return_value = (55.0, {"score": 55.0})
    res = MagicMock(); res.get_recommendation_with_color.return_value = ("HOLD", "#FFA726")
    rag = MagicMock(); rag.retrieve.return_value = ""; rag._index = MagicMock(); rag._index.ntotal = 0

    monkeypatch.setattr("backend.agents.crew_orchestrator.MarketDataService",       lambda: mds)
    monkeypatch.setattr("backend.agents.crew_orchestrator.NewsService",             lambda: ns)
    monkeypatch.setattr("backend.agents.crew_orchestrator.SentimentService",        lambda: ss)
    monkeypatch.setattr("backend.agents.crew_orchestrator.TechnicalAnalysisService", lambda: tas)
    monkeypatch.setattr("backend.agents.crew_orchestrator.ConfidenceScoreService",  lambda: css)
    monkeypatch.setattr("backend.agents.crew_orchestrator.RecommendationService",   lambda: res)
    monkeypatch.setattr("backend.agents.crew_orchestrator.RAGService",              lambda: rag)
    return {"mds": mds, "ns": ns, "rag": rag}


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory parametrised tests
# ─────────────────────────────────────────────────────────────────────────────

_AGENT_CASES = [
    ("backend.agents.backtesting_agent",            create_backtesting_agent,            "Quantitative Backtesting Specialist"),
    ("backend.agents.document_intelligence_agent",  create_document_intelligence_agent,  "Financial Document Intelligence Analyst"),
    ("backend.agents.financial_analysis_agent",     create_financial_analysis_agent,     "Senior Financial Analyst"),
    ("backend.agents.investment_decision_agent",    create_investment_decision_agent,    "Investment Decision Strategist"),
    ("backend.agents.market_data_agent",            create_market_data_agent,            "Market Data Specialist"),
    ("backend.agents.news_intelligence_agent",      create_news_intelligence_agent,      "Financial News Intelligence Analyst"),
    ("backend.agents.opportunity_scanner_agent",    create_opportunity_scanner_agent,    "Market Opportunity Scanner"),
    ("backend.agents.portfolio_risk_agent",         create_portfolio_risk_agent,         "Portfolio Risk Manager"),
    ("backend.agents.report_writer_agent",          create_report_writer_agent,          "Financial Report Writer"),
    ("backend.agents.sentiment_analysis_agent",     create_sentiment_analysis_agent,     "Financial Sentiment Analyst"),
    ("backend.agents.strategy_optimization_agent",  create_strategy_optimization_agent,  "Strategy Optimization Engineer"),
    ("backend.agents.technical_analysis_agent",     create_technical_analysis_agent,     "Technical Analysis Expert"),
]


class TestAgentFactories:
    @pytest.mark.parametrize("module_path,factory,expected_role", _AGENT_CASES)
    def test_role(self, module_path, factory, expected_role):
        kwargs = _agent_kwargs(module_path, factory)
        assert kwargs["role"] == expected_role

    @pytest.mark.parametrize("module_path,factory,_role", _AGENT_CASES)
    def test_allow_delegation_false(self, module_path, factory, _role):
        kwargs = _agent_kwargs(module_path, factory)
        assert kwargs["allow_delegation"] is False

    @pytest.mark.parametrize("module_path,factory,_role", _AGENT_CASES)
    def test_verbose_true(self, module_path, factory, _role):
        kwargs = _agent_kwargs(module_path, factory)
        assert kwargs["verbose"] is True

    @pytest.mark.parametrize("module_path,factory,_role", _AGENT_CASES)
    def test_llm_uses_settings_model(self, module_path, factory, _role):
        kwargs = _agent_kwargs(module_path, factory)
        assert kwargs["llm"] == settings.openai_model

    @pytest.mark.parametrize("module_path,factory,_role", _AGENT_CASES)
    def test_goal_is_non_empty_string(self, module_path, factory, _role):
        kwargs = _agent_kwargs(module_path, factory)
        assert isinstance(kwargs.get("goal"), str) and len(kwargs["goal"]) > 20

    @pytest.mark.parametrize("module_path,factory,_role", _AGENT_CASES)
    def test_backstory_is_non_empty_string(self, module_path, factory, _role):
        kwargs = _agent_kwargs(module_path, factory)
        assert isinstance(kwargs.get("backstory"), str) and len(kwargs["backstory"]) > 20


# ─────────────────────────────────────────────────────────────────────────────
# _generate_narrative() task wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateNarrativeWiring:
    """Verify crew task graph construction without calling any LLM."""

    @pytest.fixture
    def crew_obj(self, patched_services):
        return StockAnalysisCrew()

    def _run_capturing(self, crew_obj, ticker="TSLA", kickoff_result="LLM narrative"):
        """
        Run crew.run() with Task/Crew/executor all patched.
        Returns (task_call_list, crew_call, run_result).
        """
        with patch("backend.agents.crew_orchestrator.Task") as mock_task, \
             patch("backend.agents.crew_orchestrator.Crew") as mock_crew, \
             patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.return_value = kickoff_result
            mock_exec.submit.return_value = future
            # Return distinct mocks per Task() call so context lists are identifiable
            task_mocks = [MagicMock(name=f"task_{i}") for i in range(6)]
            mock_task.side_effect = task_mocks
            result = crew_obj.run(ticker)
            return mock_task.call_args_list, mock_crew.call_args, result

    def test_crew_instantiated_with_six_agents(self, crew_obj):
        _, crew_call, _ = self._run_capturing(crew_obj)
        agents = crew_call.kwargs["agents"]
        assert len(agents) == 6

    def test_crew_instantiated_with_six_tasks(self, crew_obj):
        _, crew_call, _ = self._run_capturing(crew_obj)
        tasks = crew_call.kwargs["tasks"]
        assert len(tasks) == 6

    def test_crew_kickoff_submitted_to_executor(self, crew_obj):
        with patch("backend.agents.crew_orchestrator.Task"), \
             patch("backend.agents.crew_orchestrator.Crew") as mock_crew, \
             patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock()
            future.result.return_value = "ok"
            mock_exec.submit.return_value = future
            crew_obj.run("TSLA")
        mock_exec.submit.assert_called_once()
        # The first positional arg to submit is the callable (crew.kickoff)
        assert callable(mock_exec.submit.call_args.args[0])

    def test_task_doc_description_contains_ticker(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj, ticker="TSLA")
        assert "TSLA" in task_calls[0].kwargs["description"]

    def test_task_news_description_contains_ticker(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj, ticker="TSLA")
        assert "TSLA" in task_calls[1].kwargs["description"]

    def test_task_tech_description_contains_ticker(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj, ticker="TSLA")
        assert "TSLA" in task_calls[2].kwargs["description"]

    def test_task_analysis_description_contains_ticker(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj, ticker="TSLA")
        assert "TSLA" in task_calls[3].kwargs["description"]

    def test_task_decision_description_contains_ticker(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj, ticker="TSLA")
        assert "TSLA" in task_calls[4].kwargs["description"]

    def test_task_report_description_contains_ticker(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj, ticker="TSLA")
        assert "TSLA" in task_calls[5].kwargs["description"]

    def test_task_analysis_context_has_three_entries(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj)
        context = task_calls[3].kwargs["context"]
        assert len(context) == 3

    def test_task_analysis_context_contains_doc_news_tech(self, crew_obj):
        with patch("backend.agents.crew_orchestrator.Task") as mock_task, \
             patch("backend.agents.crew_orchestrator.Crew"), \
             patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = "ok"
            mock_exec.submit.return_value = future
            task_mocks = [MagicMock(name=f"t{i}") for i in range(6)]
            mock_task.side_effect = task_mocks
            crew_obj.run("TSLA")
        context = mock_task.call_args_list[3].kwargs["context"]
        assert task_mocks[0] in context  # task_doc
        assert task_mocks[1] in context  # task_news
        assert task_mocks[2] in context  # task_tech

    def test_task_decision_context_contains_analysis(self, crew_obj):
        with patch("backend.agents.crew_orchestrator.Task") as mock_task, \
             patch("backend.agents.crew_orchestrator.Crew"), \
             patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = "ok"
            mock_exec.submit.return_value = future
            task_mocks = [MagicMock(name=f"t{i}") for i in range(6)]
            mock_task.side_effect = task_mocks
            crew_obj.run("TSLA")
        context = mock_task.call_args_list[4].kwargs["context"]
        assert task_mocks[3] in context  # task_analysis

    def test_task_report_context_contains_analysis_and_decision(self, crew_obj):
        with patch("backend.agents.crew_orchestrator.Task") as mock_task, \
             patch("backend.agents.crew_orchestrator.Crew"), \
             patch("backend.agents.crew_orchestrator._CREW_EXECUTOR") as mock_exec:
            future = MagicMock(); future.result.return_value = "ok"
            mock_exec.submit.return_value = future
            task_mocks = [MagicMock(name=f"t{i}") for i in range(6)]
            mock_task.side_effect = task_mocks
            crew_obj.run("TSLA")
        context = mock_task.call_args_list[5].kwargs["context"]
        assert task_mocks[3] in context  # task_analysis
        assert task_mocks[4] in context  # task_decision

    def test_all_tasks_have_non_empty_expected_output(self, crew_obj):
        task_calls, _, _ = self._run_capturing(crew_obj)
        for i, task_call in enumerate(task_calls):
            expected = task_call.kwargs.get("expected_output", "")
            assert isinstance(expected, str) and len(expected) > 0, \
                f"task[{i}] has empty expected_output"

    def test_successful_kickoff_stored_as_narrative(self, crew_obj):
        _, _, result = self._run_capturing(crew_obj, kickoff_result="Full LLM narrative.")
        assert result["narrative"] == "Full LLM narrative."

    def test_kickoff_result_cast_to_string(self, crew_obj):
        """Non-string kickoff results must be converted to str."""
        _, _, result = self._run_capturing(crew_obj, kickoff_result=42)
        assert result["narrative"] == "42"


# ─────────────────────────────────────────────────────────────────────────────
# MarketScanCrew output format
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketScanOutputFormat:

    def _make_run_result(self, ticker: str, score: float = 70.0) -> dict:
        return {
            "ticker": ticker,
            "recommendation": "BUY",
            "confidence_score": score,
            "current_price": 150.0,
            "quote": {"sector": "Technology"},
            "narrative": "narrative text",
        }

    def test_each_result_has_required_fields(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = self._make_run_result("AAPL")
        results = scan_crew.scan(tickers=["AAPL"])
        required = {"ticker", "rank", "recommendation", "confidence_score",
                    "current_price", "sector", "rationale"}
        for key in required:
            assert key in results[0], f"Missing required field: {key}"

    def test_default_tickers_from_settings(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.side_effect = lambda t: self._make_run_result(t)
        results = scan_crew.scan(tickers=None)
        scanned = {r["ticker"] for r in results}
        assert scanned == set(settings.stock_universe_list)

    def test_sector_sourced_from_quote(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = {
            **self._make_run_result("AAPL"),
            "quote": {"sector": "Consumer Electronics"},
        }
        results = scan_crew.scan(tickers=["AAPL"])
        assert results[0]["sector"] == "Consumer Electronics"

    def test_confidence_score_preserved(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = self._make_run_result("AAPL", score=91.5)
        results = scan_crew.scan(tickers=["AAPL"])
        assert results[0]["confidence_score"] == 91.5

    def test_rationale_is_string(self, patched_services):
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = self._make_run_result("AAPL")
        results = scan_crew.scan(tickers=["AAPL"])
        assert isinstance(results[0]["rationale"], str)

    def test_missing_sector_in_quote_returns_none(self, patched_services):
        """scan() gracefully handles quote dicts without a sector key."""
        scan_crew = MarketScanCrew()
        scan_crew._analysis_crew = MagicMock()
        scan_crew._analysis_crew.run.return_value = {
            **self._make_run_result("AAPL"),
            "quote": {},  # no sector key
        }
        results = scan_crew.scan(tickers=["AAPL"])
        assert results[0]["sector"] is None
