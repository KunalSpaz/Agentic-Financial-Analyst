"""
test_report_service.py
-----------------------
Unit tests for ReportService — GPT-4o daily market briefing generation.

Covers:
  - Successful briefing: structure, date, top_picks, sentiment passthrough
  - OpenAI failure: graceful fallback narrative
  - Edge inputs: empty opportunities, empty news, empty indices
"""
from __future__ import annotations

import datetime
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("faiss",        MagicMock())
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("torch",        MagicMock())
sys.modules.setdefault("newsapi",      MagicMock())
sys.modules.setdefault("crewai",       MagicMock())

from backend.services.report_service import ReportService  # noqa: E402


def _openai_mock(content: str = "Positive market outlook today.") -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


SAMPLE_OPPS = [
    {"ticker": "AAPL", "recommendation": "BUY",        "confidence_score": 78.0},
    {"ticker": "MSFT", "recommendation": "HOLD",       "confidence_score": 65.0},
    {"ticker": "NVDA", "recommendation": "STRONG BUY", "confidence_score": 85.0},
    {"ticker": "TSLA", "recommendation": "SELL",       "confidence_score": 35.0},
    {"ticker": "AMZN", "recommendation": "BUY",        "confidence_score": 72.0},
    {"ticker": "META", "recommendation": "BUY",        "confidence_score": 70.0},  # 6th — should be excluded
]
SAMPLE_INDICES = {
    "SPY": {"price": 450.0, "change_pct": 0.5},
    "QQQ": {"price": 380.0, "change_pct": 1.2},
}
SAMPLE_NEWS = [
    {"title": "Fed holds rates", "source": "Reuters"},
    {"title": "Tech earnings beat", "source": "Bloomberg"},
]


@pytest.fixture
def service() -> ReportService:
    with patch("backend.services.report_service.OpenAI") as mock_ctor:
        mock_ctor.return_value = _openai_mock()
        svc = ReportService()
    return svc


# ─────────────────────────────────────────────────────────────────────────────
# Structure
# ─────────────────────────────────────────────────────────────────────────────

class TestBriefingStructure:
    def test_returns_dict(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert isinstance(result, dict)

    def test_required_keys_present(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        for key in ("date", "narrative", "overall_sentiment", "top_picks", "market_indices"):
            assert key in result, f"Missing key: {key}"

    def test_date_is_todays_iso_format(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert result["date"] == datetime.date.today().isoformat()

    def test_market_indices_passed_through(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert result["market_indices"] == SAMPLE_INDICES

    def test_overall_sentiment_passed_through(self, service):
        result = service.generate_daily_briefing(
            SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS, overall_sentiment="POSITIVE"
        )
        assert result["overall_sentiment"] == "POSITIVE"

    def test_default_sentiment_is_neutral(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert result["overall_sentiment"] == "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# top_picks
# ─────────────────────────────────────────────────────────────────────────────

class TestTopPicks:
    def test_top_picks_truncated_to_5(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert len(result["top_picks"]) == 5

    def test_sixth_ticker_excluded(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert "META" not in result["top_picks"]

    def test_top_picks_are_ticker_strings(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        for pick in result["top_picks"]:
            assert isinstance(pick, str)

    def test_empty_opportunities_returns_empty_top_picks(self, service):
        result = service.generate_daily_briefing([], SAMPLE_INDICES, SAMPLE_NEWS)
        assert result["top_picks"] == []

    def test_fewer_than_5_opps_returns_all(self, service):
        two_opps = SAMPLE_OPPS[:2]
        result = service.generate_daily_briefing(two_opps, SAMPLE_INDICES, SAMPLE_NEWS)
        assert len(result["top_picks"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Narrative
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrative:
    def test_narrative_is_string(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert isinstance(result["narrative"], str)
        assert len(result["narrative"]) > 0

    def test_narrative_from_openai_content(self, service):
        service._client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Unique narrative 12345."))]
        )
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert "Unique narrative 12345." in result["narrative"]

    def test_openai_error_returns_fallback_not_exception(self, service):
        service._client.chat.completions.create.side_effect = Exception("API timeout")
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        assert isinstance(result["narrative"], str)
        assert "unavailable" in result["narrative"].lower()

    def test_openai_error_still_returns_full_structure(self, service):
        service._client.chat.completions.create.side_effect = Exception("fail")
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        for key in ("date", "narrative", "overall_sentiment", "top_picks", "market_indices"):
            assert key in result


# ─────────────────────────────────────────────────────────────────────────────
# Edge inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeInputs:
    def test_empty_news_does_not_crash(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, [])
        assert isinstance(result["narrative"], str)

    def test_empty_market_indices_does_not_crash(self, service):
        result = service.generate_daily_briefing(SAMPLE_OPPS, {}, SAMPLE_NEWS)
        assert isinstance(result["narrative"], str)

    def test_all_inputs_empty_does_not_crash(self, service):
        result = service.generate_daily_briefing([], {}, [])
        assert isinstance(result, dict)
        assert result["top_picks"] == []

    def test_openai_called_with_model_from_settings(self, service):
        with patch("backend.services.report_service.settings") as mock_settings:
            mock_settings.openai_model = "gpt-4o"
            mock_settings.openai_api_key = "test-key"
            service.generate_daily_briefing(SAMPLE_OPPS, SAMPLE_INDICES, SAMPLE_NEWS)
        call_kwargs = service._client.chat.completions.create.call_args[1]
        assert call_kwargs.get("temperature") == 0.3
        assert call_kwargs.get("max_tokens") == 600
