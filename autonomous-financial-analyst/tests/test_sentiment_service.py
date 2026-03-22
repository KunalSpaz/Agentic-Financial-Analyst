"""
test_sentiment_service.py
--------------------------
Unit tests for SentimentService (FinBERT).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.services.sentiment_service import SentimentService


@pytest.fixture
def service() -> SentimentService:
    # Reset singleton for each test
    SentimentService._instance = None
    SentimentService._pipeline = None
    return SentimentService()


@pytest.fixture
def mock_pipeline_output():
    """Simulate FinBERT pipeline output for a positive article."""
    return [[
        {"label": "positive", "score": 0.85},
        {"label": "neutral",  "score": 0.10},
        {"label": "negative", "score": 0.05},
    ]]


class TestAnalyseText:
    def test_returns_dict_with_three_keys(self, service, mock_pipeline_output):
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=mock_pipeline_output)
            result = service.analyse_text("Apple beats earnings expectations.")
        assert "positive" in result
        assert "neutral" in result
        assert "negative" in result

    def test_empty_text_returns_neutral(self, service):
        result = service.analyse_text("")
        assert result["neutral"] == 1.0
        assert result["positive"] == 0.0
        assert result["negative"] == 0.0

    def test_scores_sum_to_approx_one(self, service, mock_pipeline_output):
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=mock_pipeline_output)
            result = service.analyse_text("Strong quarterly results expected.")
        total = result["positive"] + result["neutral"] + result["negative"]
        assert abs(total - 1.0) < 0.01


class TestAnalyseArticles:
    def test_empty_articles_returns_neutral_label(self, service):
        result = service.analyse_articles([])
        assert result["label"] == "NEUTRAL"
        assert result["article_count"] == 0

    def test_positive_articles_return_positive_label(self, service):
        positive_output = [[
            {"label": "positive", "score": 0.90},
            {"label": "neutral",  "score": 0.07},
            {"label": "negative", "score": 0.03},
        ]]
        articles = [
            {"title": "Company posts record profits", "description": "Revenue surges 50%"},
            {"title": "Stock hits all-time high", "description": "Investors thrilled"},
        ]
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=positive_output)
            result = service.analyse_articles(articles, ticker="TEST")
        assert result["label"] == "POSITIVE"
        assert result["compound"] > 0
        assert result["article_count"] == 2

    def test_negative_articles_return_negative_label(self, service):
        negative_output = [[
            {"label": "positive", "score": 0.05},
            {"label": "neutral",  "score": 0.10},
            {"label": "negative", "score": 0.85},
        ]]
        articles = [{"title": "Layoffs announced", "description": "Company cutting 10% of workforce"}]
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=negative_output)
            result = service.analyse_articles(articles, ticker="TEST")
        assert result["label"] == "NEGATIVE"
        assert result["compound"] < 0

    def test_neutral_compound_returns_neutral_label(self, service):
        """compound in (-0.1, 0.1] → NEUTRAL."""
        near_neutral_output = [[
            {"label": "positive", "score": 0.35},
            {"label": "neutral",  "score": 0.35},
            {"label": "negative", "score": 0.30},
        ]]
        articles = [{"title": "Market flat", "description": "Unchanged"}]
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=near_neutral_output)
            result = service.analyse_articles(articles)
        assert result["label"] == "NEUTRAL"

    def test_article_count_matches_input(self, service):
        """article_count must equal the number of articles passed in."""
        output = [[
            {"label": "positive", "score": 0.7},
            {"label": "neutral",  "score": 0.2},
            {"label": "negative", "score": 0.1},
        ]]
        articles = [
            {"title": "A", "description": ""},
            {"title": "B", "description": ""},
            {"title": "C", "description": ""},
        ]
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=output)
            result = service.analyse_articles(articles)
        assert result["article_count"] == 3

    def test_article_without_description_key_does_not_crash(self, service):
        """Articles that have only a title (no description key) should be handled."""
        output = [[
            {"label": "positive", "score": 0.8},
            {"label": "neutral",  "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]]
        articles = [{"title": "Fed cuts rates"}]  # no 'description' key
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=output)
            result = service.analyse_articles(articles)
        assert isinstance(result, dict)

    def test_article_without_title_key_does_not_crash(self, service):
        """Articles with only description and no title should not crash."""
        output = [[
            {"label": "neutral", "score": 1.0},
        ]]
        articles = [{"description": "Some text without title"}]  # no 'title' key
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=output)
            result = service.analyse_articles(articles)
        assert isinstance(result, dict)

    def test_all_three_score_keys_present_in_result(self, service):
        output = [[
            {"label": "positive", "score": 0.5},
            {"label": "neutral",  "score": 0.3},
            {"label": "negative", "score": 0.2},
        ]]
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(return_value=output)
            result = service.analyse_articles([{"title": "Test"}])
        for key in ("positive", "neutral", "negative", "compound", "label", "article_count"):
            assert key in result


class TestAnalyseText:
    def test_text_truncated_to_512_chars(self, service):
        """Text longer than 512 chars should be truncated before pipeline call."""
        long_text = "x" * 1000
        output = [[
            {"label": "positive", "score": 0.9},
            {"label": "neutral",  "score": 0.1},
            {"label": "negative", "score": 0.0},
        ]]
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe_instance = MagicMock(return_value=output)
            mock_pipe.return_value = mock_pipe_instance
            service.analyse_text(long_text)
            # Verify pipeline was called with at most 512 chars
            call_arg = mock_pipe_instance.call_args[0][0]
            assert len(call_arg) <= 512

    def test_pipeline_exception_returns_neutral(self, service):
        """If FinBERT raises, analyse_text should return neutral defaults."""
        with patch.object(service, "_get_pipeline") as mock_pipe:
            mock_pipe.return_value = MagicMock(side_effect=RuntimeError("CUDA OOM"))
            result = service.analyse_text("Some text")
        assert result["neutral"] == 1.0
        assert result["positive"] == 0.0
        assert result["negative"] == 0.0

    def test_whitespace_only_returns_neutral(self, service):
        result = service.analyse_text("   \t\n  ")
        assert result["neutral"] == 1.0
