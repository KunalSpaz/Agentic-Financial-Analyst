"""
test_confidence_score_service.py
---------------------------------
Unit tests for the ConfidenceScoreService.
"""
from __future__ import annotations

import pytest

from backend.services.confidence_score_service import ConfidenceScoreService


@pytest.fixture
def service() -> ConfidenceScoreService:
    return ConfidenceScoreService()


@pytest.fixture
def bullish_signals() -> dict:
    return {
        "rsi": 28.0, "rsi_oversold": True, "rsi_overbought": False,
        "macd": 0.5, "macd_signal": 0.3, "macd_hist": 0.2,
        "macd_bullish_crossover": True,
        "golden_cross": True, "death_cross": False,
        "above_bb_upper": False, "below_bb_lower": True,
        "high_volume": True, "volume_ratio": 2.1,
        "close": 175.0, "sma_50": 165.0, "sma_200": 150.0,
    }


@pytest.fixture
def bearish_signals() -> dict:
    return {
        "rsi": 78.0, "rsi_oversold": False, "rsi_overbought": True,
        "macd": -0.5, "macd_signal": 0.0, "macd_hist": -0.5,
        "macd_bullish_crossover": False,
        "golden_cross": False, "death_cross": True,
        "above_bb_upper": True, "below_bb_lower": False,
        "high_volume": False, "volume_ratio": 0.8,
        "close": 100.0, "sma_50": 110.0, "sma_200": 120.0,
    }


@pytest.fixture
def positive_sentiment() -> dict:
    return {"label": "POSITIVE", "compound": 0.6, "positive": 0.7, "neutral": 0.2, "negative": 0.1, "article_count": 10}


@pytest.fixture
def negative_sentiment() -> dict:
    return {"label": "NEGATIVE", "compound": -0.5, "positive": 0.1, "neutral": 0.15, "negative": 0.75, "article_count": 8}


@pytest.fixture
def neutral_sentiment() -> dict:
    return {"label": "NEUTRAL", "compound": 0.0, "positive": 0.3, "neutral": 0.5, "negative": 0.2, "article_count": 5}


class TestComputeScore:
    def test_bullish_setup_gives_high_score(self, service, bullish_signals, positive_sentiment):
        score, _ = service.compute(bullish_signals, positive_sentiment)
        assert score >= 70, f"Expected high score for bullish setup, got {score}"

    def test_bearish_setup_gives_low_score(self, service, bearish_signals, negative_sentiment):
        score, _ = service.compute(bearish_signals, negative_sentiment)
        assert score <= 40, f"Expected low score for bearish setup, got {score}"

    def test_score_always_in_range(self, service, bullish_signals, bearish_signals, positive_sentiment, negative_sentiment):
        for signals in [bullish_signals, bearish_signals]:
            for sentiment in [positive_sentiment, negative_sentiment]:
                score, _ = service.compute(signals, sentiment)
                assert 0 <= score <= 100

    def test_breakdown_contains_all_components(self, service, bullish_signals, neutral_sentiment):
        _, breakdown = service.compute(bullish_signals, neutral_sentiment)
        assert "technical" in breakdown
        assert "sentiment" in breakdown
        assert "momentum" in breakdown
        assert "score" in breakdown

    def test_empty_signals_returns_midrange_score(self, service, neutral_sentiment):
        score, _ = service.compute({}, neutral_sentiment)
        assert 0 <= score <= 100

    def test_missing_sentiment_keys_do_not_crash(self, service, bullish_signals):
        """Incomplete sentiment dict should not raise — compound defaults to 0."""
        score, _ = service.compute(bullish_signals, {})
        assert 0 <= score <= 100

    def test_breakdown_score_matches_returned_score(self, service, bullish_signals, positive_sentiment):
        score, breakdown = service.compute(bullish_signals, positive_sentiment)
        assert abs(breakdown["score"] - score) < 0.001

    def test_bullish_score_higher_than_bearish(self, service, bullish_signals, bearish_signals,
                                               positive_sentiment, negative_sentiment):
        bull_score, _ = service.compute(bullish_signals, positive_sentiment)
        bear_score, _ = service.compute(bearish_signals, negative_sentiment)
        assert bull_score > bear_score

    def test_weight_components_sum_to_100(self, service):
        total = (service.WEIGHT_TECHNICAL + service.WEIGHT_SENTIMENT + service.WEIGHT_MOMENTUM)
        assert total == 100


# ─────────────────────────────────────────────────────────────────────────────
# _score_technical
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreTechnical:
    def test_neutral_baseline_with_no_signals(self, service):
        score, _ = service._score_technical({})
        assert score == 50.0

    def test_rsi_oversold_adds_20(self, service):
        score, detail = service._score_technical({"rsi_oversold": True})
        assert score == 70.0
        assert "rsi_oversold" in detail

    def test_rsi_overbought_subtracts_20(self, service):
        score, detail = service._score_technical({"rsi_overbought": True})
        assert score == 30.0
        assert "rsi_overbought" in detail

    def test_macd_bullish_crossover_adds_20(self, service):
        score, detail = service._score_technical({"macd_bullish_crossover": True})
        assert score == 70.0
        assert "macd_bullish_crossover" in detail

    def test_negative_macd_hist_subtracts_10(self, service):
        score, detail = service._score_technical({"macd_hist": -0.5})
        assert score == 40.0
        assert "macd_bearish" in detail

    def test_golden_cross_adds_15(self, service):
        score, detail = service._score_technical({"golden_cross": True})
        assert score == 65.0
        assert "golden_cross" in detail

    def test_death_cross_subtracts_15(self, service):
        score, detail = service._score_technical({"death_cross": True})
        assert score == 35.0
        assert "death_cross" in detail

    def test_below_bb_lower_adds_5(self, service):
        score, _ = service._score_technical({"below_bb_lower": True})
        assert score == 55.0

    def test_above_bb_upper_subtracts_5(self, service):
        score, _ = service._score_technical({"above_bb_upper": True})
        assert score == 45.0

    def test_score_clamped_at_100_on_all_bullish(self, service):
        all_bullish = {
            "rsi_oversold": True, "macd_bullish_crossover": True,
            "golden_cross": True, "below_bb_lower": True,
        }
        score, _ = service._score_technical(all_bullish)
        assert score <= 100.0

    def test_score_clamped_at_0_on_all_bearish(self, service):
        all_bearish = {
            "rsi_overbought": True, "macd_hist": -1.0,
            "death_cross": True, "above_bb_upper": True,
        }
        score, _ = service._score_technical(all_bearish)
        assert score >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# _score_sentiment
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreSentiment:
    def test_neutral_label_returns_50(self, service):
        score, detail = service._score_sentiment({"label": "NEUTRAL", "compound": 0.0})
        assert score == 50.0
        assert detail["label"] == "NEUTRAL"

    def test_positive_label_scores_above_70(self, service):
        score, _ = service._score_sentiment({"label": "POSITIVE", "compound": 0.5})
        assert score > 70.0

    def test_negative_label_scores_below_30(self, service):
        score, _ = service._score_sentiment({"label": "NEGATIVE", "compound": -0.5})
        assert score < 30.0

    def test_empty_dict_defaults_to_neutral_50(self, service):
        score, _ = service._score_sentiment({})
        assert score == 50.0

    def test_positive_score_clamped_to_100(self, service):
        score, _ = service._score_sentiment({"label": "POSITIVE", "compound": 10.0})
        assert score <= 100.0

    def test_negative_score_clamped_to_0(self, service):
        score, _ = service._score_sentiment({"label": "NEGATIVE", "compound": -10.0})
        assert score >= 0.0

    def test_returns_tuple_of_score_and_detail(self, service):
        result = service._score_sentiment({"label": "POSITIVE", "compound": 0.4})
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], dict)


# ─────────────────────────────────────────────────────────────────────────────
# _score_momentum
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreMomentum:
    def test_neutral_baseline_with_no_signals(self, service):
        score, _ = service._score_momentum({})
        assert score == 50.0

    def test_high_volume_adds_10(self, service):
        score, detail = service._score_momentum({"high_volume": True})
        assert score == 60.0
        assert "high_volume" in detail

    def test_very_high_volume_adds_extra_5(self, service):
        score, detail = service._score_momentum({"high_volume": True, "volume_ratio": 2.5})
        assert score == 65.0
        assert "very_high_volume" in detail

    def test_volume_ratio_2_exact_does_not_add_extra(self, service):
        """volume_ratio must be strictly > 2.0 for bonus."""
        score, _ = service._score_momentum({"high_volume": True, "volume_ratio": 2.0})
        assert score == 60.0  # only +10, not +5

    def test_above_sma50_adds_8(self, service):
        score, detail = service._score_momentum({"close": 180.0, "sma_50": 170.0})
        assert score == 58.0
        assert "above_sma50" in detail

    def test_above_sma200_adds_7(self, service):
        score, detail = service._score_momentum({"close": 180.0, "sma_200": 170.0})
        assert score == 57.0
        assert "above_sma200" in detail

    def test_below_sma50_no_bonus(self, service):
        score, detail = service._score_momentum({"close": 150.0, "sma_50": 170.0})
        assert score == 50.0
        assert "above_sma50" not in detail

    def test_score_clamped_to_100(self, service):
        all_positive = {
            "high_volume": True, "volume_ratio": 3.0,
            "close": 200.0, "sma_50": 150.0, "sma_200": 130.0,
        }
        score, _ = service._score_momentum(all_positive)
        assert score <= 100.0

    def test_score_clamped_to_0(self, service):
        """Even with extreme negative values, score must not go below 0."""
        score, _ = service._score_momentum({"close": 50.0, "sma_50": 200.0, "sma_200": 300.0})
        assert score >= 0.0
