"""
test_recommendation_service.py
--------------------------------
Unit tests for RecommendationService.

This service is a pure mapping layer that is critical to the pipeline —
every analysis result flows through it.  Tests verify exact threshold
boundaries, correct hex colors, and edge cases at the score extremes.
"""
from __future__ import annotations

import pytest

from backend.services.recommendation_service import RecommendationService


# ─────────────────────────────────────────────────────────────────────────────
# get_recommendation — label mapping
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecommendation:
    @pytest.mark.parametrize("score,expected", [
        (100.0, "STRONG BUY"),
        (80.0,  "STRONG BUY"),
        (79.9,  "BUY"),
        (65.0,  "BUY"),
        (64.9,  "HOLD"),
        (50.0,  "HOLD"),
        (49.9,  "SELL"),
        (35.0,  "SELL"),
        (34.9,  "STRONG SELL"),
        (0.0,   "STRONG SELL"),
    ])
    def test_threshold_boundaries(self, score, expected):
        assert RecommendationService.get_recommendation(score) == expected

    def test_midrange_scores(self):
        assert RecommendationService.get_recommendation(72.5) == "BUY"
        assert RecommendationService.get_recommendation(57.0) == "HOLD"
        assert RecommendationService.get_recommendation(20.0) == "STRONG SELL"

    def test_exactly_at_each_threshold(self):
        """Boundary scores belong to the higher band (>= threshold)."""
        assert RecommendationService.get_recommendation(80) == "STRONG BUY"
        assert RecommendationService.get_recommendation(65) == "BUY"
        assert RecommendationService.get_recommendation(50) == "HOLD"
        assert RecommendationService.get_recommendation(35) == "SELL"

    def test_all_five_labels_reachable(self):
        labels = {RecommendationService.get_recommendation(s) for s in [90, 70, 55, 40, 10]}
        assert labels == {"STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"}


# ─────────────────────────────────────────────────────────────────────────────
# get_recommendation_with_color — label + hex color
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecommendationWithColor:
    def test_returns_tuple_of_two(self):
        result = RecommendationService.get_recommendation_with_color(75.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_label_matches_standalone_get_recommendation(self):
        for score in [90, 70, 55, 40, 10]:
            label_from_tuple, _ = RecommendationService.get_recommendation_with_color(score)
            label_standalone = RecommendationService.get_recommendation(score)
            assert label_from_tuple == label_standalone

    def test_colors_are_valid_hex(self):
        for score in [90, 70, 55, 40, 10]:
            _, color = RecommendationService.get_recommendation_with_color(score)
            assert color.startswith("#"), f"Expected hex color, got {color!r}"
            assert len(color) == 7, f"Expected 7-char hex, got {color!r}"

    @pytest.mark.parametrize("score,expected_label,green_or_red", [
        (90, "STRONG BUY",  "green"),
        (70, "BUY",         "green"),
        (55, "HOLD",        "amber"),
        (40, "SELL",        "orange"),
        (10, "STRONG SELL", "red"),
    ])
    def test_strong_buy_is_greenest_sell_is_reddest(self, score, expected_label, green_or_red):
        label, color = RecommendationService.get_recommendation_with_color(score)
        assert label == expected_label
        # Greens have high G component, reds have high R component (in hex)
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        if green_or_red == "green":
            assert g > r, f"Expected green-dominant color for {label}, got {color}"
        elif green_or_red == "red":
            assert r > g, f"Expected red-dominant color for {label}, got {color}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge-case scores (negative and > 100)
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCaseScores:
    def test_negative_score_returns_strong_sell(self):
        """Scores below 0 fall through all thresholds → STRONG SELL fallback."""
        assert RecommendationService.get_recommendation(-1.0) == "STRONG SELL"
        assert RecommendationService.get_recommendation(-99.9) == "STRONG SELL"

    def test_score_above_100_returns_strong_buy(self):
        """Scores > 100 still match the first threshold (>= 80) → STRONG BUY."""
        assert RecommendationService.get_recommendation(101.0) == "STRONG BUY"
        assert RecommendationService.get_recommendation(999.0) == "STRONG BUY"

    def test_exactly_zero_returns_strong_sell(self):
        assert RecommendationService.get_recommendation(0.0) == "STRONG SELL"

    def test_exactly_100_returns_strong_buy(self):
        assert RecommendationService.get_recommendation(100.0) == "STRONG BUY"

    def test_color_for_negative_score_is_hex(self):
        """Even with an out-of-range score, color should be a valid hex string."""
        _, color = RecommendationService.get_recommendation_with_color(-5.0)
        assert color.startswith("#")
        assert len(color) == 7

    def test_color_for_score_above_100_is_hex(self):
        _, color = RecommendationService.get_recommendation_with_color(110.0)
        assert color.startswith("#")
        assert len(color) == 7
