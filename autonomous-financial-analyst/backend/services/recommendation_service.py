"""
recommendation_service.py
--------------------------
Maps a 0–100 confidence score to a human-readable recommendation.

Score ranges:
    80–100  →  Strong Buy
    65–79   →  Buy
    50–64   →  Hold
    35–49   →  Sell
    0–34    →  Strong Sell
"""

from __future__ import annotations

from typing import Tuple


class RecommendationService:
    """
    Translates confidence scores into categorical buy/sell recommendations.
    """

    THRESHOLDS = [
        (80, "STRONG BUY"),
        (65, "BUY"),
        (50, "HOLD"),
        (35, "SELL"),
        (0,  "STRONG SELL"),
    ]

    @classmethod
    def get_recommendation(cls, score: float) -> str:
        """
        Return recommendation label for a given confidence score.

        Args:
            score: Float between 0 and 100.

        Returns:
            One of: ``STRONG BUY``, ``BUY``, ``HOLD``, ``SELL``, ``STRONG SELL``.
        """
        for threshold, label in cls.THRESHOLDS:
            if score >= threshold:
                return label
        return "STRONG SELL"

    @classmethod
    def get_recommendation_with_color(cls, score: float) -> Tuple[str, str]:
        """
        Return (recommendation, hex_color) for dashboard rendering.

        Args:
            score: Float between 0 and 100.

        Returns:
            Tuple of (label: str, hex_color: str).
        """
        label = cls.get_recommendation(score)
        colors = {
            "STRONG BUY":  "#00C853",  # vivid green
            "BUY":         "#69F0AE",  # light green
            "HOLD":        "#FFD740",  # amber
            "SELL":        "#FF6D00",  # orange
            "STRONG SELL": "#D50000",  # red
        }
        return label, colors.get(label, "#FFFFFF")
