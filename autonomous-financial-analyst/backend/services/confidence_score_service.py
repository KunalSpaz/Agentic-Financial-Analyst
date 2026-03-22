"""
confidence_score_service.py
---------------------------
Generates a 0–100 confidence score from technical indicators, news
sentiment, and market momentum signals.

Signal weights:
    Technical indicators  = 50 %
    News sentiment        = 30 %
    Market momentum       = 20 %
"""

from __future__ import annotations

from typing import Any, Dict

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ConfidenceScoreService:
    """
    Computes a normalised confidence score for a buy/hold/sell recommendation.

    Usage::

        svc = ConfidenceScoreService()
        score, breakdown = svc.compute(technical_signals, sentiment)
    """

    # Weight allocations (must sum to 100)
    WEIGHT_TECHNICAL = 50
    WEIGHT_SENTIMENT = 30
    WEIGHT_MOMENTUM = 20

    def compute(
        self,
        technical_signals: Dict[str, Any],
        sentiment: Dict[str, Any],
    ) -> tuple[float, Dict[str, Any]]:
        """
        Compute the overall confidence score (0–100).

        Args:
            technical_signals: Output of :meth:`TechnicalAnalysisService.get_latest_signals`.
            sentiment:         Output of :meth:`SentimentService.analyse_articles`.

        Returns:
            Tuple of (score: float, breakdown: dict) where *breakdown* contains
            per-component scores and applied signals.
        """
        tech_score, tech_detail = self._score_technical(technical_signals)
        sent_score, sent_detail = self._score_sentiment(sentiment)
        mom_score, mom_detail = self._score_momentum(technical_signals)

        raw = (
            tech_score * (self.WEIGHT_TECHNICAL / 100)
            + sent_score * (self.WEIGHT_SENTIMENT / 100)
            + mom_score * (self.WEIGHT_MOMENTUM / 100)
        )

        # Normalise to [0, 100]
        score = max(0.0, min(100.0, raw))

        breakdown = {
            "score": round(score, 2),
            "technical": {"score": round(tech_score, 2), "detail": tech_detail},
            "sentiment": {"score": round(sent_score, 2), "detail": sent_detail},
            "momentum": {"score": round(mom_score, 2), "detail": mom_detail},
        }
        logger.debug("Confidence score computed: %.2f", score)
        return score, breakdown

    # ------------------------------------------------------------------
    # Private scoring sub-components
    # ------------------------------------------------------------------

    def _score_technical(self, signals: Dict[str, Any]) -> tuple[float, Dict]:
        """Score technical indicators on a 0–100 scale."""
        score = 50.0  # neutral baseline
        detail: Dict[str, Any] = {}

        if signals.get("rsi_oversold"):
            score += 20
            detail["rsi_oversold"] = "+20 (RSI < 35)"
        elif signals.get("rsi_overbought"):
            score -= 20
            detail["rsi_overbought"] = "-20 (RSI > 70)"

        if signals.get("macd_bullish_crossover"):
            score += 20
            detail["macd_bullish_crossover"] = "+20 (MACD bullish crossover)"
        elif signals.get("macd_hist") is not None and signals["macd_hist"] < 0:
            score -= 10
            detail["macd_bearish"] = "-10 (MACD below signal)"

        if signals.get("golden_cross"):
            score += 15
            detail["golden_cross"] = "+15 (SMA50 > SMA200)"
        elif signals.get("death_cross"):
            score -= 15
            detail["death_cross"] = "-15 (SMA50 < SMA200)"

        if signals.get("below_bb_lower"):
            score += 5
            detail["below_bb_lower"] = "+5 (price below lower BB)"
        elif signals.get("above_bb_upper"):
            score -= 5
            detail["above_bb_upper"] = "-5 (price above upper BB)"

        return max(0.0, min(100.0, score)), detail

    def _score_sentiment(self, sentiment: Dict[str, Any]) -> tuple[float, Dict]:
        """Score sentiment on 0–100 scale."""
        label = sentiment.get("label", "NEUTRAL")
        compound = sentiment.get("compound", 0.0)

        if label == "POSITIVE":
            score = 70.0 + compound * 30  # 70–100 range
            detail = {"label": "POSITIVE", "note": "+20 applied (positive sentiment)"}
        elif label == "NEGATIVE":
            score = 30.0 + compound * 30  # 0–30 range (compound is negative)
            detail = {"label": "NEGATIVE", "note": "-20 applied (negative sentiment)"}
        else:
            score = 50.0
            detail = {"label": "NEUTRAL", "note": "+5 applied (neutral sentiment)"}

        return max(0.0, min(100.0, score)), detail

    def _score_momentum(self, signals: Dict[str, Any]) -> tuple[float, Dict]:
        """Score market momentum on 0–100 scale."""
        score = 50.0
        detail: Dict[str, Any] = {}

        if signals.get("high_volume"):
            score += 10
            detail["high_volume"] = "+10 (volume > 1.5x average)"

        volume_ratio = signals.get("volume_ratio")
        if volume_ratio and volume_ratio > 2.0:
            score += 5
            detail["very_high_volume"] = "+5 (volume > 2x average)"

        # Price momentum relative to MAs
        close = signals.get("close")
        sma_50 = signals.get("sma_50")
        sma_200 = signals.get("sma_200")
        if close and sma_50 and close > sma_50:
            score += 8
            detail["above_sma50"] = "+8 (price above SMA50)"
        if close and sma_200 and close > sma_200:
            score += 7
            detail["above_sma200"] = "+7 (price above SMA200)"

        return max(0.0, min(100.0, score)), detail
