"""
sentiment_service.py
--------------------
FinBERT-based sentiment analysis pipeline for financial news.
Classifies text as POSITIVE, NEUTRAL, or NEGATIVE and aggregates
scores per ticker.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentService:
    """
    Singleton-style FinBERT sentiment analyser.

    Lazy-loads the model on first use to avoid startup cost.
    """

    _instance: Optional["SentimentService"] = None
    _pipeline = None

    def __new__(cls) -> "SentimentService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_pipeline(self):
        """Lazily load the FinBERT pipeline."""
        if self._pipeline is None:
            logger.info("Loading FinBERT model: %s …", settings.finbert_model)
            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=settings.finbert_model,
                tokenizer=settings.finbert_model,
                device=device,
                top_k=None,          # return all label scores
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded (device=%d).", device)
        return self._pipeline

    def analyse_text(self, text: str) -> Dict[str, float]:
        """
        Run FinBERT on a single text snippet.

        Args:
            text: Financial news headline or body text.

        Returns:
            Dict with keys ``positive``, ``neutral``, ``negative`` (float 0–1).
        """
        if not text or not text.strip():
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

        try:
            results = self._get_pipeline()(text[:512])
            scores: Dict[str, float] = {}
            for item in results[0]:
                scores[item["label"].lower()] = item["score"]
            return scores
        except Exception as exc:
            logger.error("FinBERT inference error: %s", exc)
            return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}

    def analyse_articles(
        self, articles: List[Dict], ticker: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Aggregate FinBERT sentiment over a list of news articles.

        Args:
            articles: List of article dicts with at least ``title`` and optionally ``description``.
            ticker:   Ticker symbol for logging context.

        Returns:
            Dict with keys: positive, neutral, negative, compound, label, article_count.
        """
        if not articles:
            return {
                "positive": 0.0, "neutral": 1.0, "negative": 0.0,
                "compound": 0.0, "label": "NEUTRAL", "article_count": 0,
            }

        pos_total = neu_total = neg_total = 0.0
        count = 0

        for art in articles:
            text = " ".join(
                filter(None, [art.get("title", ""), art.get("description", "")])
            )
            scores = self.analyse_text(text)
            pos_total += scores.get("positive", 0.0)
            neu_total += scores.get("neutral", 0.0)
            neg_total += scores.get("negative", 0.0)
            count += 1

        if count == 0:
            return {
                "positive": 0.0, "neutral": 1.0, "negative": 0.0,
                "compound": 0.0, "label": "NEUTRAL", "article_count": 0,
            }

        pos_avg = pos_total / count
        neu_avg = neu_total / count
        neg_avg = neg_total / count
        compound = pos_avg - neg_avg  # range [-1, 1]

        if compound > 0.1:
            label = "POSITIVE"
        elif compound < -0.1:
            label = "NEGATIVE"
        else:
            label = "NEUTRAL"

        logger.info(
            "Sentiment for %s: %s (compound=%.3f, n=%d)",
            ticker or "unknown", label, compound, count
        )
        return {
            "positive": round(pos_avg, 4),
            "neutral": round(neu_avg, 4),
            "negative": round(neg_avg, 4),
            "compound": round(compound, 4),
            "label": label,
            "article_count": count,
        }
