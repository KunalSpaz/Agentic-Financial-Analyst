"""
sentiment_score.py
------------------
ORM model for FinBERT sentiment scores associated with a ticker.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class SentimentScore(Base):
    """Aggregated FinBERT sentiment for a ticker over a date window."""

    __tablename__ = "sentiment_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    positive_score: Mapped[float] = mapped_column(Float, default=0.0)
    neutral_score: Mapped[float] = mapped_column(Float, default=0.0)
    negative_score: Mapped[float] = mapped_column(Float, default=0.0)
    # Compound: positive - negative, range [-1, 1]
    compound_score: Mapped[float] = mapped_column(Float, default=0.0)
    label: Mapped[str] = mapped_column(String(16), default="NEUTRAL")
    article_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<SentimentScore ticker={self.ticker} label={self.label}>"
