"""
analysis_report.py
------------------
ORM model for a complete AI analysis report for a stock.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class AnalysisReport(Base):
    """Full AI-generated analysis report for a given ticker."""

    __tablename__ = "analysis_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    recommendation: Mapped[str] = mapped_column(String(32))  # STRONG BUY … STRONG SELL
    confidence_score: Mapped[float] = mapped_column(Float)
    technical_summary: Mapped[str | None] = mapped_column(Text)
    sentiment_summary: Mapped[str | None] = mapped_column(Text)
    fundamental_summary: Mapped[str | None] = mapped_column(Text)
    narrative: Mapped[str | None] = mapped_column(Text)
    rsi: Mapped[float | None] = mapped_column(Float)
    macd: Mapped[float | None] = mapped_column(Float)
    ma_50: Mapped[float | None] = mapped_column(Float)
    ma_200: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<AnalysisReport ticker={self.ticker} rec={self.recommendation} score={self.confidence_score:.1f}>"
