"""
market_opportunity.py
---------------------
ORM model for a ranked market investment opportunity.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class MarketOpportunity(Base):
    """Ranked investment opportunity surfaced by the Opportunity Scanner."""

    __tablename__ = "market_opportunities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer)
    recommendation: Mapped[str] = mapped_column(String(32))
    confidence_score: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str | None] = mapped_column(Text)
    current_price: Mapped[float | None] = mapped_column(Float)
    sector: Mapped[str | None] = mapped_column(String(128))
    scan_date: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<MarketOpportunity rank={self.rank} ticker={self.ticker} score={self.confidence_score:.1f}>"
