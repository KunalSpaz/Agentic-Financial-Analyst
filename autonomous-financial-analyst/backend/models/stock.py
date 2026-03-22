"""
stock.py
--------
ORM model representing a tracked stock / equity instrument.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class Stock(Base):
    """Persisted record for a stock ticker with latest cached price data."""

    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, unique=True, index=True)
    company_name: Mapped[str | None] = mapped_column(String(256))
    sector: Mapped[str | None] = mapped_column(String(128))
    industry: Mapped[str | None] = mapped_column(String(128))
    current_price: Mapped[float | None] = mapped_column(Float)
    market_cap: Mapped[float | None] = mapped_column(Float)
    pe_ratio: Mapped[float | None] = mapped_column(Float)
    fifty_two_week_high: Mapped[float | None] = mapped_column(Float)
    fifty_two_week_low: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Stock ticker={self.ticker} price={self.current_price}>"
