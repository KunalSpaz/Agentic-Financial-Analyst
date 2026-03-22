"""
user_query.py
-------------
ORM model for logging user queries to the platform.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class UserQuery(Base):
    """Audit log of every query submitted by users."""

    __tablename__ = "user_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    endpoint: Mapped[str] = mapped_column(String(256))
    ticker: Mapped[str | None] = mapped_column(String(16))
    query_payload: Mapped[str | None] = mapped_column(Text)
    response_summary: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<UserQuery endpoint={self.endpoint} ticker={self.ticker}>"
