"""
news_article.py
---------------
ORM model for a fetched financial news article.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class NewsArticle(Base):
    """A single news article retrieved from NewsAPI."""

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True)
    source: Mapped[str | None] = mapped_column(String(128))
    author: Mapped[str | None] = mapped_column(String(256))
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1024))
    published_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<NewsArticle title={self.title[:40]!r}>"
