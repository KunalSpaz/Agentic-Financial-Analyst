"""
financial_document.py
---------------------
ORM model for financial documents ingested into the RAG pipeline.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class FinancialDocument(Base):
    """A financial document (earnings transcript, report, commentary) stored for RAG."""

    __tablename__ = "financial_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True)
    doc_type: Mapped[str] = mapped_column(String(64))  # earnings_transcript | report | analyst_note
    title: Mapped[str] = mapped_column(String(512))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(256))
    document_date: Mapped[str | None] = mapped_column(String(32))
    faiss_indexed: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<FinancialDocument ticker={self.ticker} type={self.doc_type} title={self.title[:40]!r}>"
