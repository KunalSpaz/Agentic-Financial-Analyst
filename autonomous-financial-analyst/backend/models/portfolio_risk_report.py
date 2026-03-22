"""
portfolio_risk_report.py
------------------------
ORM model for portfolio risk analysis reports.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class PortfolioRiskReport(Base):
    """Comprehensive risk metrics for a submitted portfolio."""

    __tablename__ = "portfolio_risk_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(String(64), index=True)
    holdings: Mapped[dict | None] = mapped_column(JSON)  # {ticker: weight}
    portfolio_volatility: Mapped[float | None] = mapped_column(Float)
    portfolio_beta: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float)
    correlation_matrix: Mapped[dict | None] = mapped_column(JSON)
    sector_exposure: Mapped[dict | None] = mapped_column(JSON)
    var_95: Mapped[float | None] = mapped_column(Float)   # Value at Risk 95%
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<PortfolioRiskReport id={self.portfolio_id} vol={self.portfolio_volatility:.4f}>"
