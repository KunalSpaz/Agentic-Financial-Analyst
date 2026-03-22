"""
strategy_optimization.py
------------------------
ORM model for strategy optimization run results.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class StrategyOptimization(Base):
    """Result of an automated strategy parameter optimization run."""

    __tablename__ = "strategy_optimizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    objective: Mapped[str] = mapped_column(String(64))  # maximize_return | sharpe | minimize_drawdown
    best_parameters: Mapped[dict | None] = mapped_column(JSON)
    best_return: Mapped[float | None] = mapped_column(Float)
    best_sharpe: Mapped[float | None] = mapped_column(Float)
    best_drawdown: Mapped[float | None] = mapped_column(Float)
    iterations: Mapped[int | None] = mapped_column(Integer)
    all_results: Mapped[list | None] = mapped_column(JSON)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<StrategyOptimization ticker={self.ticker} objective={self.objective}>"
