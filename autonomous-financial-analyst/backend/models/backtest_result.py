"""
backtest_result.py
------------------
ORM model for strategy backtesting results.
"""
from __future__ import annotations
import datetime
from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.connection import Base


class BacktestResult(Base):
    """Historical backtest result for a given ticker + strategy."""

    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    strategy_name: Mapped[str] = mapped_column(String(128))
    start_date: Mapped[str] = mapped_column(String(32))
    end_date: Mapped[str] = mapped_column(String(32))
    total_return: Mapped[float | None] = mapped_column(Float)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)
    win_rate: Mapped[float | None] = mapped_column(Float)
    num_trades: Mapped[int | None] = mapped_column(Integer)
    parameters: Mapped[dict | None] = mapped_column(JSON)
    equity_curve: Mapped[list | None] = mapped_column(JSON)
    trade_log: Mapped[list | None] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<BacktestResult ticker={self.ticker} return={self.total_return:.2%}>"
