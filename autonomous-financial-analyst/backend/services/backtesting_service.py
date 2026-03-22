"""
backtesting_service.py
----------------------
Event-driven strategy backtesting engine.

Default strategy:
    BUY  when RSI < 35 AND MACD bullish crossover AND sentiment >= NEUTRAL
    SELL when RSI > 70 OR sentiment == NEGATIVE
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.services.market_data_service import MarketDataService
from backend.services.technical_analysis_service import TechnicalAnalysisService
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestParameters:
    """Configurable parameters for the default RSI+MACD strategy."""
    rsi_buy_threshold: float = 35.0
    rsi_sell_threshold: float = 70.0
    macd_confirmation: bool = True
    ma_filter: bool = True          # price must be above SMA200 to buy
    initial_capital: float = 10_000.0
    position_size_pct: float = 1.0  # fraction of capital per trade (1.0 = 100%)


@dataclass
class BacktestResult:
    """Structured output from a backtest run."""
    ticker: str
    strategy_name: str
    start_date: str
    end_date: str
    parameters: Dict[str, Any]
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    equity_curve: List[float]
    trade_log: List[Dict[str, Any]]


class BacktestingService:
    """
    Vectorised backtesting engine for the default RSI/MACD/Sentiment strategy.
    """

    def __init__(self) -> None:
        self._mds = MarketDataService()
        self._tas = TechnicalAnalysisService()

    def run_backtest(
        self,
        ticker: str,
        params: Optional[BacktestParameters] = None,
        period: str = "2y",
        sentiment_label: str = "NEUTRAL",
    ) -> BacktestResult:
        """
        Execute backtest for *ticker* using the given parameters.

        Args:
            ticker:           Stock symbol.
            params:           Strategy parameters. Defaults used if ``None``.
            period:           Historical data window for yfinance.
            sentiment_label:  Current sentiment label (POSITIVE/NEUTRAL/NEGATIVE)
                              used as a static bias during the backtest.

        Returns:
            :class:`BacktestResult` with all metrics populated.
        """
        if params is None:
            params = BacktestParameters()

        df = self._mds.get_historical_data(ticker, period=period)
        if df.empty:
            raise ValueError(f"No data available for {ticker}")

        df = self._tas.compute_indicators(df)
        df = df.dropna(subset=["RSI_14"]).copy()

        if df.empty:
            raise ValueError(f"Insufficient data after indicator calculation for {ticker}")

        equity_curve, trade_log = self._simulate(df, params, sentiment_label)

        total_return = (equity_curve[-1] - params.initial_capital) / params.initial_capital
        sharpe = self._sharpe_ratio(equity_curve)
        max_dd = self._max_drawdown(equity_curve)
        wins = sum(1 for t in trade_log if t.get("pnl", 0) > 0)
        win_rate = wins / len(trade_log) if trade_log else 0.0

        return BacktestResult(
            ticker=ticker,
            strategy_name="RSI_MACD_Sentiment",
            start_date=str(df.index[0].date()),
            end_date=str(df.index[-1].date()),
            parameters={
                "rsi_buy_threshold": params.rsi_buy_threshold,
                "rsi_sell_threshold": params.rsi_sell_threshold,
                "macd_confirmation": params.macd_confirmation,
                "ma_filter": params.ma_filter,
                "initial_capital": params.initial_capital,
            },
            total_return=round(total_return, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            win_rate=round(win_rate, 4),
            num_trades=len(trade_log),
            equity_curve=[round(v, 2) for v in equity_curve],
            trade_log=trade_log,
        )

    # ------------------------------------------------------------------
    # Internal simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        df: pd.DataFrame,
        params: BacktestParameters,
        sentiment_label: str,
    ) -> Tuple[List[float], List[Dict]]:
        """Step through each bar and apply strategy logic."""
        capital = params.initial_capital
        position = 0.0      # shares held
        entry_price = 0.0
        equity_curve: List[float] = []
        trade_log: List[Dict] = []
        in_trade = False

        rsi_col = "RSI_14"
        macd_hist_col = "MACDh_12_26_9"
        sma200_col = "SMA_200"

        for i, (idx, row) in enumerate(df.iterrows()):
            price = float(row["Close"])
            rsi = row.get(rsi_col)
            macd_hist = row.get(macd_hist_col)
            sma200 = row.get(sma200_col)

            if pd.isna(rsi):
                equity_curve.append(capital + position * price)
                continue

            # --- SELL conditions ---
            if in_trade:
                sell_signal = float(rsi) > params.rsi_sell_threshold or sentiment_label == "NEGATIVE"
                if sell_signal:
                    proceeds = position * price
                    pnl = proceeds - entry_price * position
                    capital += proceeds
                    trade_log.append({
                        "date": str(idx.date()),
                        "action": "SELL",
                        "price": round(price, 2),
                        "shares": round(position, 4),
                        "pnl": round(pnl, 2),
                        "capital": round(capital, 2),
                    })
                    position = 0.0
                    in_trade = False

            # --- BUY conditions ---
            if not in_trade:
                rsi_ok = float(rsi) < params.rsi_buy_threshold
                macd_ok = (not params.macd_confirmation) or (
                    macd_hist is not None and not pd.isna(macd_hist) and float(macd_hist) > 0
                )
                ma_ok = (not params.ma_filter) or (
                    sma200 is not None and not pd.isna(sma200) and price > float(sma200)
                )
                sent_ok = sentiment_label in ("POSITIVE", "NEUTRAL")

                if rsi_ok and macd_ok and ma_ok and sent_ok:
                    position = (capital * params.position_size_pct) / price
                    entry_price = price
                    capital -= position * price
                    trade_log.append({
                        "date": str(idx.date()),
                        "action": "BUY",
                        "price": round(price, 2),
                        "shares": round(position, 4),
                        "pnl": 0.0,
                        "capital": round(capital, 2),
                    })
                    in_trade = True

            equity_curve.append(capital + position * price)

        return equity_curve, trade_log

    @staticmethod
    def _sharpe_ratio(equity_curve: List[float], risk_free_rate: float = 0.02) -> float:
        if len(equity_curve) < 2:
            return 0.0
        denom = np.array(equity_curve[:-1])
        returns = np.diff(equity_curve) / np.where(denom != 0, denom, 1.0)
        daily_rf = risk_free_rate / 252
        excess = returns - daily_rf
        std = np.std(excess)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(252))

    @staticmethod
    def _max_drawdown(equity_curve: List[float]) -> float:
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        return max_dd
