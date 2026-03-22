"""
portfolio_risk_service.py
--------------------------
Computes portfolio-level risk metrics: volatility, beta, Sharpe ratio,
correlation matrix, sector exposure, max drawdown, and VaR(95%).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backend.services.market_data_service import MarketDataService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_RISK_FREE_RATE = 0.02
_BENCHMARK = "SPY"


class PortfolioRiskService:
    """
    Computes risk metrics for a portfolio defined as {ticker: weight} dict.
    """

    def __init__(self) -> None:
        self._mds = MarketDataService()

    def analyse(
        self,
        holdings: Dict[str, float],  # {ticker: weight}, weights should sum to ~1
        period: str = "1y",
    ) -> Dict[str, Any]:
        """
        Run full risk analysis on *holdings*.

        Args:
            holdings: Dict mapping ticker symbols to portfolio weights.
            period:   Historical look-back window for yfinance.

        Returns:
            Dict with volatility, beta, sharpe, correlation_matrix,
            sector_exposure, max_drawdown, var_95, and individual asset metrics.
        """
        tickers = list(holdings.keys())
        weights = np.array([holdings[t] for t in tickers])
        total_weight = weights.sum()
        if total_weight == 0:
            return {"error": "Portfolio weights sum to zero. Provide positive weights."}
        weights = weights / total_weight  # normalise

        # Fetch returns
        returns_dict: Dict[str, pd.Series] = {}
        sectors: Dict[str, str] = {}
        for ticker in tickers:
            df = self._mds.get_historical_data(ticker, period=period)
            if not df.empty:
                returns_dict[ticker] = df["Close"].pct_change().dropna()
                quote = self._mds.get_quote(ticker)
                sectors[ticker] = quote.get("sector", "Unknown")

        if not returns_dict:
            return {"error": "No market data available for any holding."}

        returns_df = pd.DataFrame(returns_dict).dropna()

        if returns_df.empty:
            return {"error": "Insufficient return data."}

        # Portfolio returns
        w = np.array([holdings.get(t, 0) for t in returns_df.columns])
        w = w / w.sum()
        port_returns = returns_df.values @ w

        # Metrics
        portfolio_volatility = float(np.std(port_returns) * math.sqrt(252))
        portfolio_return = float(np.mean(port_returns) * 252)
        sharpe = (portfolio_return - _RISK_FREE_RATE) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        max_dd = self._max_drawdown(port_returns)
        var_95 = float(np.percentile(port_returns, 5))

        # Beta vs SPY
        beta = self._compute_beta(port_returns, period)

        # Correlation matrix
        corr = returns_df.corr().round(4).to_dict()

        # Sector exposure
        sector_weights: Dict[str, float] = {}
        for ticker, weight in zip(returns_df.columns, w):
            sec = sectors.get(ticker, "Unknown")
            sector_weights[sec] = sector_weights.get(sec, 0.0) + float(weight)

        return {
            "holdings": {t: round(float(holdings[t]), 4) for t in tickers},
            "portfolio_volatility": round(portfolio_volatility, 4),
            "portfolio_annual_return": round(portfolio_return, 4),
            "portfolio_beta": round(beta, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "var_95_daily": round(var_95, 4),
            "correlation_matrix": corr,
            "sector_exposure": {k: round(v, 4) for k, v in sector_weights.items()},
        }

    def _compute_beta(self, port_returns: np.ndarray, period: str) -> float:
        """Compute portfolio beta relative to SPY."""
        try:
            spy_df = self._mds.get_historical_data(_BENCHMARK, period=period)
            spy_returns = spy_df["Close"].pct_change().dropna().values
            n = min(len(port_returns), len(spy_returns))
            if n < 30:
                return 1.0
            p = port_returns[-n:]
            s = spy_returns[-n:]
            cov = np.cov(p, s)[0, 1]
            var = np.var(s)
            return float(cov / var) if var > 0 else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        equity = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(equity)
        drawdown = np.where(peak > 0, (peak - equity) / peak, 0.0)
        return float(np.max(drawdown))
