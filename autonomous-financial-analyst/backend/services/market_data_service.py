"""
market_data_service.py
----------------------
Wraps yfinance to provide OHLCV data, stock metadata, and real-time quotes.
Results are cached in-memory using a simple TTL dict to reduce API calls.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from backend.utils.config import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Simple in-memory TTL cache: {key: (timestamp, value)}
_cache: Dict[str, tuple[float, Any]] = {}


def _cache_get(key: str) -> Optional[Any]:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < settings.market_data_cache_ttl:
        return entry[1]
    return None


def _cache_set(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


class MarketDataService:
    """
    Facade around yfinance for fetching stock data.

    All public methods cache results for ``settings.market_data_cache_ttl`` seconds.
    """

    # ------------------------------------------------------------------
    # OHLCV historical data
    # ------------------------------------------------------------------

    @staticmethod
    def get_historical_data(
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV history for *ticker*.

        Args:
            ticker:   Stock symbol, e.g. ``"AAPL"``.
            period:   yfinance period string – ``"1y"``, ``"6mo"``, ``"3mo"`` etc.
            interval: Bar interval – ``"1d"``, ``"1h"``, ``"15m"`` etc.

        Returns:
            DataFrame with columns Open/High/Low/Close/Volume indexed by Date.
            Empty DataFrame on error.
        """
        cache_key = f"hist:{ticker}:{period}:{interval}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period=period, interval=interval)
            df.index = pd.to_datetime(df.index)
            _cache_set(cache_key, df)
            logger.debug("Fetched %d bars for %s (%s/%s)", len(df), ticker, period, interval)
            return df
        except Exception as exc:
            logger.error("Failed to fetch history for %s: %s", ticker, exc)
            return pd.DataFrame()

    @staticmethod
    def get_quote(ticker: str) -> Dict[str, Any]:
        """
        Return a real-time / delayed quote snapshot for *ticker*.

        Returns:
            Dict with keys: ticker, price, change_pct, volume, market_cap,
            pe_ratio, 52w_high, 52w_low, company_name, sector, industry.
        """
        cache_key = f"quote:{ticker}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        try:
            info = yf.Ticker(ticker).info
            result = {
                "ticker": ticker,
                "company_name": info.get("longName", ticker),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "price": (
                    info.get("currentPrice")
                    or info.get("regularMarketPrice")
                    or info.get("previousClose")
                    or info.get("navPrice")
                ),
                "change_pct": (
                    info.get("regularMarketChangePercent")
                    or info.get("regularMarketChange")
                ),
                "volume": info.get("volume"),
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "beta": info.get("beta"),
                "dividend_yield": info.get("dividendYield"),
                "eps": info.get("trailingEps"),
            }
            _cache_set(cache_key, result)
            return result
        except Exception as exc:
            logger.error("Failed to fetch quote for %s: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc)}

    @staticmethod
    def get_bulk_quotes(tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch quotes for multiple tickers.

        Args:
            tickers: List of ticker symbols.

        Returns:
            List of quote dicts (see :meth:`get_quote`).
        """
        return [MarketDataService.get_quote(t) for t in tickers]

    @staticmethod
    def get_market_index_data() -> Dict[str, Any]:
        """
        Return latest data for major market indices (SPY, QQQ, DIA).

        Returns:
            Dict keyed by index symbol with price/change_pct fields.
        """
        indices = {"SPY": "S&P 500", "QQQ": "NASDAQ 100", "DIA": "Dow Jones", "^VIX": "VIX"}
        result = {}
        for sym, name in indices.items():
            quote = MarketDataService.get_quote(sym)
            display_sym = "VIX" if sym == "^VIX" else sym
            result[display_sym] = {
                "name": name,
                "price": quote.get("price"),
                "change_pct": quote.get("change_pct"),
            }
        return result
