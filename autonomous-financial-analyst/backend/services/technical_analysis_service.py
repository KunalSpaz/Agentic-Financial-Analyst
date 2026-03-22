"""
technical_analysis_service.py
------------------------------
Computes RSI, MACD, Moving Averages (50/200), Bollinger Bands, and
Volume Trend using pandas-ta.  Accepts a DataFrame of OHLCV data and
returns both a DataFrame with all indicator columns and a summary dict.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pandas_ta as ta

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalAnalysisService:
    """
    Stateless technical indicator computation service.
    All methods accept a pandas OHLCV DataFrame and return enriched data.
    """

    # ------------------------------------------------------------------
    # Core indicator computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Append all required technical indicators to *df* in-place.

        Indicators added:
            - RSI (14)
            - MACD line, signal line, histogram
            - SMA_50, SMA_200
            - Bollinger Bands (upper, mid, lower)
            - Volume SMA (20) for volume trend

        Args:
            df: OHLCV DataFrame with columns Open/High/Low/Close/Volume.

        Returns:
            The same DataFrame with additional indicator columns.
        """
        if df.empty or len(df) < 30:
            logger.warning("Insufficient data for technical indicators (%d rows).", len(df))
            return df

        try:
            # RSI
            df.ta.rsi(length=14, append=True)

            # MACD
            df.ta.macd(fast=12, slow=26, signal=9, append=True)

            # Moving averages
            df.ta.sma(length=50, append=True)
            df.ta.sma(length=200, append=True)

            # Bollinger Bands
            df.ta.bbands(length=20, std=2, append=True)

            # Volume SMA
            df["VOLUME_SMA_20"] = df["Volume"].rolling(window=20).mean()
            df["VOLUME_RATIO"] = df["Volume"] / df["VOLUME_SMA_20"]

        except Exception as exc:
            logger.error("Error computing indicators: %s", exc)

        return df

    @staticmethod
    def get_latest_signals(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract the most recent values of all indicators from a computed DataFrame.

        Args:
            df: DataFrame with indicator columns (output of :meth:`compute_indicators`).

        Returns:
            Dict with individual indicator values and derived boolean signals.
        """
        if df.empty:
            return {}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Normalise column names – pandas-ta uses different casing
        def col(name: str) -> Optional[float]:
            for c in df.columns:
                if c.upper() == name.upper():
                    return float(latest[c]) if pd.notna(latest[c]) else None
            return None

        rsi = col("RSI_14")
        macd = col("MACD_12_26_9")
        macd_signal = col("MACDs_12_26_9")
        macd_hist = col("MACDh_12_26_9")
        sma_50 = col("SMA_50")
        sma_200 = col("SMA_200")
        bb_upper = col("BBU_20_2.0_2.0")
        bb_lower = col("BBL_20_2.0_2.0")
        bb_mid = col("BBM_20_2.0_2.0")
        close = float(latest["Close"]) if "Close" in df.columns else None
        volume_ratio = float(latest.get("VOLUME_RATIO", 1.0)) if "VOLUME_RATIO" in df.columns else None

        # Derived signals
        rsi_oversold = rsi is not None and rsi < 35
        rsi_overbought = rsi is not None and rsi > 70
        macd_bullish_crossover = (
            macd is not None and macd_signal is not None and
            macd_hist is not None and macd_hist > 0 and
            float(prev.get("MACDh_12_26_9", 0) or 0) <= 0
        )
        golden_cross = (
            sma_50 is not None and sma_200 is not None and sma_50 > sma_200
        )
        death_cross = (
            sma_50 is not None and sma_200 is not None and sma_50 < sma_200
        )
        above_bb_upper = close is not None and bb_upper is not None and close > bb_upper
        below_bb_lower = close is not None and bb_lower is not None and close < bb_lower
        high_volume = volume_ratio is not None and volume_ratio > 1.5

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "bb_mid": bb_mid,
            "close": close,
            "volume_ratio": volume_ratio,
            # Signals
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "macd_bullish_crossover": macd_bullish_crossover,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "above_bb_upper": above_bb_upper,
            "below_bb_lower": below_bb_lower,
            "high_volume": high_volume,
        }
