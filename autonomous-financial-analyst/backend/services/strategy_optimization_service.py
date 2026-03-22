"""
strategy_optimization_service.py
----------------------------------
Grid-search strategy parameter optimizer.

Optimizes RSI thresholds, MACD confirmation flag, and MA filter
to maximize return, Sharpe ratio, or minimize drawdown.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.services.backtesting_service import BacktestParameters, BacktestingService
from backend.utils.logger import get_logger

logger = get_logger(__name__)

OBJECTIVES = {"maximize_return", "maximize_sharpe", "minimize_drawdown"}


class StrategyOptimizationService:
    """
    Exhaustive grid-search optimizer over key strategy parameters.
    """

    # Parameter search space
    RSI_BUY_THRESHOLDS  = [25, 30, 35, 40]
    RSI_SELL_THRESHOLDS = [60, 65, 70, 75]
    MACD_CONFIRM        = [True, False]
    MA_FILTER           = [True, False]

    def __init__(self) -> None:
        self._bt = BacktestingService()

    def optimize(
        self,
        ticker: str,
        objective: str = "maximize_return",
        period: str = "2y",
        sentiment_label: str = "NEUTRAL",
    ) -> Dict[str, Any]:
        """
        Run full grid search and return the best parameter configuration.

        Args:
            ticker:           Stock symbol.
            objective:        One of ``maximize_return``, ``maximize_sharpe``,
                              ``minimize_drawdown``.
            period:           Historical data window.
            sentiment_label:  Static sentiment bias for backtests.

        Returns:
            Dict with best_parameters, best metrics, and all_results.
        """
        if objective not in OBJECTIVES:
            raise ValueError(f"Unknown objective '{objective}'. Choose from {OBJECTIVES}.")

        all_results: List[Dict[str, Any]] = []
        best_score = -np.inf if objective != "minimize_drawdown" else np.inf
        best_result = None
        best_params: Dict[str, Any] = {}
        iterations = 0

        for rsi_buy in self.RSI_BUY_THRESHOLDS:
            for rsi_sell in self.RSI_SELL_THRESHOLDS:
                if rsi_sell <= rsi_buy:
                    continue
                for macd in self.MACD_CONFIRM:
                    for ma in self.MA_FILTER:
                        iterations += 1
                        params = BacktestParameters(
                            rsi_buy_threshold=float(rsi_buy),
                            rsi_sell_threshold=float(rsi_sell),
                            macd_confirmation=macd,
                            ma_filter=ma,
                        )
                        try:
                            result = self._bt.run_backtest(
                                ticker, params, period=period,
                                sentiment_label=sentiment_label
                            )
                        except Exception as exc:
                            logger.warning("Backtest failed (rsi_buy=%s): %s", rsi_buy, exc)
                            continue

                        entry = {
                            "rsi_buy_threshold": rsi_buy,
                            "rsi_sell_threshold": rsi_sell,
                            "macd_confirmation": macd,
                            "ma_filter": ma,
                            "total_return": result.total_return,
                            "sharpe_ratio": result.sharpe_ratio,
                            "max_drawdown": result.max_drawdown,
                            "win_rate": result.win_rate,
                            "num_trades": result.num_trades,
                        }
                        all_results.append(entry)

                        if objective == "maximize_return":
                            score = result.total_return
                            is_better = score > best_score
                        elif objective == "maximize_sharpe":
                            score = result.sharpe_ratio
                            is_better = score > best_score
                        else:  # minimize_drawdown
                            score = result.max_drawdown
                            is_better = score < best_score

                        if is_better:
                            best_score = score
                            best_result = result
                            best_params = entry

        logger.info("Optimization complete: %d iterations, best score=%.4f", iterations, best_score)

        if best_result is None:
            return {"error": "No valid backtest results found.", "iterations": iterations}

        return {
            "ticker": ticker,
            "objective": objective,
            "best_parameters": best_params,
            "best_return": best_result.total_return,
            "best_sharpe": best_result.sharpe_ratio,
            "best_drawdown": best_result.max_drawdown,
            "iterations": iterations,
            "all_results": sorted(all_results, key=lambda x: x["total_return"], reverse=True)[:20],
        }
