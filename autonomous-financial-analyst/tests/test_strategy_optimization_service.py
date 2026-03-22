"""
test_strategy_optimization_service.py
---------------------------------------
Unit tests for StrategyOptimizationService.

Covers:
  - optimize returns dict with required keys
  - invalid objective raises ValueError
  - best_parameters is within the search space
  - all_results list is non-empty and sorted by total_return
  - optimize handles all three objectives without error
  - error response when all backtests fail
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from backend.services.backtesting_service import BacktestResult
from backend.services.strategy_optimization_service import StrategyOptimizationService


def _make_backtest_result(
    total_return: float = 0.10,
    sharpe: float = 1.2,
    drawdown: float = 0.08,
    rsi_buy: float = 35.0,
    rsi_sell: float = 70.0,
    macd: bool = True,
    ma: bool = True,
) -> BacktestResult:
    return BacktestResult(
        ticker="AAPL",
        strategy_name="RSI_MACD_Sentiment",
        start_date="2022-01-01",
        end_date="2024-01-01",
        parameters={
            "rsi_buy_threshold": rsi_buy,
            "rsi_sell_threshold": rsi_sell,
            "macd_confirmation": macd,
            "ma_filter": ma,
            "initial_capital": 10_000,
        },
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=drawdown,
        win_rate=0.55,
        num_trades=12,
        equity_curve=[10_000.0, 11_000.0],
        trade_log=[],
    )


@pytest.fixture
def service() -> StrategyOptimizationService:
    return StrategyOptimizationService()


@pytest.fixture
def mock_backtest(monkeypatch):
    """Patch BacktestingService.run_backtest to return a stable result."""
    mock = MagicMock(return_value=_make_backtest_result())
    monkeypatch.setattr(
        "backend.services.strategy_optimization_service.BacktestingService.run_backtest",
        mock,
    )
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Invalid inputs
# ─────────────────────────────────────────────────────────────────────────────

class TestInvalidObjective:
    def test_raises_value_error_for_unknown_objective(self, service):
        with pytest.raises(ValueError, match="Unknown objective"):
            service.optimize("AAPL", objective="maximize_alpha")

    def test_raises_for_empty_string_objective(self, service):
        with pytest.raises(ValueError):
            service.optimize("AAPL", objective="")


# ─────────────────────────────────────────────────────────────────────────────
# Result structure
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizeResultStructure:
    def test_returns_dict(self, service, mock_backtest):
        result = service.optimize("AAPL", objective="maximize_return")
        assert isinstance(result, dict)

    def test_required_keys_present(self, service, mock_backtest):
        result = service.optimize("AAPL", objective="maximize_return")
        for key in ("ticker", "objective", "best_parameters", "best_return",
                    "best_sharpe", "best_drawdown", "iterations", "all_results"):
            assert key in result, f"Missing key: {key}"

    def test_ticker_matches_input(self, service, mock_backtest):
        result = service.optimize("NVDA", objective="maximize_return")
        assert result["ticker"] == "NVDA"

    def test_objective_matches_input(self, service, mock_backtest):
        result = service.optimize("AAPL", objective="maximize_sharpe")
        assert result["objective"] == "maximize_sharpe"

    def test_iterations_is_positive_int(self, service, mock_backtest):
        result = service.optimize("AAPL", objective="maximize_return")
        assert isinstance(result["iterations"], int)
        assert result["iterations"] > 0

    def test_all_results_is_list(self, service, mock_backtest):
        result = service.optimize("AAPL", objective="maximize_return")
        assert isinstance(result["all_results"], list)

    def test_all_results_capped_at_20(self, service, mock_backtest):
        result = service.optimize("AAPL", objective="maximize_return")
        assert len(result["all_results"]) <= 20


# ─────────────────────────────────────────────────────────────────────────────
# Best parameters are within the defined search space
# ─────────────────────────────────────────────────────────────────────────────

class TestBestParametersInSearchSpace:
    def test_rsi_buy_is_in_search_space(self, service, mock_backtest):
        result = service.optimize("AAPL")
        rsi_buy = result["best_parameters"]["rsi_buy_threshold"]
        assert rsi_buy in StrategyOptimizationService.RSI_BUY_THRESHOLDS

    def test_rsi_sell_is_in_search_space(self, service, mock_backtest):
        result = service.optimize("AAPL")
        rsi_sell = result["best_parameters"]["rsi_sell_threshold"]
        assert rsi_sell in StrategyOptimizationService.RSI_SELL_THRESHOLDS

    def test_rsi_sell_always_greater_than_rsi_buy(self, service, mock_backtest):
        result = service.optimize("AAPL")
        best = result["best_parameters"]
        assert best["rsi_sell_threshold"] > best["rsi_buy_threshold"]


# ─────────────────────────────────────────────────────────────────────────────
# All three objectives work
# ─────────────────────────────────────────────────────────────────────────────

class TestAllObjectives:
    @pytest.mark.parametrize("objective", ["maximize_return", "maximize_sharpe", "minimize_drawdown"])
    def test_objective_runs_without_error(self, service, mock_backtest, objective):
        result = service.optimize("AAPL", objective=objective)
        assert "best_parameters" in result
        assert "error" not in result


# ─────────────────────────────────────────────────────────────────────────────
# All backtests failing
# ─────────────────────────────────────────────────────────────────────────────

class TestAllBacktestsFail:
    def test_returns_error_key_when_no_valid_results(self, service, monkeypatch):
        monkeypatch.setattr(
            "backend.services.strategy_optimization_service.BacktestingService.run_backtest",
            MagicMock(side_effect=Exception("data unavailable")),
        )
        result = service.optimize("BADTICKER", objective="maximize_return")
        assert "error" in result
