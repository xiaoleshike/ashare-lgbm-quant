"""Executable portfolio backtest framework."""

from ashare_quant.backtest.diagnostics import BacktestDiagnosticEngine, BacktestDiagnosticResult
from ashare_quant.backtest.engine import BacktestInputs, BacktestResult, simulate_portfolio
from ashare_quant.backtest.historical import HistoricalBacktestEngine, HistoricalBacktestResult
from ashare_quant.backtest.runner import BacktestRunner

__all__ = [
    "BacktestInputs",
    "BacktestDiagnosticEngine",
    "BacktestDiagnosticResult",
    "BacktestResult",
    "BacktestRunner",
    "HistoricalBacktestEngine",
    "HistoricalBacktestResult",
    "simulate_portfolio",
]
