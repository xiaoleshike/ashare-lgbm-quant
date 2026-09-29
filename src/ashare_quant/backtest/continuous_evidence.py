"""Derived continuous-account evidence; no new trading or valuation policy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.backtest.engine import (
    BacktestResult,
    _validate_corporate_action_ledger,
    _validate_trade_lifecycles,
    calculate_metrics,
)
from ashare_quant.config.settings import BacktestSettings
from ashare_quant.data.exceptions import DataValidationError


def require_close(actual: object, expected: object, reason: str) -> None:
    """Apply the engine's accounting tolerance, rejecting non-finite values."""
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.allclose(a, b, rtol=1e-10, atol=1e-6)
    ):
        raise DataValidationError(f"CONTINUOUS_ACCOUNTING_MISMATCH: {reason}")


def accounting_evidence(result: BacktestResult, settings: BacktestSettings) -> pd.DataFrame:
    """Reconcile stored daily values against holdings and cash flows, not just a status."""
    daily, trades, holdings = result.daily_returns, result.trades, result.holdings
    if daily.empty or daily.trade_date.duplicated().any():
        raise DataValidationError("CONTINUOUS_DAILY_INVALID")
    dates = daily.trade_date.astype(str)
    if dates.tolist() != sorted(dates):
        raise DataValidationError("CONTINUOUS_DAILY_NOT_CHRONOLOGICAL")
    require_close(daily.cash + daily.holdings_value, daily.equity, "cash+holdings")
    if (daily[["cash", "holdings_value", "equity"]] < -1e-6).any().any():
        raise DataValidationError("CONTINUOUS_NEGATIVE_ACCOUNT")
    previous = daily.equity.shift(1, fill_value=settings.initial_cash)
    require_close(
        daily.net_return, np.where(previous > 0, daily.equity / previous - 1, 0), "returns"
    )
    _validate_corporate_action_ledger(result.corporate_actions)
    _validate_trade_lifecycles(trades, {}, result.corporate_actions)
    held = pd.Series(0.0, index=dates)
    locked = held.copy()
    if not holdings.empty:
        if holdings.duplicated(["trade_date", "position_id"]).any():
            raise DataValidationError("CONTINUOUS_DUPLICATE_HOLDING")
        if not set(holdings.trade_date).issubset(set(dates)):
            raise DataValidationError("CONTINUOUS_HOLDING_OUTSIDE_CALENDAR")
        require_close(
            holdings.market_value, holdings.shares * holdings.last_valid_close, "holding marks"
        )
        if (holdings.last_valid_price_date > holdings.trade_date).any():
            raise DataValidationError("CONTINUOUS_FUTURE_VALUATION")
        held = holdings.groupby("trade_date").market_value.sum().reindex(dates, fill_value=0)
        past = holdings[holdings.trade_date > holdings.target_exit_date]
        locked = past.groupby("trade_date").market_value.sum().reindex(dates, fill_value=0)
        if (holdings.trade_date == dates.iloc[-1]).any():
            raise DataValidationError("CONTINUOUS_UNRESOLVED_POSITION")
    require_close(daily.holdings_value, held.to_numpy(), "holding ledger")
    costs = pd.Series(0.0, index=dates)
    cashflow = costs.copy()
    gross = costs.copy()
    writeoffs = costs.copy()
    if not trades.empty:
        if not set(trades.trade_date).issubset(set(dates)):
            raise DataValidationError("CONTINUOUS_TRADE_OUTSIDE_CALENDAR")
        costs = trades.groupby("trade_date").cost.sum().reindex(dates, fill_value=0)
        filled = trades[trades.status == "filled"].copy()
        filled["cashflow"] = (
            np.where(filled.side == "buy", -filled.gross_value, filled.gross_value) - filled.cost
        )
        cashflow = filled.groupby("trade_date").cashflow.sum().reindex(dates, fill_value=0)
        gross = filled.groupby("trade_date").gross_value.sum().reindex(dates, fill_value=0)
        terminal = trades[trades.status == "terminal_writeoff"]
        writeoffs = terminal.groupby("trade_date").size().reindex(dates, fill_value=0).astype(float)
        if set(terminal.position_id) != set(result.terminal_events.position_id):
            raise DataValidationError("CONTINUOUS_TERMINAL_LEDGER_MISMATCH")
    require_close(daily.cash, settings.initial_cash + cashflow.cumsum().to_numpy(), "cash ledger")
    require_close(daily.cost, costs.to_numpy(), "cost ledger")
    require_close(
        daily.turnover, np.where(previous > 0, gross.to_numpy() / previous, 0), "turnover"
    )
    if not result.terminal_events.empty:
        if result.terminal_events.position_id.duplicated().any():
            raise DataValidationError("CONTINUOUS_DUPLICATE_TERMINAL_EVENT")
        require_close(result.terminal_events.cash_recovery, 0.0, "terminal zero-cash contract")
    output = daily[["trade_date", "cash", "holdings_value", "equity", "cost", "turnover"]].copy()
    output["accounting_residual"] = daily.cash + daily.holdings_value - daily.equity
    output["writeoff_count"] = writeoffs.to_numpy()
    output["corporate_action_mark_delta"] = (
        result.corporate_actions.groupby("effective_date")
        .mark_value_delta.sum()
        .reindex(dates, fill_value=0)
        .to_numpy()
        if not result.corporate_actions.empty
        else 0.0
    )
    output["locked_capital_value"] = locked.to_numpy()
    denominator = daily.equity.where(daily.equity > 0)
    output["locked_capital_fraction_of_equity"] = locked.to_numpy() / denominator
    output["cash_ratio"] = daily.cash / denominator
    output["invested_ratio"] = daily.holdings_value / denominator
    return output


def period_metrics(result: BacktestResult, frequency: str) -> pd.DataFrame:
    """Compound actual continuous session returns, never the parent fold returns."""
    frame = result.daily_returns.copy()
    frame["period"] = frame.trade_date.str[: 6 if frequency == "month" else 4]
    rows = []
    for period, group in frame.groupby("period", sort=True):
        ret = float(np.prod(1 + group.net_return) - 1)
        benchmark = float(np.prod(1 + group.benchmark_return) - 1)
        rows.append(
            {
                "top_n": result.top_n,
                "period": period,
                "portfolio_return": ret,
                "benchmark_return": benchmark,
                "geometric_excess_return": (1 + ret) / (1 + benchmark) - 1,
                "turnover": float(group.turnover.sum()),
                "transaction_cost": float(group.cost.sum()),
                "ending_equity": float(group.equity.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def evidence_summary(
    result: BacktestResult, reconciliation: pd.DataFrame, settings: BacktestSettings
) -> dict[str, Any]:
    """Describe utilization and risk without asserting a deeper P&L attribution."""
    r, trades = reconciliation, result.trades
    cost_columns = ("commission", "stamp_duty", "transfer_fee", "slippage", "cost")
    costs = {name: float(trades[name].sum()) if not trades.empty else 0.0 for name in cost_columns}
    resolved = (
        trades[trades.status.isin(["filled", "terminal_writeoff"]) & trades.side.eq("sell")]
        if not trades.empty
        else trades
    )
    breaches = resolved[resolved.sell_delay_breached] if not resolved.empty else resolved
    yearly = []
    if not trades.empty:
        for year, group in trades.groupby(trades.trade_date.str[:4]):
            yearly.append(
                {"year": year, **{name: float(group[name].sum()) for name in cost_columns}}
            )
    average_equity = float(r.equity.mean())
    return {
        **result.accounting_summary,
        "terminal_writeoff_assumption": "current_zero_cash_recovery",
        "pnl_decomposition": "NOT_PROVEN; raw ledgers retained for separate accounting audit",
        "cost_components": costs,
        "cost_by_year": yearly,
        "cost_over_initial_capital": costs["cost"] / settings.initial_cash,
        "cost_over_average_equity": costs["cost"] / average_equity if average_equity > 0 else None,
        "mean_cash_ratio": _finite_stat(r.cash_ratio, "mean"),
        "median_cash_ratio": _finite_stat(r.cash_ratio, "median"),
        "p95_cash_ratio": float(r.cash_ratio.quantile(0.95))
        if r.cash_ratio.notna().any()
        else None,
        "mean_invested_ratio": _finite_stat(r.invested_ratio, "mean"),
        "days_cash_ratio_above_50pct": int((r.cash_ratio > 0.5).sum()),
        "days_cash_ratio_above_90pct": int((r.cash_ratio > 0.9).sum()),
        "days_cash_approximately_zero": int(np.isclose(r.cash, 0, rtol=0, atol=1e-6).sum()),
        "cash_zero_absolute_tolerance": 1e-6,
        "undefined_ratio_days_zero_equity": int((r.equity <= 0).sum()),
        "maximum_locked_capital_fraction": _finite_stat(r.locked_capital_fraction_of_equity, "max"),
        "mean_locked_capital_fraction": _finite_stat(r.locked_capital_fraction_of_equity, "mean"),
        "days_with_locked_capital": int((r.locked_capital_value > 0).sum()),
        "breach_resolution_types": breaches.status.value_counts().to_dict()
        if not breaches.empty
        else {},
        "intent_outcomes": result.execution_intents.outcome.value_counts().to_dict(),
    }


def _finite_stat(values: pd.Series, operation: str) -> float | None:
    return float(getattr(values, operation)()) if values.notna().any() else None


def comparison_metrics(
    results: list[BacktestResult], settings: BacktestSettings, signal_end: str
) -> dict[str, Any]:
    """Report both actual resolution windows and a common marked-equity window."""
    ends = {str(r.top_n): str(r.daily_returns.trade_date.max()) for r in results}
    end = min(ends.values())
    common = {}
    for r in results:
        daily = r.daily_returns[r.daily_returns.trade_date <= end]
        trades = r.trades[r.trades.trade_date <= end] if not r.trades.empty else r.trades
        common[str(r.top_n)] = calculate_metrics(daily, trades, settings)
    return {
        "signal_end": signal_end,
        "actual_execution_end_by_top_n": ends,
        "common_comparison_end": end,
        "common_window_policy": "minimum_actual_end_marked_equity_no_forced_sale",
        "full_lifecycle": {str(r.top_n): r.metrics for r in results},
        "common_window": common,
        "common_window_trade_metrics_scope": "closed trades by cutoff; open holdings remain marked",
    }
