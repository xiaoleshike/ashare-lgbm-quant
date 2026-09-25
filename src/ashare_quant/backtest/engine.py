"""Executable next-open portfolio simulation with evidence-grade accounting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ashare_quant.backtest.corporate_actions import (
    CORPORATE_ACTION_MARK_VALUE_TOLERANCE,
    CorporateActionExecutionPolicy,
    default_corporate_action_execution_policy,
    validate_corporate_action_execution_policy,
)
from ashare_quant.backtest.costs import ExecutionCostPolicy, TradeCosts
from ashare_quant.config.settings import BacktestSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity_transition import SecurityIdentityTransition

type DataFrame = pd.DataFrame
type BacktestPurpose = Literal["diagnostic", "oos_evidence", "executable_validation"]
type DelayedExitPolicy = Literal["fail_at_alert", "carry_to_calendar_end"]

ACCOUNTING_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class BacktestInputs:
    """Prepared inputs for one Top-N backtest."""

    signals: DataFrame
    prices: DataFrame
    calendar: tuple[str, ...]
    benchmark: DataFrame
    identity_transitions: tuple[SecurityIdentityTransition, ...] = ()
    identity_transition_version: str | None = None
    identity_transition_hash: str | None = None
    corporate_action_policy: CorporateActionExecutionPolicy = field(
        default_factory=default_corporate_action_execution_policy
    )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Complete backtest outputs for persistence."""

    top_n: int
    daily_returns: DataFrame
    trades: DataFrame
    holdings: DataFrame
    corporate_actions: DataFrame
    metrics: dict[str, float | int | None]
    accounting_summary: dict[str, float | int] = field(default_factory=dict)
    cost_policy: dict[str, object] = field(default_factory=dict)
    corporate_action_policy: dict[str, object] = field(default_factory=dict)
    execution_provenance: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Position:
    """One open stock position and its independently tracked valuation state."""

    position_id: str
    ts_code: str
    original_entry_ts_code: str
    shares: float
    entry_date: str
    entry_calendar_index: int
    target_exit_date: str
    last_valid_close: float
    last_valid_price_date: str
    valuation_status: str = "CURRENT"
    stale_valuation_days: int = 0
    delayed_exit_days: int = 0
    sell_delay_breached: bool = False


@dataclass(frozen=True, slots=True)
class PriceRow:
    """Per-stock execution, valuation, and lifecycle fields for one date."""

    open: float
    close: float
    can_buy: bool
    can_sell: bool
    is_suspended: bool = False
    is_listed: bool = True
    delist_date: str | None = None


def simulate_portfolio(
    inputs: BacktestInputs,
    *,
    top_n: int,
    settings: BacktestSettings,
    purpose: BacktestPurpose = "diagnostic",
    delayed_exit_policy: DelayedExitPolicy = "fail_at_alert",
) -> BacktestResult:
    """Simulate equal-weight Top-N signals with next-open execution.

    Evidence-grade modes fail closed on unexplained price gaps, accounting
    violations, and positions that cannot be resolved by the supplied calendar.
    """

    if settings.execution != "next_open":
        raise DataValidationError(f"BACKTEST_UNSUPPORTED_EXECUTION: execution={settings.execution}")
    if delayed_exit_policy not in {"fail_at_alert", "carry_to_calendar_end"}:
        raise DataValidationError(
            f"BACKTEST_DELAYED_EXIT_POLICY_INVALID: policy={delayed_exit_policy}"
        )
    strict = purpose != "diagnostic"
    if strict and (
        not inputs.identity_transition_version
        or not inputs.identity_transition_hash
        or len(inputs.identity_transition_hash) != 64
    ):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_REQUIRED")
    if strict:
        validate_corporate_action_execution_policy(
            inputs.corporate_action_policy.to_dict(), inputs.corporate_action_policy.policy_hash
        )
    calendar = list(inputs.calendar)
    if not calendar:
        raise DataValidationError("BACKTEST_MARKET_DATA_INCOMPLETE: empty trading calendar")
    signal_by_entry = _signals_by_entry_date(inputs.signals, calendar, top_n)
    price_map = _price_map(inputs.prices)
    transition_by_predecessor = {
        item.predecessor_ts_code: item for item in inputs.identity_transitions
    }
    if len(transition_by_predecessor) != len(inputs.identity_transitions):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_UNSUPPORTED_TOPOLOGY")
    benchmark_returns = _benchmark_returns(inputs.benchmark)
    cost_policy = ExecutionCostPolicy(settings.execution_costs)
    positions: dict[str, Position] = {}
    cash = settings.initial_cash
    previous_equity = settings.initial_cash
    daily_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    corporate_action_rows: list[dict[str, object]] = []
    counters = {
        "stale_valuation_days": 0,
        "maximum_stale_days": 0,
        "terminal_writeoffs": 0,
        "delayed_sells": 0,
        "sell_delay_breaches": 0,
        "maximum_delayed_exit_days": 0,
        "resolved_after_sell_delay_breach": 0,
        "unsupported_transition_crossings": 0,
    }

    signal_start = str(inputs.signals["trade_date"].astype(str).min())
    simulation_dates = [date for date in calendar if date >= signal_start]
    if strict:
        _validate_benchmark_coverage(inputs.benchmark, simulation_dates)
    calendar_index = {date: index for index, date in enumerate(calendar)}
    final_entry_date = max(signal_by_entry, default=simulation_dates[0])
    for current_date in simulation_dates:
        day_cost = 0.0

        _apply_effective_corporate_actions(
            positions,
            transition_by_predecessor,
            current_date=current_date,
            transition_version=str(inputs.identity_transition_version),
            transition_hash=str(inputs.identity_transition_hash),
            policy=inputs.corporate_action_policy,
            ledger_rows=corporate_action_rows,
            top_n=top_n,
        )

        for code in list(positions):
            position = positions[code]
            if current_date < position.target_exit_date:
                continue
            price = price_map.get((current_date, code))
            if _is_explicit_terminal(price, current_date):
                trade_rows.append(
                    _terminal_position(
                        current_date,
                        position,
                        top_n,
                        holding_sessions=calendar_index[current_date]
                        - position.entry_calendar_index,
                    )
                )
                counters["terminal_writeoffs"] += 1
                if position.sell_delay_breached:
                    counters["resolved_after_sell_delay_breach"] += 1
                del positions[code]
                continue
            if price is None or not _can_sell(price):
                position.delayed_exit_days += 1
                counters["delayed_sells"] += 1
                counters["maximum_delayed_exit_days"] = max(
                    counters["maximum_delayed_exit_days"], position.delayed_exit_days
                )
                if (
                    position.delayed_exit_days > settings.sell_delay_max_days
                    and not position.sell_delay_breached
                ):
                    position.sell_delay_breached = True
                    counters["sell_delay_breaches"] += 1
                trade_rows.append(
                    _rejected_trade(
                        current_date,
                        code,
                        "sell",
                        "not_sellable",
                        top_n=top_n,
                        position_id=position.position_id,
                        delayed_exit_days=position.delayed_exit_days,
                        sell_delay_breached=position.sell_delay_breached,
                    )
                )
                if (
                    position.sell_delay_breached
                    and strict
                    and delayed_exit_policy == "fail_at_alert"
                ):
                    raise DataValidationError(
                        "BACKTEST_UNRESOLVED_POSITION: maximum sell delay exceeded without "
                        f"a terminal event position_id={position.position_id} date={current_date}"
                    )
                continue
            gross_value = position.shares * price.open
            costs = cost_policy.calculate(current_date, "sell", gross_value)
            cash += gross_value - costs.total
            day_cost += costs.total
            trade_rows.append(
                _filled_trade(
                    current_date,
                    position,
                    "sell",
                    price.open,
                    gross_value,
                    costs,
                    top_n,
                    holding_sessions=calendar_index[current_date] - position.entry_calendar_index,
                )
            )
            if position.sell_delay_breached:
                counters["resolved_after_sell_delay_breach"] += 1
            del positions[code]

        candidates = [
            code for code in signal_by_entry.get(current_date, []) if code not in positions
        ]
        executable = [
            code
            for code in candidates
            if (price := price_map.get((current_date, code))) is not None and _can_buy(price)
        ]
        for code in candidates:
            price = price_map.get((current_date, code))
            if price is None or not _can_buy(price):
                trade_rows.append(
                    _rejected_trade(current_date, code, "buy", "not_buyable", top_n=top_n)
                )
        if executable and cash > 0:
            allocation = cash / len(executable)
            for code in executable:
                price = price_map[(current_date, code)]
                gross_notional = cost_policy.maximum_affordable_gross(
                    current_date, "buy", allocation
                )
                shares = gross_notional / price.open
                costs = cost_policy.calculate(current_date, "buy", gross_notional)
                total_cash_used = gross_notional + costs.total
                if shares <= 0 or total_cash_used > cash * (1.0 + 1e-9):
                    continue
                exit_date = _calendar_offset(calendar, current_date, settings.holding_period_days)
                position_id = f"{top_n}:{current_date}:{code}"
                position = Position(
                    position_id=position_id,
                    ts_code=code,
                    original_entry_ts_code=code,
                    shares=shares,
                    entry_date=current_date,
                    entry_calendar_index=calendar_index[current_date],
                    target_exit_date=exit_date,
                    last_valid_close=price.close,
                    last_valid_price_date=current_date,
                )
                positions[code] = position
                cash -= total_cash_used
                day_cost += costs.total
                trade_rows.append(
                    _filled_trade(
                        current_date,
                        position,
                        "buy",
                        price.open,
                        gross_notional,
                        costs,
                        top_n,
                    )
                )

        holdings_value = 0.0
        for position in positions.values():
            market_value = _market_value(
                position, price_map.get((current_date, position.ts_code)), current_date, strict
            )
            holdings_value += market_value
            if position.valuation_status != "CURRENT":
                counters["stale_valuation_days"] += 1
                counters["maximum_stale_days"] = max(
                    counters["maximum_stale_days"], position.stale_valuation_days
                )
            holding_rows.append(
                {
                    "trade_date": current_date,
                    "position_id": position.position_id,
                    "ts_code": position.ts_code,
                    "original_entry_ts_code": position.original_entry_ts_code,
                    "shares": position.shares,
                    "market_value": market_value,
                    "entry_date": position.entry_date,
                    "target_exit_date": position.target_exit_date,
                    "delayed_exit_days": position.delayed_exit_days,
                    "last_valid_close": position.last_valid_close,
                    "last_valid_price_date": position.last_valid_price_date,
                    "valuation_status": position.valuation_status,
                    "stale_valuation_days": position.stale_valuation_days,
                    "top_n": top_n,
                }
            )
        equity = cash + holdings_value
        _validate_accounting(cash, holdings_value, equity, positions, current_date)
        net_return = equity / previous_equity - 1.0 if previous_equity > 0 else 0.0
        gross_return = (equity + day_cost) / previous_equity - 1.0 if previous_equity > 0 else 0.0
        turnover = _daily_turnover(trade_rows, current_date, top_n, previous_equity)
        daily_rows.append(
            {
                "trade_date": current_date,
                "top_n": top_n,
                "cash": cash,
                "holdings_value": holdings_value,
                "equity": equity,
                "turnover": turnover,
                "gross_return": gross_return,
                "net_return": net_return,
                "cost": day_cost,
                "benchmark_return": benchmark_returns.get(current_date, 0.0),
            }
        )
        previous_equity = equity
        if delayed_exit_policy == "carry_to_calendar_end" and not positions:
            if current_date >= final_entry_date:
                break

    if positions and strict:
        unresolved = ",".join(sorted(position.position_id for position in positions.values()))
        raise DataValidationError(
            "BACKTEST_UNRESOLVED_POSITION: open positions remain at governed cutoff "
            f"date={simulation_dates[-1]} positions={unresolved}"
        )
    daily = pd.DataFrame(daily_rows)
    trades = pd.DataFrame(trade_rows)
    holdings = pd.DataFrame(holding_rows)
    corporate_actions = pd.DataFrame(corporate_action_rows, columns=_corporate_action_columns())
    _validate_corporate_action_ledger(corporate_actions)
    _validate_trade_lifecycles(trades, positions, corporate_actions)
    summary = _accounting_summary(daily, trades, positions, counters, corporate_actions)
    metrics = calculate_metrics(daily, trades, settings)
    metrics.update(
        {
            "corporate_action_transformations": int(len(corporate_actions)),
            "corporate_action_positions": int(
                corporate_actions["position_id"].nunique() if not corporate_actions.empty else 0
            ),
            "corporate_action_mark_value_delta_max": float(
                corporate_actions["mark_value_delta"].abs().max()
                if not corporate_actions.empty
                else 0.0
            ),
            "unsupported_transition_crossings": 0,
        }
    )
    return BacktestResult(
        top_n=top_n,
        daily_returns=daily,
        trades=trades,
        holdings=holdings,
        corporate_actions=corporate_actions,
        metrics=metrics,
        accounting_summary=summary,
        cost_policy=cost_policy.to_dict(),
        corporate_action_policy=inputs.corporate_action_policy.to_dict(),
        execution_provenance={
            "accounting_schema_version": ACCOUNTING_SCHEMA_VERSION,
            "security_identity_transition_version": inputs.identity_transition_version,
            "security_identity_transition_hash": inputs.identity_transition_hash,
            "corporate_action_execution_policy_version": inputs.corporate_action_policy.version,
            "corporate_action_execution_policy_hash": inputs.corporate_action_policy.policy_hash,
            "corporate_action_count": len(corporate_actions),
            "corporate_action_ledger_hash": _frame_hash(corporate_actions),
        },
    )


def calculate_metrics(
    daily: DataFrame, trades: DataFrame, settings: BacktestSettings
) -> dict[str, float | int | None]:
    """Compute schema-v3 metrics from compounded portfolio and session returns."""

    if daily.empty:
        return {}
    returns = daily["net_return"].astype(float)
    benchmark = daily["benchmark_return"].astype(float)
    equity = daily["equity"].astype(float)
    days = len(daily)
    total_return = float(equity.iloc[-1] / settings.initial_cash - 1.0)
    benchmark_total = float(np.prod(1.0 + benchmark.to_numpy(dtype=float)) - 1.0)
    cumulative_excess = (1.0 + total_return) / (1.0 + benchmark_total) - 1.0
    annual_return = (1.0 + total_return) ** (settings.annualization_days / days) - 1.0
    annual_vol = returns.std(ddof=1) * np.sqrt(settings.annualization_days) if days > 1 else 0.0
    daily_risk_free = (1.0 + settings.risk_free_annual_rate) ** (
        1.0 / settings.annualization_days
    ) - 1.0
    excess_daily = returns - daily_risk_free
    return_std = returns.std(ddof=1) if days > 1 else 0.0
    sharpe = (
        excess_daily.mean() / return_std * np.sqrt(settings.annualization_days)
        if return_std > 0
        else None
    )
    active = returns - benchmark
    active_std = active.std(ddof=1) if days > 1 else 0.0
    information_ratio = (
        active.mean() / active_std * np.sqrt(settings.annualization_days)
        if active_std > 0
        else None
    )
    drawdown = equity / equity.cummax() - 1.0
    filled = trades[trades["status"] == "filled"] if not trades.empty else trades
    closures = (
        trades[trades["status"].isin(["filled", "terminal_writeoff"])]
        if not trades.empty
        else trades
    )
    closed_pnl = _closed_trade_pnl(closures)
    profitable = [value for value in closed_pnl if value > 0]
    losing = [value for value in closed_pnl if value < 0]
    holding_sessions = _holding_sessions(closures)
    daily_win_rate = float((returns > 0).mean())
    average_turnover = float(daily["turnover"].astype(float).mean())
    return {
        "schema_version": ACCOUNTING_SCHEMA_VERSION,
        "days": int(days),
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annualized_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "annualized_volatility": float(annual_vol),
        "sharpe": None if sharpe is None else float(sharpe),
        "information_ratio": None if information_ratio is None else float(information_ratio),
        "benchmark_total_return": benchmark_total,
        "cumulative_excess_return": float(cumulative_excess),
        "excess_return_vs_benchmark": float(cumulative_excess),
        "maximum_drawdown": float(drawdown.min()),
        "daily_win_rate": daily_win_rate,
        "win_rate": daily_win_rate,
        "trade_win_rate": (
            None
            if not closed_pnl
            else float(sum(value > 0 for value in closed_pnl) / len(closed_pnl))
        ),
        "profit_loss_ratio": (
            None
            if not profitable or not losing
            else float(np.mean(profitable) / abs(np.mean(losing)))
        ),
        "average_two_way_turnover": average_turnover,
        "average_turnover": average_turnover,
        "average_holding_period_sessions": _mean(holding_sessions),
        "median_holding_period_sessions": _median(holding_sessions),
        "p95_holding_period_sessions": _percentile(holding_sessions, 95),
        "average_holding_period": _mean(holding_sessions),
        "filled_trades": int(len(filled)),
        "terminal_writeoffs": (
            int((trades["status"] == "terminal_writeoff").sum()) if not trades.empty else 0
        ),
        "written_off_positions": (
            int((trades["status"] == "terminal_writeoff").sum()) if not trades.empty else 0
        ),
        "rejected_trades": int((trades["status"] == "rejected").sum()) if not trades.empty else 0,
    }


def _closed_trade_pnl(filled: DataFrame) -> list[float]:
    if filled.empty:
        return []
    cost_basis: dict[str, float] = {}
    closed: list[float] = []
    for row in filled.itertuples(index=False):
        typed = cast(Any, row)
        position_id = str(getattr(typed, "position_id", typed.ts_code))
        if str(typed.side) == "buy":
            cost_basis[position_id] = float(typed.gross_value) + float(typed.cost)
            continue
        basis = cost_basis.pop(position_id, None)
        if basis is not None:
            closed.append(float(typed.gross_value) - float(typed.cost) - basis)
    return closed


def _holding_sessions(filled: DataFrame) -> list[float]:
    if filled.empty or "holding_sessions" not in filled:
        return []
    return [
        float(value)
        for value in filled.loc[
            (filled["side"] == "sell") & filled["holding_sessions"].notna(),
            "holding_sessions",
        ].tolist()
    ]


def _signals_by_entry_date(
    signals: DataFrame, calendar: list[str], top_n: int
) -> dict[str, list[str]]:
    by_entry: dict[str, list[str]] = {}
    for trade_date, group in signals.groupby("trade_date", sort=True):
        entry_date = _calendar_offset(calendar, str(trade_date), 1)
        selected = (
            group.sort_values(["score", "ts_code"], ascending=[False, True])
            .head(top_n)["ts_code"]
            .astype(str)
            .tolist()
        )
        by_entry.setdefault(entry_date, []).extend(selected)
    return by_entry


def _apply_effective_corporate_actions(
    positions: dict[str, Position],
    transition_by_predecessor: dict[str, SecurityIdentityTransition],
    *,
    current_date: str,
    transition_version: str,
    transition_hash: str,
    policy: CorporateActionExecutionPolicy,
    ledger_rows: list[dict[str, object]],
    top_n: int | None = None,
) -> None:
    """Validate all due chains, then atomically migrate position identities."""

    plans: list[tuple[str, Position, tuple[SecurityIdentityTransition, ...]]] = []
    occupied = set(positions)
    final_codes: set[str] = set()
    for source_code, position in positions.items():
        chain: list[SecurityIdentityTransition] = []
        current_code = position.ts_code
        seen: set[str] = set()
        previous_effective_date = ""
        while (transition := transition_by_predecessor.get(current_code)) is not None:
            if transition.effective_date > current_date:
                break
            if current_code in seen or transition.successor_ts_code in seen:
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONFLICT: cycle")
            if previous_effective_date and transition.effective_date <= previous_effective_date:
                raise DataValidationError(
                    "SECURITY_IDENTITY_TRANSITION_CONFLICT: transition chain dates are "
                    "not strictly increasing"
                )
            eligibility = policy.classify(transition)
            if not eligibility.supported:
                raise DataValidationError(
                    "CORPORATE_ACTION_EXECUTION_UNSUPPORTED: predecessor position crosses "
                    f"security-code transition position_id={position.position_id} "
                    f"predecessor={transition.predecessor_ts_code} "
                    f"successor={transition.successor_ts_code} "
                    f"effective_date={transition.effective_date} reason={eligibility.reason}"
                )
            seen.add(current_code)
            chain.append(transition)
            previous_effective_date = transition.effective_date
            current_code = transition.successor_ts_code
        if not chain:
            continue
        traversed_successors = {item.successor_ts_code for item in chain}
        conflicting = (traversed_successors & occupied) - {source_code}
        if conflicting:
            raise DataValidationError(
                "CORPORATE_ACTION_POSITION_COLLISION: successor already has an independent "
                f"position codes={sorted(conflicting)}"
            )
        if current_code in final_codes:
            raise DataValidationError(
                f"CORPORATE_ACTION_POSITION_COLLISION: multiple positions target {current_code}"
            )
        final_codes.add(current_code)
        plans.append((source_code, position, tuple(chain)))

    for source_code, position, plan_chain in plans:
        del positions[source_code]
        for transition in plan_chain:
            ratio = transition.share_conversion_ratio
            if ratio is None:
                raise DataValidationError("CORPORATE_ACTION_EXECUTION_UNSUPPORTED: ratio missing")
            old_shares = position.shares
            old_close = position.last_valid_close
            new_shares = old_shares * ratio
            adjusted_close = old_close / ratio
            before = old_shares * old_close
            after = new_shares * adjusted_close
            values = (old_shares, old_close, ratio, new_shares, adjusted_close, before, after)
            if any(not np.isfinite(value) or value <= 0 for value in values):
                raise DataValidationError(
                    "BACKTEST_ACCOUNTING_INVARIANT_FAILED: non-finite corporate action value"
                )
            delta = after - before
            if not np.isclose(
                before,
                after,
                rtol=CORPORATE_ACTION_MARK_VALUE_TOLERANCE,
                atol=CORPORATE_ACTION_MARK_VALUE_TOLERANCE,
            ):
                raise DataValidationError(
                    "BACKTEST_ACCOUNTING_INVARIANT_FAILED: corporate action mark value changed"
                )
            ledger_rows.append(
                {
                    "effective_date": transition.effective_date,
                    "position_id": position.position_id,
                    "predecessor_ts_code": transition.predecessor_ts_code,
                    "successor_ts_code": transition.successor_ts_code,
                    "transition_type": transition.transition_type,
                    "continuity_type": transition.continuity_type,
                    "old_shares": old_shares,
                    "new_shares": new_shares,
                    "share_conversion_ratio": ratio,
                    "old_last_valid_close": old_close,
                    "adjusted_last_valid_close": adjusted_close,
                    "pre_action_mark_value": before,
                    "post_action_mark_value": after,
                    "mark_value_delta": delta,
                    "cash_delta": 0.0,
                    "transaction_cost": 0.0,
                    "turnover_contribution": 0.0,
                    "transition_version": transition_version,
                    "transition_hash": transition_hash,
                    "evidence_package_id": transition.evidence_package_id,
                    "evidence_package_hash": transition.evidence_package_hash,
                    "execution_policy_version": policy.version,
                    "execution_policy_hash": policy.policy_hash,
                    "top_n": top_n,
                }
            )
            position.ts_code = transition.successor_ts_code
            position.shares = new_shares
            position.last_valid_close = adjusted_close
        positions[position.ts_code] = position


def _corporate_action_columns() -> list[str]:
    return [
        "effective_date",
        "position_id",
        "predecessor_ts_code",
        "successor_ts_code",
        "transition_type",
        "continuity_type",
        "old_shares",
        "new_shares",
        "share_conversion_ratio",
        "old_last_valid_close",
        "adjusted_last_valid_close",
        "pre_action_mark_value",
        "post_action_mark_value",
        "mark_value_delta",
        "cash_delta",
        "transaction_cost",
        "turnover_contribution",
        "transition_version",
        "transition_hash",
        "evidence_package_id",
        "evidence_package_hash",
        "execution_policy_version",
        "execution_policy_hash",
        "top_n",
    ]


def _validate_corporate_action_ledger(frame: DataFrame) -> None:
    if frame.empty:
        return
    numeric = frame[
        [
            "old_shares",
            "new_shares",
            "share_conversion_ratio",
            "old_last_valid_close",
            "adjusted_last_valid_close",
            "pre_action_mark_value",
            "post_action_mark_value",
            "mark_value_delta",
            "cash_delta",
            "transaction_cost",
            "turnover_contribution",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise DataValidationError(
            "BACKTEST_ACCOUNTING_INVARIANT_FAILED: corporate action ledger is non-finite"
        )
    if not np.allclose(
        frame["pre_action_mark_value"].to_numpy(dtype=float),
        frame["post_action_mark_value"].to_numpy(dtype=float),
        rtol=CORPORATE_ACTION_MARK_VALUE_TOLERANCE,
        atol=CORPORATE_ACTION_MARK_VALUE_TOLERANCE,
    ):
        raise DataValidationError(
            "BACKTEST_ACCOUNTING_INVARIANT_FAILED: corporate action ledger does not reconcile"
        )
    if not np.equal(
        frame[["cash_delta", "transaction_cost", "turnover_contribution"]].to_numpy(dtype=float),
        0.0,
    ).all():
        raise DataValidationError(
            "BACKTEST_ACCOUNTING_INVARIANT_FAILED: unsupported corporate action consideration"
        )


def _frame_hash(frame: DataFrame) -> str:
    records = frame.where(pd.notna(frame), None).to_dict("records")
    payload = json.dumps(
        records,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _calendar_offset(calendar: list[str], date: str, offset: int) -> str:
    index = calendar.index(date)
    return calendar[min(index + offset, len(calendar) - 1)]


def _price_map(prices: DataFrame) -> dict[tuple[str, str], PriceRow]:
    mapping: dict[tuple[str, str], PriceRow] = {}
    for row in prices.itertuples(index=False):
        typed = cast(Any, row)
        delist_date = getattr(typed, "delist_date", None)
        mapping[(str(typed.trade_date), str(typed.ts_code))] = PriceRow(
            open=float(typed.open),
            close=float(typed.close),
            can_buy=bool(typed.can_buy),
            can_sell=bool(typed.can_sell),
            is_suspended=bool(getattr(typed, "is_suspended", False)),
            is_listed=bool(getattr(typed, "is_listed", True)),
            delist_date=None if pd.isna(delist_date) else str(delist_date),
        )
    return mapping


def _benchmark_returns(benchmark: DataFrame) -> dict[str, float]:
    if benchmark.empty:
        return {}
    frame = benchmark.sort_values("trade_date").copy()
    frame["benchmark_return"] = frame["close"].astype(float).pct_change().fillna(0.0)
    return dict(zip(frame["trade_date"].astype(str), frame["benchmark_return"], strict=False))


def _validate_benchmark_coverage(benchmark: DataFrame, dates: list[str]) -> None:
    if benchmark.empty:
        raise DataValidationError("BACKTEST_MARKET_DATA_INCOMPLETE: benchmark is empty")
    frame = benchmark.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    available = set(frame.loc[np.isfinite(frame["close"].astype(float)), "trade_date"])
    missing = [date for date in dates if date not in available]
    if missing:
        raise DataValidationError(
            f"BACKTEST_MARKET_DATA_INCOMPLETE: benchmark dates are missing: {missing[:5]}"
        )


def _can_buy(price: PriceRow) -> bool:
    return bool(price.can_buy and price.is_listed and np.isfinite(price.open) and price.open > 0)


def _can_sell(price: PriceRow) -> bool:
    return bool(price.can_sell and np.isfinite(price.open) and price.open > 0)


def _market_value(position: Position, price: PriceRow | None, date: str, strict: bool) -> float:
    if price is not None and np.isfinite(price.close) and price.close > 0:
        position.last_valid_close = price.close
        position.last_valid_price_date = date
        position.valuation_status = "CURRENT"
        position.stale_valuation_days = 0
        return position.shares * price.close
    if price is not None and price.is_suspended:
        position.valuation_status = "STALE_SUSPENDED"
    elif strict:
        raise DataValidationError(
            "BACKTEST_MARKET_DATA_INCOMPLETE: unexplained missing or malformed close "
            f"position_id={position.position_id} date={date}"
        )
    else:
        position.valuation_status = "STALE_MISSING_DATA"
    position.stale_valuation_days += 1
    return position.shares * position.last_valid_close


def _is_explicit_terminal(price: PriceRow | None, date: str) -> bool:
    return bool(
        price is not None
        and not price.is_listed
        and price.delist_date is not None
        and price.delist_date <= date
    )


def _validate_accounting(
    cash: float,
    holdings_value: float,
    equity: float,
    positions: dict[str, Position],
    date: str,
) -> None:
    values = (cash, holdings_value, equity)
    if any(not np.isfinite(value) for value in values):
        raise DataValidationError(
            f"BACKTEST_ACCOUNTING_INVARIANT_FAILED: non-finite portfolio value date={date}"
        )
    if cash < -1e-6 or holdings_value < -1e-6 or equity < -1e-6:
        raise DataValidationError(
            f"BACKTEST_ACCOUNTING_INVARIANT_FAILED: negative unlevered value date={date}"
        )
    if not np.isclose(equity, cash + holdings_value, rtol=1e-10, atol=1e-6):
        raise DataValidationError(
            f"BACKTEST_ACCOUNTING_INVARIANT_FAILED: equity reconciliation failed date={date}"
        )
    if any(
        position.shares <= 0 or not np.isfinite(position.shares) for position in positions.values()
    ):
        raise DataValidationError(
            f"BACKTEST_ACCOUNTING_INVARIANT_FAILED: invalid position shares date={date}"
        )


def _filled_trade(
    date: str,
    position: Position,
    side: str,
    price: float,
    gross_value: float,
    costs: TradeCosts,
    top_n: int,
    *,
    holding_sessions: int | None = None,
) -> dict[str, object]:
    return {
        "trade_date": date,
        "position_id": position.position_id,
        "ts_code": position.ts_code,
        "side": side,
        "status": "filled",
        "shares": position.shares,
        "price": price,
        "gross_value": gross_value,
        "commission": costs.commission,
        "stamp_duty": costs.stamp_duty,
        "transfer_fee": costs.transfer_fee,
        "slippage": costs.slippage,
        "cost": costs.total,
        "holding_sessions": holding_sessions,
        "delayed_exit_days": position.delayed_exit_days,
        "sell_delay_breached": position.sell_delay_breached,
        "reason": "",
        "top_n": top_n,
    }


def _daily_turnover(
    trade_rows: list[dict[str, object]], date: str, top_n: int, previous_equity: float
) -> float:
    if previous_equity <= 0:
        return 0.0
    traded = [
        float(cast(float, row["gross_value"]))
        for row in trade_rows
        if row.get("trade_date") == date
        and row.get("top_n") == top_n
        and row.get("status") == "filled"
    ]
    return sum(traded) / previous_equity


def _rejected_trade(
    date: str,
    code: str,
    side: str,
    reason: str,
    *,
    top_n: int | None = None,
    position_id: str | None = None,
    delayed_exit_days: int = 0,
    sell_delay_breached: bool = False,
) -> dict[str, object]:
    return {
        "trade_date": date,
        "position_id": position_id,
        "ts_code": code,
        "side": side,
        "status": "rejected",
        "shares": 0.0,
        "price": np.nan,
        "gross_value": 0.0,
        "commission": 0.0,
        "stamp_duty": 0.0,
        "transfer_fee": 0.0,
        "slippage": 0.0,
        "cost": 0.0,
        "holding_sessions": None,
        "delayed_exit_days": delayed_exit_days,
        "sell_delay_breached": sell_delay_breached,
        "reason": reason,
        "top_n": top_n,
    }


def _terminal_position(
    date: str,
    position: Position,
    top_n: int,
    *,
    holding_sessions: int,
) -> dict[str, object]:
    return {
        **_rejected_trade(
            date,
            position.ts_code,
            "sell",
            "verified_terminal_security",
            top_n=top_n,
            position_id=position.position_id,
            delayed_exit_days=position.delayed_exit_days,
            sell_delay_breached=position.sell_delay_breached,
        ),
        "status": "terminal_writeoff",
        "shares": position.shares,
        "holding_sessions": holding_sessions,
    }


def _validate_trade_lifecycles(
    trades: DataFrame, positions: dict[str, Position], corporate_actions: DataFrame
) -> None:
    if trades.empty:
        return
    opened: dict[str, float] = {}
    events: list[tuple[str, int, str, float, float]] = []
    for row in trades.itertuples(index=False):
        typed = cast(Any, row)
        if str(typed.status) not in {"filled", "terminal_writeoff"}:
            continue
        priority = 0 if str(typed.side) == "buy" else 2
        events.append(
            (
                str(typed.trade_date),
                priority,
                str(typed.position_id),
                float(typed.shares),
                float(typed.cost),
            )
        )
    for row in corporate_actions.itertuples(index=False):
        typed = cast(Any, row)
        events.append(
            (
                str(typed.effective_date),
                1,
                str(typed.position_id),
                float(typed.new_shares),
                0.0,
            )
        )
    for _, priority, position_id, shares, costs in sorted(events):
        if shares < 0 or costs < 0 or not np.isfinite(shares + costs):
            raise DataValidationError(
                "BACKTEST_ACCOUNTING_INVARIANT_FAILED: invalid trade shares or costs"
            )
        if priority == 0:
            if position_id in opened:
                raise DataValidationError(
                    "BACKTEST_ACCOUNTING_INVARIANT_FAILED: duplicate open position"
                )
            opened[position_id] = shares
        elif priority == 1:
            if position_id not in opened:
                raise DataValidationError(
                    "BACKTEST_ACCOUNTING_INVARIANT_FAILED: corporate action without position"
                )
            opened[position_id] = shares
        else:
            held = opened.pop(position_id, None)
            if held is None or shares > held + 1e-9:
                raise DataValidationError(
                    "BACKTEST_ACCOUNTING_INVARIANT_FAILED: sell exceeds held position"
                )
    if set(opened) != {position.position_id for position in positions.values()}:
        raise DataValidationError(
            "BACKTEST_ACCOUNTING_INVARIANT_FAILED: position lifecycle reconciliation failed"
        )


def _accounting_summary(
    daily: DataFrame,
    trades: DataFrame,
    positions: dict[str, Position],
    counters: dict[str, int],
    corporate_actions: DataFrame,
) -> dict[str, float | int]:
    filled = trades[trades["status"] == "filled"] if not trades.empty else trades
    return {
        "accounting_schema_version": ACCOUNTING_SCHEMA_VERSION,
        "stale_valuation_days": counters["stale_valuation_days"],
        "maximum_stale_days": counters["maximum_stale_days"],
        "unresolved_positions": len(positions),
        "terminal_writeoffs": counters["terminal_writeoffs"],
        "filled_buys": int((filled["side"] == "buy").sum()) if not filled.empty else 0,
        "filled_sells": int((filled["side"] == "sell").sum()) if not filled.empty else 0,
        "rejected_buys": int(((trades["status"] == "rejected") & (trades["side"] == "buy")).sum())
        if not trades.empty
        else 0,
        "rejected_sells": int(((trades["status"] == "rejected") & (trades["side"] == "sell")).sum())
        if not trades.empty
        else 0,
        "delayed_sells": counters["delayed_sells"],
        "sell_delay_breaches": counters["sell_delay_breaches"],
        "maximum_delayed_exit_days": counters["maximum_delayed_exit_days"],
        "resolved_after_sell_delay_breach": counters["resolved_after_sell_delay_breach"],
        "corporate_action_transformations": len(corporate_actions),
        "corporate_action_positions": int(
            corporate_actions["position_id"].nunique() if not corporate_actions.empty else 0
        ),
        "corporate_action_mark_value_delta_max": float(
            corporate_actions["mark_value_delta"].abs().max()
            if not corporate_actions.empty
            else 0.0
        ),
        "unsupported_transition_crossings": counters["unsupported_transition_crossings"],
        "commission_total": _column_sum(trades, "commission"),
        "stamp_duty_total": _column_sum(trades, "stamp_duty"),
        "transfer_fee_total": _column_sum(trades, "transfer_fee"),
        "slippage_total": _column_sum(trades, "slippage"),
        "total_cost": _column_sum(trades, "cost"),
        "largest_positive_daily_return": _column_extreme(daily, "net_return", "max"),
        "largest_negative_daily_return": _column_extreme(daily, "net_return", "min"),
        "largest_equity_jump": _largest_equity_jump(daily),
    }


def _column_sum(frame: DataFrame, column: str) -> float:
    return float(frame[column].astype(float).sum()) if not frame.empty and column in frame else 0.0


def _column_extreme(frame: DataFrame, column: str, operation: str) -> float:
    if frame.empty:
        return 0.0
    series = frame[column].astype(float)
    return float(series.max() if operation == "max" else series.min())


def _largest_equity_jump(daily: DataFrame) -> float:
    if daily.empty:
        return 0.0
    return float(daily["equity"].astype(float).diff().abs().fillna(0.0).max())


def _mean(values: list[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _median(values: list[float]) -> float | None:
    return None if not values else float(np.median(values))


def _percentile(values: list[float], value: float) -> float | None:
    return None if not values else float(np.percentile(values, value))
