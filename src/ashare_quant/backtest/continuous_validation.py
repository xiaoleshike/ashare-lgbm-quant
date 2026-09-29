"""Recursive validation of continuous replay sources, ledgers and derived tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.backtest.continuous import (
    CONTRACT,
    TABLES,
    ContinuousStrictOOSReplayService,
    ReplayPreflight,
)
from ashare_quant.backtest.continuous_evidence import (
    accounting_evidence,
    comparison_metrics,
    evidence_summary,
    period_metrics,
    require_close,
)
from ashare_quant.backtest.continuous_source import file_hash, frame_hash, payload_hash, read_json
from ashare_quant.backtest.corporate_actions import default_corporate_action_execution_policy
from ashare_quant.backtest.costs import ExecutionCostPolicy
from ashare_quant.backtest.engine import (
    BacktestResult,
    _accounting_summary,
    _signals_by_entry_date,
    calculate_metrics,
)
from ashare_quant.backtest.executable_validation import REQUIRED_TOP_N, _signals
from ashare_quant.data.exceptions import DataValidationError


def validate_continuous_artifact(path: Path) -> dict[str, Any]:
    """Fail closed on one corrupt child, mismatched source, or invalid business invariant."""
    manifest = read_json(path / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != CONTRACT
        or manifest.get("status") != "COMPLETE"
    ):
        raise DataValidationError("CONTINUOUS_MANIFEST_INVALID")
    logical = manifest.get("logical_identity")
    digest = payload_hash(logical)
    if (
        manifest.get("identity") != digest
        or manifest.get("run_id") != f"continuous_strict_oos_{digest[:24]}"
        or path.name != manifest["run_id"]
    ):
        raise DataValidationError("CONTINUOUS_IDENTITY_MISMATCH")
    expected = {
        "source_inventory.json",
        "stitched_predictions.parquet",
        "signal_lineage.parquet",
        "comparison.json",
        "monthly_metrics.csv",
        "yearly_metrics.csv",
    }
    for n in REQUIRED_TOP_N:
        expected.update(
            f"top_{n}/{name}.parquet" for name in (*TABLES, "daily_accounting_reconciliation")
        )
        expected.update(
            f"top_{n}/{name}.json" for name in ("metrics", "accounting_summary", "evidence_summary")
        )
    hashes = manifest.get("artifact_hashes")
    actual = {
        p.relative_to(path).as_posix()
        for p in path.rglob("*")
        if p.is_file() and p != path / "manifest.json"
    }
    directories = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_dir()}
    if (
        not isinstance(hashes, dict)
        or set(hashes) != expected
        or actual != expected
        or directories != {f"top_{n}" for n in REQUIRED_TOP_N}
    ):
        raise DataValidationError("CONTINUOUS_CHILD_SET_MISMATCH")
    for name, expected_hash in hashes.items():
        if (path / name).is_symlink() or file_hash(path / name) != expected_hash:
            raise DataValidationError(f"CONTINUOUS_CHILD_HASH_MISMATCH: {name}")
    inventory = read_json(path / "source_inventory.json")
    if inventory.get("logical_identity") != logical:
        raise DataValidationError("CONTINUOUS_SOURCE_INVENTORY_MISMATCH")
    service = ContinuousStrictOOSReplayService(
        **{k: Path(v) for k, v in inventory["locators"].items()}, output_root=path.parent
    )
    context = service.preflight()
    if context.logical != logical:
        raise DataValidationError("CONTINUOUS_SOURCE_IDENTITY_MISMATCH")
    predictions = pd.read_parquet(path / "stitched_predictions.parquet")
    lineage = pd.read_parquet(path / "signal_lineage.parquet")
    if (
        frame_hash(predictions) != context.source.identity["stitched_content_hash"]
        or frame_hash(lineage) != context.source.identity["signal_lineage_hash"]
    ):
        raise DataValidationError("CONTINUOUS_PREDICTION_SOURCE_MISMATCH")
    results = []
    for n in REQUIRED_TOP_N:
        folder = path / f"top_{n}"
        frames = {name: pd.read_parquet(folder / f"{name}.parquet") for name in TABLES}
        result = BacktestResult(
            top_n=n,
            daily_returns=frames["daily_returns"],
            trades=frames["trades"],
            holdings=frames["holdings"],
            corporate_actions=frames["corporate_actions"],
            execution_intents=frames["execution_intents"],
            terminal_events=frames["terminal_events"],
            metrics=read_json(folder / "metrics.json"),
            accounting_summary=read_json(folder / "accounting_summary.json"),
        )
        _validate_portfolio(result, context)
        reconciliation = accounting_evidence(result, context.settings)
        _same_frame(
            pd.read_parquet(folder / "daily_accounting_reconciliation.parquet"),
            reconciliation,
            "daily reconciliation",
        )
        if read_json(folder / "evidence_summary.json") != evidence_summary(
            result, reconciliation, context.settings
        ):
            raise DataValidationError("CONTINUOUS_EVIDENCE_SUMMARY_MISMATCH")
        results.append(result)
    if read_json(path / "comparison.json") != comparison_metrics(
        results, context.settings, context.source.identity["signal_end"]
    ):
        raise DataValidationError("CONTINUOUS_COMPARISON_MISMATCH")
    for frequency in ("month", "year"):
        expected_frame = pd.concat(
            [period_metrics(r, frequency) for r in results], ignore_index=True
        )
        actual_frame = pd.read_csv(path / f"{frequency}ly_metrics.csv", dtype={"period": str})
        _same_frame(actual_frame, expected_frame, "period returns")
    return manifest


def _same_frame(actual: pd.DataFrame, expected: pd.DataFrame, reason: str) -> None:
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-10, rtol=1e-10)
    except AssertionError as error:
        raise DataValidationError(f"CONTINUOUS_TABLE_MISMATCH: {reason}") from error


def _validate_portfolio(result: BacktestResult, context: ReplayPreflight) -> None:
    daily, trades, holdings, actions = (
        result.daily_returns,
        result.trades,
        result.holdings,
        result.corporate_actions,
    )
    end = str(daily.trade_date.max())
    if (
        end > context.logical["execution_inputs"]["governed_execution_cutoff"]
        or end >= context.logical["execution_inputs"]["lockbox_start"]
    ):
        raise DataValidationError("CONTINUOUS_CUTOFF_VIOLATION")
    if (
        daily.trade_date.tolist() != [d for d in context.calendar if d <= end]
        or end < context.source.identity["signal_end"]
    ):
        raise DataValidationError("CONTINUOUS_DAILY_COVERAGE_MISMATCH")
    for frame in (
        daily,
        trades,
        holdings,
        actions,
        result.execution_intents,
        result.terminal_events,
    ):
        if not frame.empty and not frame.top_n.eq(result.top_n).all():
            raise DataValidationError("CONTINUOUS_TOP_N_MISMATCH")
    metrics = calculate_metrics(daily, trades, context.settings)
    metrics.update(
        {
            "corporate_action_transformations": len(actions),
            "corporate_action_positions": int(actions.position_id.nunique())
            if not actions.empty
            else 0,
            "corporate_action_mark_value_delta_max": float(actions.mark_value_delta.abs().max())
            if not actions.empty
            else 0.0,
            "unsupported_transition_crossings": 0,
        }
    )
    if result.metrics != metrics:
        raise DataValidationError("CONTINUOUS_METRICS_MISMATCH")
    counters = {
        "stale_valuation_days": int(holdings.valuation_status.ne("CURRENT").sum())
        if not holdings.empty
        else 0,
        "maximum_stale_days": int(holdings.stale_valuation_days.max()) if not holdings.empty else 0,
        "terminal_writeoffs": 0,
        "delayed_sells": 0,
        "sell_delay_breaches": 0,
        "maximum_delayed_exit_days": 0,
        "resolved_after_sell_delay_breach": 0,
        "unsupported_transition_crossings": 0,
    }
    if not trades.empty:
        delayed = trades[trades.status.eq("rejected") & trades.side.eq("sell")]
        breached = trades[trades.sell_delay_breached]
        counters.update(
            {
                "terminal_writeoffs": int(trades.status.eq("terminal_writeoff").sum()),
                "delayed_sells": len(delayed),
                "sell_delay_breaches": int(breached.position_id.nunique()),
                "maximum_delayed_exit_days": int(trades.delayed_exit_days.max()),
                "resolved_after_sell_delay_breach": int(
                    (
                        breached.side.eq("sell")
                        & breached.status.isin(["filled", "terminal_writeoff"])
                    ).sum()
                ),
            }
        )
    if result.accounting_summary != _accounting_summary(daily, trades, {}, counters, actions):
        raise DataValidationError("CONTINUOUS_ACCOUNTING_SUMMARY_MISMATCH")
    selected = _signals_by_entry_date(
        _signals(context.source.predictions[["trade_date", "ts_code", "prediction_score"]]),
        list(context.calendar),
        result.top_n,
    )
    expected = [
        (context.calendar[context.calendar.index(date) - 1], date, code, rank)
        for date, codes in selected.items()
        for rank, code in enumerate(codes, 1)
    ]
    intents = result.execution_intents
    actual = list(
        intents[["signal_date", "trade_date", "ts_code", "selected_rank"]].itertuples(
            index=False, name=None
        )
    )
    if (
        actual != expected
        or not intents.selected.all()
        or not intents.outcome.isin(
            [
                "FILLED",
                "NOT_BUYABLE",
                "ALREADY_HELD",
                "NO_AVAILABLE_CASH",
                "INSUFFICIENT_AFFORDABLE_GROSS",
                "TERMINAL",
            ]
        ).all()
    ):
        raise DataValidationError("CONTINUOUS_INTENT_COVERAGE_MISMATCH")
    filled_intents = intents[intents.outcome.eq("FILLED")]
    buys = (
        trades[trades.status.eq("filled") & trades.side.eq("buy")]
        if not trades.empty
        else pd.DataFrame(columns=["position_id"])
    )
    if sorted(filled_intents.position_id) != sorted(buys.position_id):
        raise DataValidationError("CONTINUOUS_INTENT_FILL_MISMATCH")
    _validate_position_history(result, context)
    if not actions.empty:
        policy = default_corporate_action_execution_policy()
        if (
            not actions.transition_hash.eq(context.transitions.artifact_hash).all()
            or not actions.transition_version.eq(context.transitions.artifact_version).all()
            or not actions.execution_policy_hash.eq(policy.policy_hash).all()
            or not actions.execution_policy_version.eq(policy.version).all()
        ):
            raise DataValidationError("CONTINUOUS_ACTION_IDENTITY_MISMATCH")
        transitions = {t.predecessor_ts_code: t for t in context.transitions.transition_records()}
        for row in actions.itertuples(index=False):
            transition = transitions.get(str(row.predecessor_ts_code))
            if (
                transition is None
                or not policy.classify(transition).supported
                or any(
                    getattr(row, key) != getattr(transition, key)
                    for key in (
                        "successor_ts_code",
                        "effective_date",
                        "transition_type",
                        "continuity_type",
                        "share_conversion_ratio",
                        "evidence_package_id",
                        "evidence_package_hash",
                    )
                )
            ):
                raise DataValidationError("CONTINUOUS_ACTION_EVIDENCE_MISMATCH")
        require_close(
            actions.new_shares,
            actions.old_shares * actions.share_conversion_ratio,
            "converted shares",
        )
        require_close(
            actions.pre_action_mark_value,
            actions.old_shares * actions.old_last_valid_close,
            "pre-action marks",
        )
        require_close(
            actions.post_action_mark_value,
            actions.new_shares * actions.adjusted_last_valid_close,
            "post-action marks",
        )
        require_close(actions.cash_delta, np.zeros(len(actions)), "corporate cash")


def _validate_position_history(result: BacktestResult, context: ReplayPreflight) -> None:
    """Reconcile holdings to original buys, explicit transformations and final closures."""
    trades, holdings, actions = result.trades, result.holdings, result.corporate_actions
    if trades.empty:
        if not holdings.empty or not actions.empty or not result.terminal_events.empty:
            raise DataValidationError("CONTINUOUS_POSITION_WITHOUT_TRADES")
        return
    buys = trades[trades.side.eq("buy") & trades.status.eq("filled")]
    closes = trades[trades.side.eq("sell") & trades.status.isin(["filled", "terminal_writeoff"])]
    if (
        buys.position_id.duplicated().any()
        or closes.position_id.duplicated().any()
        or set(buys.position_id) != set(closes.position_id)
    ):
        raise DataValidationError("CONTINUOUS_POSITION_CLOSURE_MISMATCH")
    held_groups = (
        {key: group for key, group in holdings.groupby("position_id")} if not holdings.empty else {}
    )
    action_groups = (
        {key: group for key, group in actions.groupby("position_id")} if not actions.empty else {}
    )
    close_map = {row.position_id: row for row in cast(Any, closes).itertuples(index=False)}
    cost_policy = ExecutionCostPolicy.from_backtest_settings(context.settings)
    for row in cast(Any, trades[trades.status.eq("filled")]).itertuples(index=False):
        if row.side not in {"buy", "sell"}:
            raise DataValidationError("CONTINUOUS_INVALID_TRADE_SIDE")
        costs = cost_policy.calculate(str(row.trade_date), row.side, float(row.gross_value))
        require_close(
            [row.commission, row.stamp_duty, row.transfer_fee, row.slippage, row.cost],
            [costs.commission, costs.stamp_duty, costs.transfer_fee, costs.slippage, costs.total],
            "effective-dated costs",
        )
        require_close(row.gross_value, row.shares * row.price, "trade notional")
    for buy in cast(Any, buys).itertuples(index=False):
        close = close_map[buy.position_id]
        expected_dates = [d for d in context.calendar if buy.trade_date <= d < close.trade_date]
        group = held_groups.get(buy.position_id, pd.DataFrame())
        if group.empty or group.trade_date.tolist() != expected_dates:
            raise DataValidationError("CONTINUOUS_HOLDING_LIFECYCLE_MISMATCH")
        target = context.calendar[
            context.calendar.index(str(buy.trade_date)) + context.settings.holding_period_days
        ]
        if (
            not group.entry_date.eq(buy.trade_date).all()
            or not group.target_exit_date.eq(target).all()
            or not group.original_entry_ts_code.eq(buy.ts_code).all()
        ):
            raise DataValidationError("CONTINUOUS_POSITION_LIFECYCLE_RESET")
        shares, code = float(buy.shares), str(buy.ts_code)
        own_actions = action_groups.get(buy.position_id, pd.DataFrame())
        pending = iter(cast(Any, own_actions).itertuples(index=False))
        event: Any = next(pending, None)
        for row in cast(Any, group).itertuples(index=False):
            while event is not None and event.effective_date <= row.trade_date:
                if code != event.predecessor_ts_code:
                    raise DataValidationError("CONTINUOUS_POSITION_CODE_MISMATCH")
                require_close(event.old_shares, shares, "action opening shares")
                shares, code = float(event.new_shares), str(event.successor_ts_code)
                event = next(pending, None)
            require_close(row.shares, shares, "holding shares")
            if row.ts_code != code:
                raise DataValidationError("CONTINUOUS_POSITION_CODE_MISMATCH")
        while event is not None and event.effective_date <= close.trade_date:
            if code != event.predecessor_ts_code:
                raise DataValidationError("CONTINUOUS_POSITION_CODE_MISMATCH")
            require_close(event.old_shares, shares, "action closing shares")
            shares, code = float(event.new_shares), str(event.successor_ts_code)
            event = next(pending, None)
        if event is not None or close.ts_code != code:
            raise DataValidationError("CONTINUOUS_POSITION_CLOSURE_MISMATCH")
        require_close(close.shares, shares, "closure shares")
    for event in cast(Any, result.terminal_events).itertuples(index=False):
        close = close_map.get(event.position_id)
        if (
            close is None
            or close.status != "terminal_writeoff"
            or close.trade_date != event.trade_date
            or close.ts_code != event.ts_code
        ):
            raise DataValidationError("CONTINUOUS_TERMINAL_LEDGER_MISMATCH")
        last = held_groups[event.position_id].iloc[-1]
        require_close(event.pre_writeoff_mark_value, last.market_value, "terminal opening mark")
        if event.entry_date != last.entry_date or event.target_exit_date != last.target_exit_date:
            raise DataValidationError("CONTINUOUS_TERMINAL_LIFECYCLE_MISMATCH")
