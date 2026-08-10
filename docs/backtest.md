# Executable Portfolio Backtest

The executable backtest scores a frozen Ranker and applies the shared next-open portfolio engine.
It is measurement infrastructure, not a training, tuning, Promotion, or trading stage.

## Evidence Boundary

`backtest run` and `backtest historical` are evidence-grade. They require the exact model
`manifest.json`, resolve training and selection dates from that manifest, and require the first
evaluation date to be strictly later than the effective selection boundary. Dates and model identity
are never inferred from directory names. A missing or unsupported manifest fails with
`BACKTEST_MODEL_PROVENANCE_REQUIRED`; overlap fails with `BACKTEST_IN_SAMPLE_OVERLAP` before model
scoring or publication.

A production model can be valid for live inference while being invalid for an in-sample historical
performance backtest. Legacy models may still be loaded by explicitly diagnostic code, but those
results are not OOS evidence.

## Timing And Execution

Signals are observed after close on signal date `T`. The only supported execution rule is
`next_open`: entry is attempted at the next trading-session open and exit at the configured horizon
open. Unsupported values fail configuration validation instead of silently using next-open logic.

Entry rejects suspension, ST, missing open, and limit-up conditions. Exit rejects suspension,
missing open, and limit-down conditions. An unsellable position remains owned and retains economic
value. Exceeding `sell_delay_max_days` makes evidence-grade validation fail with
`BACKTEST_UNRESOLVED_POSITION`; it does not write the position down to zero.

## Valuation

Every open position records a deterministic `position_id`, last valid close, last valid price date,
valuation status, and stale-day count.

- `CURRENT`: use the current valid close.
- `STALE_SUSPENDED`: a known suspension has no current quote, so carry the last valid close for
  accounting only. This does not make the position tradable.
- `STALE_MISSING_DATA`: available only in diagnostic mode. Evidence-grade runs fail on unexplained
  missing or malformed prices with `BACKTEST_MARKET_DATA_INCOMPLETE`.
- Terminal write-off: allowed only when universe data explicitly proves a delisted terminal state.

The engine checks nonnegative cash/equity for the unlevered strategy, finite values, equity
reconciliation, nonnegative shares and costs, sell quantity, duplicate positions, and complete
position lifecycles. There is intentionally no universal daily-return cap.

## Execution Costs

`backtest.execution_costs` is a versioned effective-dated schedule. The authoritative rate is
resolved by trade date and side. The default schedule uses sell-side stamp duty of `0.001` before
2023-08-28 and `0.0005` from 2023-08-28. Commission, optional minimum commission, optional transfer
fee, and deterministic slippage are explicit fields. Buy trades never pay sell-side stamp duty.

The complete schedule and `cost_policy_hash` are frozen in schema-v2 backtest and executable
validation manifests. Legacy scalar settings remain readable for explicitly constructed fixed-cost
diagnostic fixtures, but new repository configuration uses the schedule.

## Metrics

- Total return: `final_equity / initial_equity - 1`.
- Benchmark total return: geometric compounding of benchmark session returns.
- Cumulative excess return: `(1 + strategy_total) / (1 + benchmark_total) - 1`.
- Annualized return: CAGR using configured annualization sessions.
- Sharpe: mean session return in excess of the configured risk-free session return, divided by sample
  standard deviation, multiplied by `sqrt(annualization_sessions)`. Default risk-free rate is zero.
- Information ratio: mean active session return divided by sample active-return standard deviation,
  annualized by the same square-root rule.
- Maximum drawdown: `equity / running_max(equity) - 1`.
- `daily_win_rate` and `trade_win_rate` are distinct.
- Holding period is measured from entry to exit in trading sessions.
- Turnover is two-way traded gross notional, buys plus sells, divided by previous equity.

Legacy aliases remain in JSON where needed, but schema-v2 fields define the units explicitly.

## Outputs

```bash
ashare-quant --config config/default.yaml backtest run \
  --model-dir models/<experiment_id> \
  --start-date YYYYMMDD \
  --end-date YYYYMMDD
```

Outputs under `backtests/<experiment_id>_backtest_<timestamp>/` include predictions, daily returns,
trades, holdings, metrics, and a manifest written last. New artifacts use backtest/accounting schema
version 2 and contain accounting diagnostics and cost-policy provenance.

## Legacy Invalidation

Invalidation is append-only and never edits the original run:

```bash
ashare-quant --config config/default.yaml backtest invalidate \
  --backtest-id BACKTEST_ID \
  --reason IN_SAMPLE_MODEL_EVALUATION \
  --reviewed-by OPERATOR \
  --note "Reviewed against immutable model training boundary"
```

Records are published under `reports/backtest_invalidations/`. Exact repeats are idempotent;
different review content creates a different immutable identity.
