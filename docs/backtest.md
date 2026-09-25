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

The shared engine also exposes an explicit `carry_to_calendar_end` policy for governed historical
walk-forward evidence. In that mode `sell_delay_max_days` is an alert threshold rather than a
write-off or immediate failure threshold. The position remains valued at its last valid close and
is sold at the first authoritative tradable open. The walk-forward caller supplies a separately
versioned, bounded execution tail; an open position at that cutoff still fails with
`BACKTEST_UNRESOLVED_POSITION`. Ordinary backtests and retraining executable validation retain the
default `fail_at_alert` behavior.

## Valuation

Every open position records a deterministic `position_id`, last valid close, last valid price date,
valuation status, and stale-day count.

- `CURRENT`: use the current valid close.
- `STALE_SUSPENDED`: a known suspension has no current quote, so carry the last valid close for
  accounting only. This does not make the position tradable.
- `STALE_MISSING_DATA`: available only in diagnostic mode. Evidence-grade runs fail on unexplained
  missing or malformed prices with `BACKTEST_MARKET_DATA_INCOMPLETE`.
- Terminal write-off: allowed only when universe data explicitly proves a delisted terminal state.
- Security aliases: execution prices use the same versioned canonical identity as
  universe construction before security/date joins.

Only an effective `stock_basic.delist_date` is authoritative terminal evidence. A security's last
observed market-data date is coverage metadata, not a delisting date. Missing quotes, prolonged
suspension, and exceeding the sell-delay threshold do not prove delisting and cannot authorize a
terminal write-off.

An authoritative `stock_basic.delist_date` is the first session on which the security is no longer
listed: listing eligibility therefore requires `trade_date < delist_date`. The execution loader
normalizes legacy processed snapshots to this boundary before terminal handling. This does not infer
delisting from quote absence; a terminal write-off still requires the explicit delisting date.

A missing quote is not a suspension. Only explicit point-in-time suspension evidence
permits `STALE_SUSPENDED` valuation using the last valid close. Otherwise an
evidence-grade run raises `BACKTEST_MARKET_DATA_INCOMPLETE`.

Exchange decisions that suspend a security's listing are separate from ordinary daily
`suspend_d` events. They are represented by the versioned
`config/security_identity/security_lifecycle_events.json` evidence catalog and apply only over its explicit
effective interval. Universe construction and execution-price loading share this policy. Its hash
is evidence provenance; quote absence, a long gap, or a delisting date never creates an interval.
For example, the policy records the exchange decisions for the 2019 and 2020 suspended-listing
cohorts, including `300028.SZ`, `300216.SZ`, and `300431.SZ`, through the session before each
security's delisting-board trading began or authoritative delisting became effective. Missing quotes
do not create lifecycle events.

Security-code continuity does not by itself authorize portfolio accounting. Accounting schema v3
uses the content-hashed `corporate_action_execution_policy_v1`, which supports only an
authoritatively validated `CODE_CHANGE_CONTINUITY` with `SAME_LISTED_ENTITY`, a finite positive
share ratio, unambiguous one-to-one topology, and shares-only consideration. The engine applies the
effective transition before same-day exits, preserves the position ID and entry lifecycle, adjusts
shares and the stale mark inversely, and writes an immutable `corporate_actions` ledger. It emits
no trade, cash movement, cost, or turnover for the transformation. Every other crossing, including
restructuring with an otherwise known ratio, remains `CORPORATE_ACTION_EXECUTION_UNSUPPORTED`.

The engine checks nonnegative cash/equity for the unlevered strategy, finite values, equity
reconciliation, nonnegative shares and costs, sell quantity, duplicate positions, and complete
position lifecycles. There is intentionally no universal daily-return cap.

## Execution Costs

`backtest.execution_costs` is a versioned effective-dated schedule. The authoritative rate is
resolved by trade date and side. The default schedule uses sell-side stamp duty of `0.001` before
2023-08-28 and `0.0005` from 2023-08-28. Commission, optional minimum commission, optional transfer
fee, and deterministic slippage are explicit fields. Buy trades never pay sell-side stamp duty.

The complete schedule and `cost_policy_hash` are frozen in backtest and executable-validation
manifests. Accounting-schema-v3 evidence additionally binds the identity-transition version/hash,
corporate-action policy version/hash, and content hash of each corporate-action ledger. Legacy
scalar settings remain readable for explicitly constructed fixed-cost diagnostic fixtures, but new
repository configuration uses the schedule.

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

Legacy aliases remain in JSON where needed, but current fields define the units explicitly.

## Outputs

```bash
ashare-quant --config config/default.yaml backtest run \
  --model-dir models/<experiment_id> \
  --start-date YYYYMMDD \
  --end-date YYYYMMDD
```

Outputs under `backtests/<experiment_id>_backtest_<timestamp>/` include predictions, daily returns,
trades, holdings, corporate actions, metrics, and a manifest written last. New artifacts use
accounting schema version 3 and contain accounting diagnostics plus cost, transition, and
corporate-action-policy provenance. Earlier accounting-schema-v2 artifacts remain historical and
cannot serve as current governed executable evidence.

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
Strict backtests carry an explicit identity-transition version/hash even when the reviewed
contract states that no transitions exist. A missing contract is an error. A predecessor
position crossing a known effective transition fails closed; it is never silently renamed,
sold, converted, or written off.
