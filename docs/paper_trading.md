# Paper Trading

The paper-trading layer converts immutable production rankings into isolated virtual
accounts. It never connects to a broker and never creates real orders.

## Accounts and Ledgers

Configuration under `paper_trading.portfolios` defines four independent accounts:
`champion_top20`, `h5_top20`, `h10_top20`, and `ensemble_top20`. Each account stores
`account.json`, `orders.parquet`, `trades.parquet`, `positions.parquet`, and
`equity_curve.parquet` under `paper_trading/<portfolio_id>/`.

Ledger rows are logically append-only. Each row has a deterministic identity, and a
repeated command either appends no rows or fails if the same identity has different
content. Publication uses a temporary Parquet file followed by atomic replacement.
Cash and positions are reconstructed by replaying prior trade and position events.

## Timing and Execution

A signal produced after close on session T creates target-weight orders for the next
open trading session. Execution uses only that session's `daily.open`, `stk_limit`,
and universe suspension/ST state. Buys at the upper limit, sells at the lower limit,
and suspended executions are rejected. Shares are rounded down to 100-share lots.
Commission and slippage apply to both sides; stamp duty applies only to sells.

Orders store the immutable rule `next_open`, not a future calendar date. On each
completed session, execution selects orders from the immediately preceding open
session. This preserves T+1 semantics without requiring `trade_cal` to contain
future sessions.

`champion_top20` consumes the published production `candidates.csv`, including its
configured A-share eligibility filters. Challenger accounts score the same-date
production feature and model universe with their own immutable model artifacts.
The ensemble averages daily percentile ranks, never raw LightGBM scores.

Last known prices may be retained for NAV marking when a current close is absent.
They are never used as an execution price.

## Commands

```bash
ashare-quant --config config/default.yaml paper-trading init
ashare-quant --config config/default.yaml paper-trading rebalance --as-of 20260724
ashare-quant --config config/default.yaml paper-trading execute --as-of 20260727
ashare-quant --config config/default.yaml paper-trading report --as-of 20260727
```

The production pipeline calls the same service API after publishing its production
summary:

```bash
ashare-quant --config config/default.yaml pipeline production --as-of 20260724
```

Daily reports are written to `reports/paper_trading_daily/YYYYMMDD/`. Orders whose
execution session has not arrived remain pending. Rejected orders are audit records;
the next production signal may create a new target order for a later session.

## Replay and Safety

All accounts use `environment=paper`, `broker_connected=false`, and
`real_orders_generated=false`. Source signal hashes connect orders, executions, and
equity events to immutable model and production artifacts. To replay independently,
use a separate `paths.paper_trading` root with the same configuration and source
artifacts.
