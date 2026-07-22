# Executable OOS Portfolio Validation

Phase 2.6.2D3 compares the current Champion with one frozen Challenger under
identical executable portfolio rules. It consumes the Challenger's immutable,
mature final-test prediction scope and rescales neither model score. The
Champion is scored on exactly the same dates and universe keys.

## Execution Contract

Signals are available after the close of day T. Orders may execute no earlier
than the next trading-day open. Each filled position targets an exit at the
open `horizon` trading sessions after entry. Suspensions and limit prices can
reject buys or delay sells. Both models use the same calendar, execution
prices, benchmark, initial cash, and costs:

- commission from `backtest.commission` on buys and sells;
- stamp duty from `backtest.stamp_duty` on sells;
- slippage from `backtest.slippage` on buys and sells.

The existing single-account simulation deploys available cash and does not
assume leverage. Consequently, daily signals cannot create a new fully funded
vintage while earlier holdings consume all cash. This is an executable cash
constraint, not a daily overlapping-label return proxy.

## CLI

The Challenger prediction artifact must already exist. For the current 10-day
comparison run:

```bash
ashare-quant --config config/default.yaml backtest executable-validation \
  --model-id champion \
  --model-id experiment_c_h10_<id> \
  --top-n 10,20,50
```

Outputs are immutable under `reports/executable_validation/<run_id>/`:

```text
summary.json
report.md
daily_returns.parquet
trades.parquet
holdings.parquet
manifest.json
```

The report includes annual return, Sharpe ratio, maximum drawdown, turnover,
daily portfolio win rate, closed-trade win rate, and profit/loss ratio. It does
not modify either model, change the registry, promote a Challenger, or create
live orders.
