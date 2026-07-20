# Historical Champion Backtest

The historical engine evaluates the registered LightGBM Ranker champion with immutable same-date
features and point-in-time `in_model_universe` membership. Stocks are scored and ranked before any
future data is accessed. The existing executable simulator then applies next-session open execution,
suspension and price-limit constraints, configured costs, and a five-trading-day holding period.

Run a named chronological period:

```bash
ashare-quant --config config/default.yaml backtest historical --period 2023-2026
```

Or provide explicit dates:

```bash
ashare-quant --config config/default.yaml backtest historical \
  --start-date 20230101 --end-date 20260717
```

Outputs are written idempotently under `reports/backtest/<run_id>/`: `summary.json`,
`backtest_report.md`, `predictions.parquet`, `daily_returns.parquet`, `holdings.parquet`, and
`manifest.json`. `predictions.parquet` preserves every in-model-universe score and rank before
portfolio execution so later diagnostics are not restricted to selected holdings.

Labels are loaded only after rankings have been frozen. They validate that selected rows have future
entry and exit dates and report coverage; label returns never affect scores, candidate selection, or
portfolio returns. Annual bull, bear, and neutral labels are post-hoc reporting groups based on the
realized annual benchmark return and are never trading inputs.

The default OOS gate rejects a period beginning on or before the champion's training end date. With
the current champion trained through 2019, `2015-2020` is retained as a configured rolling boundary
but cannot be reported as a valid strategy backtest. It requires a model trained strictly before
2015. The `2020-2023` and `2023-2026` windows are OOS for that champion.

Run post-hoc alpha diagnostics only after the historical run is complete:

```bash
ashare-quant --config config/default.yaml backtest diagnostics --run-id <run_id>
```

See `docs/backtest_diagnostics.md` for metric definitions and leakage controls.
