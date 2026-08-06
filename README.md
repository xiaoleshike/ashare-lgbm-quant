# A-share LightGBM Quant

Production-oriented research scaffold for China A-share stock selection, feature research,
walk-forward validation, and backtesting. Phase 0 contains the package structure, typed
configuration, structured logging, CLI hooks, tooling, and smoke tests.

## Quick start

```bash
source .venv/bin/activate
make install
make test
make lint
make typecheck
```

Set `TUSHARE_TOKEN` in the process environment before running data-ingestion workflows.
Phase 0 does not download market data or train models.

## Development verification

CI runs for pushes to `main`, pull requests targeting `main`, and manual dispatch. It uses Python
3.12 and the editable `.[dev]` installation. CI requires no Tushare token, paid API, production
market data, or environment file; tests use fixtures and temporary directories.

```bash
pytest
ruff check .
ruff format --check .
mypy src
git diff --check
```

## Data ingestion

Phase 1 adds Tushare raw data ingestion and partitioned Parquet storage.
Set the token only in the process environment:

```bash
export TUSHARE_TOKEN=your-token
```

Common commands:

```bash
ashare-quant data init --start-date 20200101 --end-date 20200131
ashare-quant data update
ashare-quant data status
ashare-quant data validate
```

Use `--dataset daily` to limit a run to one dataset and `--storage-root /path/to/parquet`
to inspect or validate an alternate local store. Unit tests mock Tushare responses;
optional real API tests must use the `integration` pytest marker.

Extended data can be pulled explicitly, for example:

```bash
ashare-quant data init --dataset fund_basic --dataset fund_daily --start-date 20240101
ashare-quant data init --dataset income --dataset balancesheet --start-date 20200101
```

Use `--all-datasets` only when you intentionally want every configured Tushare
endpoint, including ETF, option, financial, macro, reference, and special datasets.

## Feature diagnostics

After rebuilding and validating universe, labels, and features, run leakage-controlled diagnostics
with explicit chronological periods:

```bash
ashare-quant --config config/default.yaml diagnostics run \
  --train-start 20100101 --train-end 20191231 \
  --validation-start 20200101 --validation-end 20221231 \
  --test-start 20230101 --test-end 20260710 --horizon 5
ashare-quant --config config/default.yaml diagnostics status
```

Reports are written under `reports/feature_diagnostics/`. See
`docs/feature_diagnostics.md` for metric definitions and the test-period isolation contract.

## Ranker baseline

Run the fixed Experiment A top-50 and Experiment B robust-subset baselines after diagnostics:

```bash
ashare-quant --config config/default.yaml models ranker-baseline
```

Artifacts are written to `models/<experiment-id>/`. These experiments report ranking diagnostics
and equal-weighted top-bucket forward returns; they are not transaction-cost or execution-aware
backtests. See `docs/ranker_baseline.md` for target and split semantics.

## Production model

After selecting the robust baseline, train the final fixed-parameter Ranker on the full approved
history:

```bash
ashare-quant --config config/default.yaml models train-production
```

The production artifact is written to `models/production/` and uses
`config/feature_sets/robust_features.json`. It does not run validation/test evaluation. See
`docs/production_model.md`.

## Executable backtest

Run the Phase 8 portfolio simulation from a saved Ranker model:

```bash
ashare-quant --config config/default.yaml backtest run \
  --model-dir models/<experiment-id> \
  --start-date 20230101 --end-date 20260710
```

Outputs are written to `backtests/<experiment-id>_backtest_<timestamp>/`. The default simulation
uses next-day open execution, 5-trading-day holding periods, Top-10/20/50 selections, commission,
stamp duty, slippage, and suspension/limit-up/limit-down constraints. See `docs/backtest.md`.
