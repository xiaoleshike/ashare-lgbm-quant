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
