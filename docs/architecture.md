# Phase 0 Architecture

This project uses a `src/` layout with one production package: `ashare_quant`.
Phase 0 establishes stable module boundaries, configuration, logging, CLI entry
points, and validation tooling. It intentionally does not implement market data
downloads, feature libraries, model training, or backtesting logic.

## Package Boundaries

- `ashare_quant.config`: YAML configuration loading with typed Pydantic validation.
  Secrets such as `TUSHARE_TOKEN` are read only from environment variables.
- `ashare_quant.cli`: command entry points for local checks and future pipeline stages.
- `ashare_quant.utils`: shared infrastructure such as structured logging.
- `ashare_quant.data`: future Tushare ingestion, retries, caching, and Parquet writes.
- `ashare_quant.universe`: point-in-time universe construction and tradability filters.
- `ashare_quant.features`: feature computation and diagnostics.
- `ashare_quant.labels`: forward return and excess-return label generation.
- `ashare_quant.models`: LightGBM baselines, ranking experiments, and Optuna studies.
- `ashare_quant.validation`: chronological walk-forward validation utilities.
- `ashare_quant.backtest`: out-of-sample backtesting with realistic execution rules.
- `ashare_quant.strategy`: portfolio construction and Top-N recommendations.
- `ashare_quant.reporting`: reports, diagnostics, and experiment summaries.

## Configuration

Runtime configuration starts from `config/default.yaml` and is validated by
`AppSettings`. Non-secret settings live in YAML. Secrets never belong in YAML or
Git; set `TUSHARE_TOKEN` in the process environment. `.env.example` documents the
expected variable names without real values.

## Logging

CLI and future batch jobs should call `configure_logging()` once at process start.
The default JSON format is suitable for scheduled jobs and log aggregation.

## Phase 0 Constraints

All later implementation must preserve chronological train, validation, and test
splits; prevent look-ahead and survivorship bias; and document non-obvious
quantitative assumptions in code, configuration, or docs.


## Phase 1 Data Ingestion

The ingestion layer wraps the official Tushare client in `ashare_quant.data.tushare_client`
for retry, pacing, rate limiting, structured logging, permission diagnostics, and request
statistics. Dataset metadata lives in `ashare_quant.data.datasets`; partitioned Parquet
storage and validation are separated from API access so tests can use mocked responses.
See `docs/data_ingestion.md` for partitioning, deduplication, and resume behavior.
