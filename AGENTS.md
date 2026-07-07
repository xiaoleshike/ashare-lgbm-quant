You are working on an A-share quantitative stock selection research and backtesting system.

## Project objective

Build a production-quality quantitative research pipeline for China A-shares using:

* Tushare Pro for market and fundamental data
* Parquet as the canonical data storage format
* DuckDB for analytical querying
* Polars for large-scale feature computation where practical
* Pandas only where ecosystem compatibility makes it appropriate
* LightGBM for cross-sectional stock return prediction and ranking
* Optuna for controlled hyperparameter optimization
* pytest for testing
* Ruff for linting
* mypy for type checking

The system must support:

1. Historical Tushare data download with retries, rate limiting, local caching, idempotency, and resumability.
2. Daily incremental data update.
3. Point-in-time-correct universe construction.
4. Feature generation.
5. Forward return and excess-return label generation.
6. Feature diagnostics and feature selection.
7. LightGBM regression baseline.
8. LightGBM ranking model experiment.
9. Expanding-window or rolling-window walk-forward validation.
10. Out-of-sample backtesting.
11. Transaction costs, slippage, suspension, ST filtering, limit-up and limit-down tradability constraints.
12. Daily stock scoring and Top-N recommendation output.
13. Reproducible experiment configuration.
14. Unit tests and data validation tests.

## Critical quantitative research rules

Never introduce look-ahead bias.

Never use future information in a feature.

Financial statement data must only become available according to an explicit publication or announcement date. Do not backfill later-known financial data into earlier dates.

Do not build historical universes using only the currently listed stock universe.

Avoid survivorship bias.

All train, validation, and test splits must be chronological. Random train_test_split with shuffled time-series rows is prohibited.

Preprocessing, winsorization parameters, normalization parameters, feature selection, and hyperparameter optimization must be fitted only on training data and then applied forward.

Backtests must use signals available at time t and trades executable no earlier than the configured execution time, normally next trading day's open or a realistic VWAP proxy.

Do not assume a limit-up stock can always be bought or a limit-down stock can always be sold.

Do not silently forward-fill price or volume data across suspension periods.

All backtest returns presented as strategy results must be out-of-sample.

## Modelling philosophy

Do not assume more features automatically improve accuracy.

Create a broad candidate feature library, then evaluate feature quality using:

* cross-sectional IC
* Rank IC
* IC mean
* IC standard deviation
* ICIR
* year-by-year IC stability
* market-regime stability
* feature coverage
* turnover characteristics
* pairwise feature correlation
* model importance
* permutation importance where computationally practical
* out-of-sample ablation tests

Start with approximately 150 to 220 candidate features.

The final production model should select a substantially smaller robust subset based on evidence, not a fixed arbitrary feature count.

Do not select features using the final test period.

## Engineering requirements

Use a src-layout Python package.

Prefer small cohesive modules over very large files.

Use type hints for public functions.

Every important public module and non-obvious quantitative assumption must be documented.

Avoid notebooks as the core production workflow. Notebooks may be used for exploration only.

All pipeline stages must be runnable from CLI commands.

Each task is considered complete only when:

* implementation is complete
* relevant unit tests pass
* integration tests pass where applicable
* lint checks pass
* existing functionality is not broken
* README or architecture documentation is updated when behavior changes

Before making a large architectural change:

1. inspect the current repository,
2. explain the proposed change,
3. identify risks,
4. implement incrementally,
5. run tests,
6. report exact commands and results.

When requirements are ambiguous, prefer scientifically conservative choices and record assumptions explicitly in documentation or configuration.
