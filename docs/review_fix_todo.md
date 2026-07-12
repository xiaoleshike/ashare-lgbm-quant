# Review Fix TODO

Last reviewed: 2026-07-12

This document records issues found during the full project review across data
ingestion, universe construction, label generation, and feature engineering.
These items should be fixed before relying on the pipeline for model training or
backtest conclusions.

## Critical

### Financial point-in-time leakage

- Location: `src/ashare_quant/features/fundamentals.py`
- Problem: financial joins use `ann_date` only and ignore `f_ann_date`,
  `end_date`, statement type, and later correction/update records. Some local
  rows have `f_ann_date` later than `ann_date`, which can expose corrected
  financial data too early.
- Impact: fundamental features may contain look-ahead bias.
- Fix: use an explicit effective availability date, preferably `f_ann_date`
  with documented fallback to `ann_date`; preserve report period and statement
  metadata; select the latest record available as of each trade date only.

### Benchmark configuration mismatch

- Location: `config/default.yaml`, `src/ashare_quant/labels/builder.py`,
  `src/ashare_quant/features/market.py`
- Problem: labels and market-relative features default to `000300.SH`, but
  `index_daily` is configured to fetch only `000001.SH`, `399001.SZ`, and
  `399006.SZ`.
- Impact: labels can become unavailable because of missing benchmark prices;
  beta, residual volatility, and market-relative momentum may be mostly missing.
- Fix: either add `000300.SH` to configured index downloads or change the default
  benchmark to a stored index. Add a config validation test.

### Historical ST and industry look-ahead

- Location: `src/ashare_quant/universe/builder.py`
- Problem: historical universe uses current `stock_basic.name` and industry for
  all historical dates. This misclassifies stocks that were ST in the past or
  whose industry/name changed.
- Impact: model universe and industry-neutral features may contain survivorship
  and classification leakage.
- Fix: use historical name/ST change data when available, build dated intervals,
  and document any residual limitation.

### Next-open label tradability uses close-day state

- Location: `src/ashare_quant/universe/tradability.py`,
  `src/ashare_quant/labels/builder.py`
- Problem: tradability flags are based on close relative to limit prices, while
  labels enter at the next trading day's open. Entry/exit feasibility should be
  evaluated against the execution price and time.
- Impact: label availability may be incorrectly filtered and can include
  information unavailable at the assumed execution time.
- Fix: separate close-limit status from open-execution tradability; label entry
  and exit checks should use entry/exit date open prices, suspension data, and
  limit prices.

### Full-history build performance is not scalable

- Location: `src/ashare_quant/universe/builder.py`,
  `src/ashare_quant/labels/builder.py`, `src/ashare_quant/features/builder.py`
- Problem: universe construction cross-joins all dates and stocks; labels loop
  row-by-row and repeatedly scan full frames; feature loading reads large raw
  datasets into memory.
- Impact: full-history production runs may be extremely slow or memory-bound.
- Fix: use DuckDB/Polars partition pruning and vectorized joins; process by date
  chunks or partitions; avoid repeated boolean scans inside loops.

## High

### Snapshot datasets do not refresh

- Location: `src/ashare_quant/data/ingestion.py`
- Problem: datasets without a date column are skipped once any snapshot exists.
  This can make `stock_basic`, concepts, macro tables, and other reference data
  stale.
- Impact: new listings, delistings, changed names, changed industries, and
  updated reference tables may be missed.
- Fix: define refresh policies per dataset, including full snapshot replacement
  or dated snapshots where appropriate.

### Incremental updates do not repair interior gaps

- Location: `src/ashare_quant/data/ingestion.py`
- Problem: updates continue from the maximum stored date and do not retry missing
  dates inside the historical range.
- Impact: network failures or partial API responses can leave permanent gaps.
- Fix: maintain per-dataset completeness manifests and schedule gap repair
  before or after normal incremental update.

### Validation failures do not fail the CLI reliably

- Location: `src/ashare_quant/cli/__init__.py`,
  `src/ashare_quant/data/validation.py`
- Problem: validation issues can be logged while the CLI still exits
  successfully; empty datasets can be treated as warnings.
- Impact: scheduled jobs may look successful even when data quality failed.
- Fix: make critical validation failures return non-zero exit status and write
  dated error logs.

### Suspensions distort rolling features

- Location: `src/ashare_quant/features/market.py`,
  `src/ashare_quant/features/builder.py`
- Problem: rolling features are computed on available quote rows instead of an
  authoritative trading-calendar grid. Resume-day returns can span long
  suspension periods.
- Impact: momentum, volatility, turnover, liquidity, and price-volume features
  can be economically misleading.
- Fix: compute features on a trade-date grid joined to universe/tradability; make
  suspension gaps explicit and avoid silent forward-fill.

### Cross-sectional ranks include ineligible stocks

- Location: `src/ashare_quant/features/builder.py`
- Problem: rank features are computed over all rows with data, not the configured
  model/base universe.
- Impact: ST, new, suspended, illiquid, or otherwise ineligible stocks can distort
  percentile features used by eligible stocks.
- Fix: rank over a configurable eligible universe mask and document the chosen
  cross-section.

### Downside volatility is mostly missing

- Location: `src/ashare_quant/features/market.py`
- Problem: downside volatility masks non-negative returns to null before rolling
  standard deviation, then requires too many negative observations.
- Impact: a useful risk feature is mostly unavailable.
- Fix: implement downside semideviation with non-negative returns contributing
  zero downside deviation, or lower/document the minimum negative observation
  rule.

### Label tail validation is incomplete

- Location: `src/ashare_quant/labels/validation.py`
- Problem: end-of-data checks do not reliably detect labels generated too close
  to the latest available price date and do not compare expected row counts
  against eligible base-universe rows.
- Impact: missing or invalid labels can pass validation.
- Fix: validate per horizon using calendar-derived required exit dates and compare
  expected/actual label rows by date.

## Medium

### Feature builder ignores date pruning during raw input loading

- Location: `src/ashare_quant/features/builder.py`
- Problem: raw data is loaded broadly before filtering.
- Impact: unnecessary memory and runtime cost on full-history builds.
- Fix: push start/end filters into storage reads or DuckDB queries.

### Status commands can scan too much data

- Location: data, universe, labels, and features CLI status paths
- Problem: status commands read large Parquet datasets into pandas for simple
  counts or date summaries.
- Impact: slow status checks, especially after moving raw data to mechanical
  storage.
- Fix: use DuckDB metadata/count queries with partition pruning.

### Feature rows are based on price availability, not universe dates

- Location: `src/ashare_quant/features/builder.py`
- Problem: suspended dates are absent instead of represented as explicit
  unavailable feature rows.
- Impact: downstream joins may silently drop observations.
- Fix: build feature output from universe/date grid, then attach available market
  data.

### Feature registry taxonomy needs tightening

- Location: `src/ashare_quant/features/registry.py`
- Problem: some families are misclassified or too broad, and metadata does not
  fully cover earnings quality and fundamental change assumptions.
- Impact: diagnostics and feature selection by family become less reliable.
- Fix: audit family assignments and ensure every feature has correct availability
  and economic rationale metadata.

### Documentation is stale

- Location: `README.md`, `docs/architecture.md`
- Problem: some docs still describe early phases and do not reflect current data,
  universe, labels, and feature behavior.
- Impact: operators can run stale commands or misunderstand limitations.
- Fix: update docs after the critical/high fixes are implemented.

### Dependency reproducibility is weak

- Location: `pyproject.toml`
- Problem: dependencies use broad minimum versions without a lock file.
- Impact: pandas, Polars, DuckDB, or Tushare behavior may drift across installs.
- Fix: add a lock workflow or document supported pinned environment versions.

## Operational Follow-ups

- Confirm the long-running `cyq_chips` ingestion process completes cleanly before
  changing ingestion scheduling.
- Re-run `make test`, `make lint`, and `make typecheck` after each repair batch.
- After benchmark and financial PIT fixes, rebuild affected labels/features
  before running any model experiments.
- Do not start labels, features, modelling, or backtesting conclusions from
  outputs produced before these fixes are applied.
