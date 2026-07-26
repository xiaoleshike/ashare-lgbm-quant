# Data Ingestion and Storage

Phase 1 implements the raw Tushare ingestion boundary. Unit tests use mocked API
responses only; real API checks must be marked with `@pytest.mark.integration`.

## Token Handling

`TUSHARE_TOKEN` is read only from the process environment by `load_settings()`.
Do not place real tokens in YAML, `.env.example`, tests, or committed files.

## Canonical Parquet Layout

Raw datasets are stored under `paths.parquet_store`, defaulting to `data/parquet`.
Reference snapshots use:

```text
data/parquet/stock_basic/snapshot=latest/data.parquet
```

Date-based datasets use month partitions:

```text
data/parquet/daily/year=2024/month=01/data.parquet
data/parquet/trade_cal/year=2024/month=01/data.parquet
```

`trade_cal` is the authoritative calendar. Daily equity endpoints are downloaded
by open trading day from `trade_cal`; `index_daily` is downloaded by configured
index codes and date range.

## Idempotency and Resume

Every write reads the existing partition, appends the newly downloaded rows,
drops duplicates by the configured primary key, sorts by key, and atomically
replaces the partition file. Re-running the same command for the same date range
therefore produces stable data instead of duplicate rows. Trade-date datasets are
written after each successful trading-day request, and date-range datasets are
chunked by calendar year before writing. If a process stops mid-run, completed
partitions remain valid and the next run resumes by merging or filling the
remaining dates.

## Validation

`ashare-quant data validate` checks required columns, primary-key uniqueness,
duplicate rows, and missing open trading days for datasets that should have data
on every open day. `suspend_d` is allowed to be empty on open days because no
suspension events may occur.

## Permission Handling

Tushare account permissions can differ by endpoint. Permission failures are
reported as clear diagnostics and the affected dataset is skipped safely; other
configured datasets can continue.

## Extended Tushare Datasets

The initial default dataset set remains conservative: stock list, trading calendar,
stock daily prices, adjustment factors, daily basic indicators, suspensions, price
limits, and selected index daily bars. This prevents an accidental `data init` from
starting a very large multi-domain download.

Additional configured datasets can be selected explicitly with repeated `--dataset`
arguments, or all configured datasets can be requested with `--all-datasets`.
Examples:

```bash
ashare-quant data init --dataset fund_basic --dataset fund_daily --start-date 20240101
ashare-quant data init --dataset income --dataset balancesheet --start-date 20200101
ashare-quant data init --all-datasets --start-date 20200101
```

Currently configured extended datasets include:

- ETF and fund data: `fund_basic`, `fund_daily`.
- Options: `opt_basic`.
- Low-frequency quotes: `weekly`, `monthly`.
- Financial statements and forecasts: `income`, `balancesheet`, `cashflow`,
  `fina_indicator`, `forecast`, `express`.
- ST and connect references: `namechange`, `hs_const`.
- Reference data: `pledge_stat`, `pledge_detail`, `share_float`, `repurchase`,
  `stk_holdertrade`, `top_list`, `top_inst`, `margin`, `margin_detail`.
- Special datasets: `concept`, `concept_detail`, `moneyflow`, `moneyflow_hsgt`,
  `broker_recommend`, `cyq_chips`, `cyq_perf`, `stk_factor`.
- Macro data: `cn_gdp`, `cn_cpi`, `cn_ppi`, `cn_m`.

Tushare permissions and fields vary by account and endpoint. Permission errors are
reported per dataset and skipped safely. If Tushare changes endpoint fields, update
`src/ashare_quant/data/datasets.py` before running large downloads.

The global request ceiling is configured by `data.rate_limit_per_minute`. Endpoints
with stricter service-specific limits use `data.endpoint_rate_limits_per_minute`;
`cyq_chips` is paced at 200 requests per minute independently of the global ceiling.


### `cyq_chips` row-limit protection

Tushare requires `ts_code` for `cyq_chips` and returns at most 6000 rows per
request. A response at that exact limit is treated as potentially truncated. The
ingestion service recursively splits the requested range at authoritative open
trading dates until every accepted response contains fewer than 6000 rows. If a
single trading day still reaches the limit, ingestion fails explicitly rather than
storing data whose completeness cannot be guaranteed.

Repair historical coverage idempotently with:

```bash
ashare-quant --config config/default.yaml data init \
  --dataset cyq_chips --start-date 20180101
```

Existing rows are merged by `(ts_code, trade_date, price)`; the repair does not
create duplicate canonical rows.

## Request Granularity

- `stock_basic` is requested for all configured listing statuses: `L`, `D`, and `P`,
  so historical universes are not built only from currently listed stocks.
- Trade-date datasets such as `daily`, `adj_factor`, `daily_basic`, `margin_detail`,
  and `moneyflow` are requested one open trading day at a time using `trade_cal`.
- Date-range reference datasets are split into calendar-year chunks to reduce
  timeout and row-limit risk.
- Financial datasets use the account's VIP endpoints when available. They are
  queried by report quarter and paginated with `limit`/`offset`; this replaces
  thousands of per-stock requests with a small number of cross-sectional calls.
- If a VIP financial endpoint is unavailable, ingestion falls back to the ordinary
  per-stock endpoint and passes `start_date`/`end_date` to the server. The fallback
  is correct but materially slower.
- Snapshot datasets without a date column are skipped during `data update` once a
  local snapshot exists; refresh them explicitly with `data init --dataset ...`.


## Data Quality Error Logs

Data ingestion commands append structured JSONL records to:

```text
logs/data_quality/YYYYMMDD/errors.jsonl
```

The log captures ingestion failures, post-ingestion validation warnings/errors,
and issues found by non-blocking cross-source checks. Raw Parquet data is not
rewritten by these checks. Any downstream cleaning must be explicit and
reproducible.

After each successful `data init` or `data update`, the CLI always runs local
validation. The optional background BaoStock comparison is controlled by
`data.run_baostock_post_ingestion_check` and is disabled by default while the
BaoStock service is unreliable. When enabled, the CLI launches
`scripts/data_checks/run_baostock_previous_day_check.py` without blocking
ingestion and logs only failures or mismatches. The comparison can still be run
manually while automatic execution is disabled.

## Raw OHLC Reliability Rule

Tushare raw `daily` rows can contain source-side OHLC inconsistencies, especially
in historical pre-BSE data mapped to current BJ codes. Keep raw data unchanged,
but apply this cleaning rule before features, labels, or backtests:

- If `trade_date < 20200101` and `high < max(open, close)`, `low > min(open, close)`,
  or `high < low`, exclude the row from research datasets.
- If `trade_date >= 20200101` and the same condition occurs, keep the raw row but
  mark it unavailable for modelling and trading logic.

This rule prevents invalid high/low values from contaminating volatility,
tradability, execution, and limit-price assumptions while preserving the canonical
raw Tushare record for audit.


## Limit Price Special Values

Keep raw `stk_limit` values unchanged, but downstream tradability logic must not
interpret special sentinel values as executable prices:

- `up_limit=99999.99` and `down_limit=0` means no price-limit bound for that
  stock-date, commonly BSE listing days or other no-limit trading sessions.
- `up_limit=0` and `down_limit=0` on rows matched by `suspend_d` means the stock
  was suspended and has no executable limit price for that date.

Feature generation and backtests should convert both cases into explicit
tradability flags instead of numeric limit prices.


## Rolling Revision Backfill

Some Tushare endpoints publish late or revise rows after the first successful
response for a trade date. Incremental updates therefore re-fetch the most recent
five open trading days for revision-prone trade-date datasets and rely on
primary-key idempotent Parquet writes to merge corrections without duplicates.

Current rolling-backfill datasets are `top_list`, `top_inst`, `margin`,
`margin_detail`, `moneyflow`, `moneyflow_hsgt`, `cyq_chips`, `cyq_perf`, and
`stk_factor`. This fixes cases where a date exists locally but Tushare later
adds additional rows for that same date.

`top_list` uses `(ts_code, trade_date, reason)` as its canonical primary key
because the same stock can appear on the same trade date for multiple
Longhubang reasons. Fully duplicated source rows are collapsed during the
idempotent Parquet merge.

## Financial Statement Batching

`income`, `balancesheet`, `cashflow`, `fina_indicator`, `forecast`, and `express`
are downloaded through their corresponding `*_vip` endpoints by report period.
Pages contain at most `data.tushare_page_size` rows (default 6000) and are written
immediately, so a stopped process can safely resume through idempotent merges.

Financial incremental updates re-fetch announcements within
`data.finance_revision_lookback_days` (default 550 days). Report periods are
enumerated for an additional 550-day lookback because annual reports and later
corrections can be announced well after the period end. This bounded revision
window avoids scanning every stock's complete history on every daily update.

Income, balance-sheet, and cash-flow keys include `f_ann_date` and `update_flag`;
forecast and express keys also include `update_flag`. Both original and updated
versions therefore remain auditable. Exact duplicate source rows are collapsed.
Downstream point-in-time joins must select the version available as of the research
date and must not backfill later revisions into earlier dates.

Stores created before `update_flag` became part of the financial primary keys may
have retained only one version of older revised reports. The rolling update repairs
the configured 550-day window. To reconstruct all historical versions, run one
idempotent financial re-initialization during a maintenance window:

```bash
ashare-quant data init \
  --dataset income --dataset balancesheet --dataset cashflow \
  --dataset fina_indicator --dataset forecast --dataset express \
  --start-date 20100101
```

Use the read-only benchmark before changing request granularity:

```bash
python scripts/experiments/tushare_batch_probe.py \
  --start-date 20260708 --end-date 20260709 \
  --finance-period 20260331 --finance-endpoint cashflow_vip
```
