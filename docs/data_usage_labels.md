# Data Usage Labels

This document records raw data conditions that must be converted into explicit
labels or availability flags before feature generation, label generation, model
training, or backtesting. Do not mutate canonical raw Parquet data to encode
these rules.

## Daily OHLC Reliability

Dataset: `daily`

Condition: `high < max(open, close)`, `low > min(open, close)`, or `high < low`.

Required labels:

- `price_ohlc_valid`: `false` for affected rows.
- `research_usable`: `false` for affected rows before `20200101`; exclude them
  from research datasets.
- `tradable`: `false` for affected rows on or after `20200101`; keep the raw row
  for audit but do not use it for modelling or trading decisions.

Reason: invalid OHLC values contaminate volatility, tradability, limit-price,
and execution assumptions.

## Limit Price Special Values

Dataset: `stk_limit`

Condition: `up_limit=99999.99` and `down_limit=0`.

Required labels:

- `has_price_limit`: `false`.
- `limit_price_usable`: `false`.
- `tradability_limit_reason`: `no_price_limit_session`.

Reason: this is a Tushare sentinel for no price-limit bound, commonly BSE
listing days or other no-limit sessions. Do not treat `99999.99` or `0` as
executable limit prices.

Condition: `up_limit=0` and `down_limit=0` with a matching `suspend_d` row.

Required labels:

- `has_price_limit`: `false`.
- `limit_price_usable`: `false`.
- `tradable`: `false`.
- `tradability_limit_reason`: `suspended_no_limit_price`.

Reason: suspended stock-dates have no executable limit price and must be excluded
from the tradable universe.

## Suspension Rows

Dataset: `suspend_d`

Condition: stock-date appears in `suspend_d`.

Required labels:

- `suspended`: `true`.
- `tradable`: `false`, unless a later execution model explicitly supports a
  verified partial suspension timing rule.

Reason: price and volume must not be silently forward-filled across suspension
periods.

## Late-Published Or Revised Rows

Datasets: `top_list`, `top_inst`, `margin`, `margin_detail`, `moneyflow`,
`moneyflow_hsgt`, `cyq_chips`, `cyq_perf`, `stk_factor`.

Condition: endpoint can publish or revise rows after the first successful request
for a trade date.

Required labels:

- `data_revision_window_checked`: `true` only after the configured rolling
  backfill window has been refreshed.
- `source_available_asof`: use the ingestion/update date when this matters for
  point-in-time research.

Reason: a date existing locally does not guarantee the source has finished
publishing all rows for that date. Incremental ingestion re-fetches the most
recent five open trading days for these datasets.

## Longhubang Multiple Reasons

Datasets: `top_list`, `top_inst`

Condition: the same stock can appear on the same trade date for multiple reasons
or sides.

Required labels or keys:

- Preserve `reason` for `top_list` and do not collapse to `ts_code + trade_date`.
- Preserve `side` and `reason` for `top_inst`.
- If deriving features, aggregate with explicit rules such as reason count,
  net amount sum, or top reason category.

Reason: collapsing these rows hides multiple abnormal-trading events for the
same stock-date.

## Usage Rule

Any pipeline stage that consumes these datasets must make the labels above
explicit in its intermediate output schema. A row with an unavailable or special
raw value must never be interpreted as an ordinary executable price, valid quote,
or tradable stock-date by default.
