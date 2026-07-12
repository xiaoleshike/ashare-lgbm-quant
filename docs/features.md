# Point-in-Time Feature Library

The feature phase builds `features_daily`, stored under
`data/processed/features_daily/year=YYYY/month=MM/data.parquet`.

## Availability Rules

Daily market, volume, turnover, liquidity, and rank features are available only
after the close of `trade_date`. They may be used for signals that trade no
earlier than the next trading day.

Financial features use an explicit point-in-time availability date. `f_ann_date`
is preferred because corrected statements may keep an original `ann_date` but
only become knowable later; `ann_date` is used only when `f_ann_date` is missing.
A statement row is eligible only when `availability_date <= trade_date`.

When several statement records are available for the same stock and report
period, the deterministic tie-breaker is latest availability date, then latest
`end_date`, then `report_type`, `update_flag`, and `ann_date` ordering. Later
corrections affect only feature dates on or after their own `f_ann_date`.
Cross-statement ratios such as `ocf_to_profit` first align income and cash-flow
records by the same `ts_code,end_date`; the ratio becomes available only after
both components are available, using the later component availability date.

Suspended or missing price rows are not forward-filled. Rolling windows use the
current row and prior rows only.

## Feature Families

The production registry currently contains 153 point-in-time-safe candidate
features across these groups:

- returns and momentum;
- short-term reversal;
- trend strength;
- moving-average relative position;
- rolling high/low distance and drawdown;
- realized and downside volatility;
- candle and gap behavior;
- volume, turnover, amount, liquidity, and Amihud illiquidity;
- price-volume correlation;
- benchmark beta and residual volatility;
- market-relative momentum;
- market-wide cross-sectional percentile ranks;
- valuation;
- profitability, growth, balance-sheet quality, cash-flow quality, and changes.

The count is intentionally broad but avoids dense grids of nearly identical TA
parameters. Windows use short, medium, and long horizons: 1-5 days for reversal
and gaps, 5-60 days for liquidity and volatility, and 20-120 days for trend,
drawdown, beta, and market-relative behavior.

## Normalization

Raw price levels are not used as cross-stock signals. Price information is
expressed as returns, moving-average ratios, rolling range position, volatility,
beta, residual volatility, percentile ranks, or industry-relative values.

Valuation is expressed through inverse ratios and log market capitalization.
Cross-sectional rank features are added only for selected economically meaningful
base features rather than every generated column.

## Known Limitations

The first implementation uses pandas for correctness and testability. Large
full-history production runs may later be optimized with Polars or DuckDB.

`stock_basic.industry` is a current snapshot and is retained in
`universe_daily` only as descriptive display metadata. It is not treated as a
historical classification. Until a verified point-in-time industry membership
source is added, the production registry disables all 7
`industry_excess_ret_*`, 7 `cs_rank_industry_excess_ret_*`, and 28 `ind_rank_*`
features. Setting `features.enable_industry_features: true` fails configuration
validation rather than silently using the current industry snapshot.

Financial statement fields vary by Tushare endpoint and account permission. When
source columns are absent, the corresponding feature remains missing rather than
being inferred from future data.

`revenue_yoy` is defined as operating revenue year-over-year growth and is
sourced from Tushare `fina_indicator.or_yoy`. `fina_indicator.tr_yoy` represents
total operating revenue year-over-year growth and is not substituted into
`revenue_yoy`.

The local `fina_indicator` table does not contain `f_ann_date`, so later
revisions cannot be assigned a reliable revision availability date. The support
code can still compute direct `fina_indicator` fields for exploratory research,
but the production registry disables these features by default: `roe`, `roa`,
`grossprofit_margin`, `netprofit_margin`, `revenue_yoy`, `netprofit_yoy`,
`roe_delta`, `revenue_yoy_delta`, and `netprofit_yoy_delta`. `update_flag` is
not treated as an availability timestamp. Statement-derived features using
tables with `f_ann_date` remain enabled: `debt_to_assets`, `current_ratio`, and
`ocf_to_profit`.
