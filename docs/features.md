# Point-in-Time Feature Library

The feature phase builds `features_daily`, stored under
`data/processed/features_daily/year=YYYY/month=MM/data.parquet`.

## Availability Rules

Daily market, volume, turnover, liquidity, and rank features are available only
after the close of `trade_date`. They may be used for signals that trade no
earlier than the next trading day.

Financial features are joined by announcement date: a statement row is eligible
only when `ann_date <= trade_date`. The builder does not backfill later-known
financial statements into earlier dates.

Suspended or missing price rows are not forward-filled. Rolling windows use the
current row and prior rows only.

## Feature Families

The registry currently contains 204 candidate features across these groups:

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
- market-relative and industry-relative momentum;
- cross-sectional and industry-neutral percentile ranks;
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

Industry-relative features depend on the current `universe_daily.industry`
field. If historical industry classifications change and only current industry
is available, this can introduce classification drift. Treat this as an input
data limitation until historical industry data is added.

Financial statement fields vary by Tushare endpoint and account permission. When
source columns are absent, the corresponding feature remains missing rather than
being inferred from future data.
