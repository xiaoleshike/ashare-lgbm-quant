# Executable Forward-Return Labels

Phase 3 builds `labels_forward`, stored by default under
`data/processed/labels_forward/year=YYYY/month=MM/data.parquet`.

## Signal and Execution Timing

Signals are assumed to be generated after the market close on trade date `T`.
The default entry is therefore the next trading day's open, not `close[T]`.
Using `close[T]` to `close[T+H]` would assume the strategy can trade at a price
that was already known when the signal was produced, which is unrealistic for
after-close stock selection.

For horizon `H`:

- `entry_date` is the next open trading day after `T`;
- `exit_date` is `H` open trading days after `entry_date`;
- `stock_forward_ret_H = adjusted_exit_open / adjusted_entry_open - 1`;
- `future_excess_ret_H = stock_forward_ret_H - benchmark_forward_ret_H`.

## Adjusted Prices

Stock labels use adjusted open prices computed as:

`adjusted_open = daily.open * adj_factor`

This is a back-adjusted style ratio from Tushare adjustment factors. The label
never mixes adjusted close with unadjusted open. If either `daily.open` or
`adj_factor` is missing at entry or exit, the row is marked unavailable.

Benchmark returns use the configured index open price from `index_daily`. The
default benchmark is `000300.SH`. Index prices are not adjusted by `adj_factor`;
changing the benchmark changes the excess-return target.

## Tradability Constraints

Labels are only considered for stocks that are in the base universe on signal
date `T`. Tradability is evaluated on the actual entry and exit dates using the
daily universe table:

- by default, if entry `can_buy=false`, the label is unavailable;
- by default, if exit `can_sell=false`, the label is unavailable;
- limit-up entry and limit-down exit may be explicitly allowed in configuration;
- delayed exit is off by default and must be explicitly enabled.

Prices are never forward-filled across suspended or missing trading days.

## Ranking Labels

For each `trade_date` and `horizon`, available `future_excess_ret` values are
ranked cross-sectionally. `future_rank_pct` is the percentile rank in `[0, 1]`.
`future_quantile` maps that rank into `labels.quantile_buckets` buckets, with
`0` as the worst group and `bucket_count - 1` as the best group.

## Known Limitations

Target construction uses future prices by design; these columns must only be
merged as training labels, never as features. Entry and exit tradability also
use future dates relative to signal time and must not be exposed to feature
generation.

Benchmark data availability can make otherwise valid stock labels unavailable.
For production research, benchmark choice and index data quality should be
validated before comparing strategies.
