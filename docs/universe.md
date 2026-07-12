# Point-in-Time Universe

Phase 2 builds `universe_daily`, a daily A-share universe and tradability table stored under `data/processed/universe_daily/year=YYYY/month=MM/data.parquet` by default.

## Base Universe

`in_base_universe` means a stock belongs to the daily historical candidate set according to `stock_basic.list_date`, `stock_basic.delist_date`, and historical appearances in `daily`. The builder uses the union of `stock_basic` and all `daily.ts_code` values, so historical stocks are not dropped merely because they are absent from a current listed-only view.

If a stock is not listed on a specific date, it remains in the daily table with `in_base_universe=false` and `exclude_reason=not_listed`.

## Model Universe

`in_model_universe` is a filtered subset for later stock selection research. The default rules exclude:

- stocks with fewer than 180 listed trading days;
- ST or `*ST` names;
- suspended stocks;
- rows without usable price data;
- rows whose trailing 20-day average amount is below `universe.min_avg_amount`;
- rows below `universe.min_price` when configured.

Liquidity windows are computed from current and prior `daily` rows only. They do not use future volume or amount data.

## Tradability Flags

Tradability is separate from membership. `is_suspended`, `is_limit_up`, `is_limit_down`, `can_buy`, and `can_sell` describe whether a realistic strategy can trade the stock on that date. A suspended stock cannot be bought or sold. Under the default execution assumptions, limit-up stocks cannot be bought and limit-down stocks cannot be sold.

These flags do not permanently remove a stock from the universe.

## Known Limitations

Tushare `stock_basic` is a snapshot. This project expects it to be downloaded with list statuses `L`, `D`, and `P`; if delisting fields are incomplete, explicit delisting is only available where `delist_date` exists. For codes found only in `daily`, the builder uses the first and last available `daily` dates as a conservative metadata fallback.

ST status currently comes from the available stock name. If historical name-change or special-treatment data is incomplete, past ST flags may be approximate. Do not treat this as a final production ST filter until historical ST events are validated.

When `trade_cal` does not cover dates before a stock's `list_date`, list age before the first available calendar date is approximated by weekdays. This prevents old stocks from being treated as brand-new at the start of a local data window, but exact listed trading-day counts require a longer authoritative calendar.

## Survivorship Bias Control

Using only today's listed stocks would remove delisted and formerly listed securities from historical backtests. This builder instead combines stock metadata with historical `daily` appearances and writes one row per `trade_date, ts_code`, including not-listed, suspended, limit-up, limit-down, and low-liquidity cases with explicit flags and reasons.
