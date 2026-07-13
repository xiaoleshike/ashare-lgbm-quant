"""Daily market, liquidity, risk, and rank features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ashare_quant.config.settings import FeatureSettings
from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame


def build_market_features(
    daily: DataFrame,
    adj_factor: DataFrame,
    daily_basic: DataFrame,
    index_daily: DataFrame,
    trade_cal: DataFrame,
    universe: DataFrame,
    settings: FeatureSettings,
) -> DataFrame:
    """Build point-in-time daily market features."""

    prices = prepare_price_frame(daily, adj_factor)
    if prices.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    prices = align_to_trading_calendar(prices, universe, trade_cal)
    prices = add_benchmark_returns(prices, index_daily, settings.benchmark_index_code)
    prices = add_daily_basic(prices, daily_basic)
    prices = prices.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return add_market_features_polars(prices, settings)


def add_market_features_polars(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Add market-derived feature columns using Polars grouped rolling operations."""

    if frame.empty:
        return frame
    working = pl.from_pandas(frame).sort(["ts_code", "trade_date"])
    for column in (
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "vol",
        "amount",
        "benchmark_ret_1d",
        "turnover_rate",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "dv_ttm",
        "total_mv",
        "circ_mv",
    ):
        if column not in working.columns:
            working = working.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
        else:
            working = working.with_columns(pl.col(column).cast(pl.Float64, strict=False))
    if "_is_traded_observation" not in working.columns:
        working = working.with_columns(pl.lit(True).alias("_is_traded_observation"))
    else:
        working = working.with_columns(pl.col("_is_traded_observation").cast(pl.Boolean))
    if "in_model_universe" not in working.columns:
        working = working.with_columns(pl.lit(True).alias("in_model_universe"))
    else:
        working = working.with_columns(
            pl.col("in_model_universe").fill_null(False).cast(pl.Boolean)
        )

    working = working.with_columns(
        [
            (
                pl.when(
                    pl.col("_is_traded_observation")
                    & pl.col("_is_traded_observation").shift(1).over("ts_code")
                )
                .then(pl.col("adj_close") / pl.col("adj_close").shift(1).over("ts_code") - 1.0)
                .otherwise(None)
            ).alias("ret_1d"),
            (pl.col("adj_close") / pl.col("adj_open") - 1.0).alias("intraday_ret"),
            (pl.col("adj_close") - pl.col("adj_open")).alias("_body"),
            (pl.col("adj_high") - pl.col("adj_low")).alias("_intraday_range"),
            pl.col("adj_close").shift(1).over("ts_code").alias("_prev_close"),
        ]
    )
    working = working.with_columns(
        [
            (pl.col("ret_1d") + 1.0).log().alias("logret_1d"),
            (pl.col("_body") / zero_to_null(pl.col("_intraday_range"))).alias("candle_body_pct"),
            (
                (pl.col("adj_high") - pl.max_horizontal("adj_open", "adj_close"))
                / zero_to_null(pl.col("_intraday_range"))
            ).alias("upper_shadow_pct"),
            (
                (pl.min_horizontal("adj_open", "adj_close") - pl.col("adj_low"))
                / zero_to_null(pl.col("_intraday_range"))
            ).alias("lower_shadow_pct"),
            (
                (pl.col("adj_close") - pl.col("adj_low"))
                / zero_to_null(pl.col("_intraday_range"))
            ).alias("close_location_value"),
            (pl.col("adj_open") / zero_to_null(pl.col("_prev_close")) - 1.0).alias(
                "gap_open_ret"
            ),
            (pl.col("ret_1d").abs() / zero_to_null(pl.col("amount"))).alias("amihud_raw"),
        ]
    )
    working = working.with_columns(pl.col("gap_open_ret").abs().alias("gap_abs"))

    return_windows = tuple(dict.fromkeys((*settings.return_windows, *settings.short_windows)))
    for window in return_windows:
        minp = min_periods(window, settings)
        expressions: list[pl.Expr] = []
        if window != 1:
            rolling_logret = (
                pl.col("logret_1d")
                .rolling_sum(window_size=window, min_samples=minp)
                .over("ts_code")
            )
            expressions.extend(
                [
                    (rolling_logret.exp() - 1.0).alias(f"ret_{window}d"),
                    rolling_logret.alias(f"logret_sum_{window}d"),
                ]
            )
        else:
            expressions.append(pl.col("logret_1d").alias("logret_sum_1d"))
        working = working.with_columns(expressions)
        if window in settings.return_windows:
            working = working.with_columns(
                (
                    pl.col(f"ret_{window}d")
                    - (
                        (pl.col("benchmark_ret_1d") + 1.0)
                            .log()
                            .rolling_sum(window_size=window, min_samples=minp)
                            .over("ts_code")
                            .exp()
                        - 1.0
                    )
                ).alias(f"market_excess_ret_{window}d")
            )

    for window in settings.short_windows:
        expressions = [
            pl.col("gap_open_ret")
            .rolling_mean(window_size=window, min_samples=1)
            .over("ts_code")
            .alias(f"gap_mean_{window}d")
        ]
        ret_name = f"ret_{window}d"
        if ret_name in working.columns:
            expressions.append((-pl.col(ret_name)).alias(f"reversal_ret_{window}d"))
        working = working.with_columns(expressions)

    for window in settings.medium_windows:
        minp = min_periods(window, settings)
        ma = pl.col("adj_close").rolling_mean(window_size=window, min_samples=minp).over("ts_code")
        std = pl.col("adj_close").rolling_std(window_size=window, min_samples=minp).over("ts_code")
        amount_mean = (
            pl.col("amount").rolling_mean(window_size=window, min_samples=minp).over("ts_code")
        )
        downside_squared = downside_squared_expr(settings)
        working = working.with_columns(
            [
                (pl.col("adj_close") / zero_to_null(ma) - 1.0).alias(f"ma_ratio_{window}d"),
                ((pl.col("adj_close") - ma) / zero_to_null(std)).alias(f"ma_z_{window}d"),
                (
                    ma / zero_to_null(pl.col("adj_close").shift(window).over("ts_code")) - 1.0
                ).alias(f"trend_slope_{window}d"),
                (pl.col("ret_1d") > 0)
                .cast(pl.Float64)
                .rolling_mean(window_size=window, min_samples=minp)
                .over("ts_code")
                .alias(f"positive_ret_ratio_{window}d"),
                pl.col("ret_1d")
                .rolling_std(window_size=window, min_samples=minp)
                .over("ts_code")
                .alias(f"realized_vol_{window}d"),
                downside_squared
                .rolling_mean(window_size=window, min_samples=minp)
                .over("ts_code")
                .sqrt()
                .alias(f"downside_vol_{window}d"),
                (pl.col("vol") / zero_to_null(
                    pl.col("vol").rolling_mean(window_size=window, min_samples=minp).over("ts_code")
                )).alias(f"volume_ratio_{window}d"),
                (pl.col("amount") / zero_to_null(amount_mean)).alias(
                    f"amount_ratio_{window}d"
                ),
                (pl.col("turnover_rate") / zero_to_null(
                    pl.col("turnover_rate")
                    .rolling_mean(window_size=window, min_samples=minp)
                    .over("ts_code")
                )).alias(f"turnover_ratio_{window}d"),
                (
                    pl.col("amount")
                    .rolling_std(window_size=window, min_samples=minp)
                    .over("ts_code")
                    / zero_to_null(amount_mean)
                ).alias(f"amount_cv_{window}d"),
            ]
        )

    for window in settings.long_windows:
        minp = min_periods(window, settings)
        high = pl.col("adj_close").rolling_max(window_size=window, min_samples=minp).over("ts_code")
        low = pl.col("adj_close").rolling_min(window_size=window, min_samples=minp).over("ts_code")
        range_width = high - low
        cov_ret_bench = rolling_cov_expr("ret_1d", "benchmark_ret_1d", window, minp)
        var_bench = (
            pl.col("benchmark_ret_1d")
            .rolling_var(window_size=window, min_samples=minp)
            .over("ts_code")
        )
        beta = cov_ret_bench / zero_to_null(var_bench)
        residual = pl.col("ret_1d") - beta * pl.col("benchmark_ret_1d")
        drawdown = pl.col("adj_close") / zero_to_null(high) - 1.0
        working = working.with_columns(
            [
                (pl.col("adj_close") / zero_to_null(high) - 1.0).alias(f"dist_high_{window}d"),
                (pl.col("adj_close") / zero_to_null(low) - 1.0).alias(f"dist_low_{window}d"),
                ((pl.col("adj_close") - low) / zero_to_null(range_width)).alias(
                    f"range_pos_{window}d"
                ),
                drawdown.alias(f"drawdown_{window}d"),
                drawdown
                .rolling_min(window_size=window, min_samples=minp)
                .over("ts_code")
                .alias(f"max_drawdown_{window}d"),
                pl.col("amihud_raw")
                .rolling_mean(window_size=window, min_samples=minp)
                .over("ts_code")
                .alias(f"amihud_{window}d"),
                rolling_corr_expr("ret_1d", "amount", window, minp).alias(
                    f"ret_amount_corr_{window}d"
                ),
                rolling_corr_expr("ret_1d", "vol", window, minp).alias(
                    f"ret_volume_corr_{window}d"
                ),
                beta.alias(f"beta_{window}d"),
                residual.rolling_std(window_size=window, min_samples=minp)
                .over("ts_code")
                .alias(f"residual_vol_{window}d"),
            ]
        )

    working = working.with_columns(
        [
            inverse_expr("pe").alias("earnings_yield"),
            inverse_expr("pb").alias("book_to_market"),
            inverse_expr("ps").alias("sales_yield"),
            inverse_expr("pe_ttm").alias("pe_ttm_inv"),
            inverse_expr("ps_ttm").alias("ps_ttm_inv"),
            safe_log_expr("total_mv").alias("ln_total_mv"),
            safe_log_expr("circ_mv").alias("ln_circ_mv"),
        ]
    )

    from ashare_quant.features.registry import FEATURE_REGISTRY

    for spec in FEATURE_REGISTRY:
        if spec.family == "cross_sectional_percentile_ranks":
            base = spec.required_source_columns[0]
            if base in working.columns:
                working = working.with_columns(
                    eligible_percent_rank_expr(base, ["trade_date"]).alias(spec.name)
                )

    return working.drop(
        ["_body", "_intraday_range", "_prev_close", "_is_traded_observation"],
        strict=False,
    ).to_pandas()


def zero_to_null(expr: pl.Expr) -> pl.Expr:
    """Convert zero denominators to null before division."""

    return pl.when(expr == 0).then(None).otherwise(expr)


def downside_squared_expr(settings: FeatureSettings) -> pl.Expr:
    """Return squared downside return using MAR, preserving missing returns."""

    downside_return = pl.min_horizontal(pl.col("ret_1d") - settings.downside_mar, pl.lit(0.0))
    return pl.when(pl.col("ret_1d").is_not_null()).then(downside_return.pow(2)).otherwise(None)


def rolling_cov_expr(left: str, right: str, window: int, minimum_periods: int) -> pl.Expr:
    """Return grouped rolling covariance expression."""

    left_expr = pl.col(left)
    right_expr = pl.col(right)
    mean_left = left_expr.rolling_mean(window_size=window, min_samples=minimum_periods).over(
        "ts_code"
    )
    mean_right = right_expr.rolling_mean(window_size=window, min_samples=minimum_periods).over(
        "ts_code"
    )
    mean_product = (left_expr * right_expr).rolling_mean(
        window_size=window, min_samples=minimum_periods
    ).over("ts_code")
    count = (
        (left_expr.is_not_null() & right_expr.is_not_null())
        .cast(pl.Float64)
        .rolling_sum(window_size=window, min_samples=1)
        .over("ts_code")
    )
    return (mean_product - mean_left * mean_right) * count / (count - 1.0)


def rolling_corr_expr(left: str, right: str, window: int, minimum_periods: int) -> pl.Expr:
    """Return grouped rolling correlation expression."""

    cov = rolling_cov_expr(left, right, window, minimum_periods)
    left_std = pl.col(left).rolling_std(window_size=window, min_samples=minimum_periods).over(
        "ts_code"
    )
    right_std = pl.col(right).rolling_std(window_size=window, min_samples=minimum_periods).over(
        "ts_code"
    )
    return cov / zero_to_null(left_std * right_std)


def inverse_expr(column: str) -> pl.Expr:
    """Return inverse of a positive numeric column."""

    return pl.when(pl.col(column) > 0).then(1.0 / pl.col(column)).otherwise(None)


def safe_log_expr(column: str) -> pl.Expr:
    """Return log of positive values only."""

    return pl.when(pl.col(column) > 0).then(pl.col(column).log()).otherwise(None)


def percent_rank_expr(column: str, over: list[str]) -> pl.Expr:
    """Return Pandas-compatible percentile rank within a group."""

    count = pl.col(column).count().over(over)
    return pl.when(pl.col(column).is_not_null()).then(pl.col(column).rank().over(over) / count)


def eligible_percent_rank_expr(column: str, over: list[str]) -> pl.Expr:
    """Return percentile rank using only same-date model-universe rows."""

    eligible_value = pl.when(pl.col("in_model_universe")).then(pl.col(column)).otherwise(None)
    count = eligible_value.count().over(over)
    rank = eligible_value.rank(method="average").over(over) / count
    return pl.when(pl.col("in_model_universe") & pl.col(column).is_not_null()).then(rank)


def prepare_price_frame(daily: DataFrame, adj_factor: DataFrame) -> DataFrame:
    """Return daily prices with adjusted open/close and no forward filling."""

    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"}
    required_adj = {"ts_code", "trade_date", "adj_factor"}
    if (
        daily.empty
        or adj_factor.empty
        or not required_daily.issubset(daily.columns)
        or not required_adj.issubset(adj_factor.columns)
    ):
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    daily_work = daily[list(required_daily)].copy()
    daily_work["ts_code"] = daily_work["ts_code"].astype(str)
    daily_work["trade_date"] = daily_work["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "vol", "amount"):
        daily_work[column] = pd.to_numeric(daily_work[column], errors="coerce")
    adj_work = adj_factor[["ts_code", "trade_date", "adj_factor"]].copy()
    adj_work["ts_code"] = adj_work["ts_code"].astype(str)
    adj_work["trade_date"] = adj_work["trade_date"].astype(str)
    adj_work["adj_factor"] = pd.to_numeric(adj_work["adj_factor"], errors="coerce")
    frame = daily_work.merge(adj_work, on=["ts_code", "trade_date"], how="left")
    for column in ("open", "high", "low", "close"):
        frame[f"adj_{column}"] = frame[column] * frame["adj_factor"]
    frame["_is_traded_observation"] = True
    return frame.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")


def align_to_trading_calendar(
    prices: DataFrame,
    universe: DataFrame,
    trade_cal: DataFrame,
) -> DataFrame:
    """Represent every universe stock on every open trading day without forward-filling data.

    Rolling features use this calendar-aligned frame so a suspension day with no
    `daily` row consumes one trading-day slot in the window. Price, return,
    amount, volume, and turnover observations remain missing on those days.
    """

    open_dates = open_trade_dates(trade_cal)
    if not open_dates:
        raise DataValidationError("trade_cal with open trading dates is required for features")
    if universe.empty or not {"trade_date", "ts_code"}.issubset(universe.columns):
        raise DataValidationError("universe with trade_date and ts_code is required for features")

    context = normalize_universe_context(universe)
    context = context[context["trade_date"].isin(open_dates)].copy()
    if context.empty:
        return pd.DataFrame(columns=[*prices.columns, "industry", "in_model_universe"])
    context = context.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")

    price_columns = [column for column in prices.columns if column not in {"industry"}]
    merged = context.merge(
        prices[price_columns],
        on=["trade_date", "ts_code"],
        how="left",
        suffixes=("", "_price"),
    )
    merged["_is_traded_observation"] = merged["_is_traded_observation"].fillna(False).astype(bool)
    return merged


def open_trade_dates(trade_cal: DataFrame) -> list[str]:
    """Return authoritative open dates from trade_cal."""

    if trade_cal.empty or not {"cal_date", "is_open"}.issubset(trade_cal.columns):
        return []
    calendar = trade_cal[["cal_date", "is_open"]].copy()
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="coerce").fillna(0).astype(int)
    return sorted(calendar.loc[calendar["is_open"] == 1, "cal_date"].drop_duplicates().tolist())


def normalize_universe_context(universe: DataFrame) -> DataFrame:
    """Return universe context used for calendar alignment."""

    keep = [
        column
        for column in ("trade_date", "ts_code", "industry", "is_suspended", "in_model_universe")
        if column in universe.columns
    ]
    context = universe[keep].copy()
    context["trade_date"] = context["trade_date"].astype(str)
    context["ts_code"] = context["ts_code"].astype(str)
    if "industry" not in context.columns:
        context["industry"] = ""
    if "is_suspended" not in context.columns:
        context["is_suspended"] = False
    if "in_model_universe" not in context.columns:
        context["in_model_universe"] = True
    context["industry"] = context["industry"].fillna("").astype(str)
    context["is_suspended"] = context["is_suspended"].fillna(False).astype(bool)
    context["in_model_universe"] = context["in_model_universe"].fillna(False).astype(bool)
    return context


def add_benchmark_returns(
    frame: DataFrame, index_daily: DataFrame, benchmark_index_code: str
) -> DataFrame:
    """Merge benchmark daily returns available after close."""

    working = frame.copy()
    if index_daily.empty or not {"ts_code", "trade_date", "close"}.issubset(index_daily.columns):
        raise DataValidationError(
            "index_daily with ts_code/trade_date/close is required for market benchmark features"
        )
    benchmark = index_daily[index_daily["ts_code"].astype(str) == benchmark_index_code].copy()
    if benchmark.empty:
        available = sorted(index_daily["ts_code"].dropna().astype(str).unique().tolist())
        raise DataValidationError(
            f"benchmark index {benchmark_index_code} is missing from index_daily; "
            f"available_index_codes={available}"
        )
    benchmark["trade_date"] = benchmark["trade_date"].astype(str)
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.sort_values("trade_date")
    benchmark["benchmark_ret_1d"] = benchmark["close"].pct_change()
    return working.merge(benchmark[["trade_date", "benchmark_ret_1d"]], on="trade_date", how="left")


def add_universe_context(frame: DataFrame, universe: DataFrame) -> DataFrame:
    """Merge industry and suspension context from the point-in-time universe."""

    working = frame.copy()
    if universe.empty or not {"trade_date", "ts_code"}.issubset(universe.columns):
        working["industry"] = ""
        working["is_suspended"] = False
        return working
    keep = [
        column
        for column in ("trade_date", "ts_code", "industry", "is_suspended", "in_model_universe")
        if column in universe.columns
    ]
    context = universe[keep].copy()
    context["trade_date"] = context["trade_date"].astype(str)
    context["ts_code"] = context["ts_code"].astype(str)
    if "industry" not in context.columns:
        context["industry"] = ""
    if "is_suspended" not in context.columns:
        context["is_suspended"] = False
    if "in_model_universe" not in context.columns:
        context["in_model_universe"] = True
    context = context.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    merged = working.merge(context, on=["trade_date", "ts_code"], how="left")
    merged["industry"] = merged["industry"].fillna("").astype(str)
    merged["is_suspended"] = merged["is_suspended"].fillna(False).astype(bool)
    merged["in_model_universe"] = merged["in_model_universe"].fillna(False).astype(bool)
    return merged


def add_daily_basic(frame: DataFrame, daily_basic: DataFrame) -> DataFrame:
    """Merge daily_basic values without using future rows."""

    working = frame.copy()
    if daily_basic.empty or not {"trade_date", "ts_code"}.issubset(daily_basic.columns):
        return working
    basic = daily_basic.copy()
    basic["trade_date"] = basic["trade_date"].astype(str)
    basic["ts_code"] = basic["ts_code"].astype(str)
    columns = [
        column
        for column in (
            "trade_date",
            "ts_code",
            "turnover_rate",
            "turnover_rate_f",
            "pe",
            "pe_ttm",
            "pb",
            "ps",
            "ps_ttm",
            "dv_ttm",
            "total_mv",
            "circ_mv",
        )
        if column in basic.columns
    ]
    basic = basic[columns].drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    for column in columns:
        if column not in {"trade_date", "ts_code"}:
            basic[column] = pd.to_numeric(basic[column], errors="coerce")
    return working.merge(basic, on=["trade_date", "ts_code"], how="left")


def add_return_features(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Add returns, momentum, reversal, and market-relative returns."""

    working = frame.copy()
    grouped = working.groupby("ts_code", sort=False)
    working["ret_1d"] = grouped["adj_close"].pct_change()
    working["logret_1d"] = np.log1p(working["ret_1d"])
    return_windows = tuple(dict.fromkeys((*settings.return_windows, *settings.short_windows)))
    for window in return_windows:
        if window != 1:
            working[f"ret_{window}d"] = grouped["adj_close"].pct_change(window)
            working[f"logret_sum_{window}d"] = grouped["logret_1d"].transform(
                lambda values, w=window: values.rolling(
                    w, min_periods=min_periods(w, settings)
                ).sum()
            )
        else:
            working["logret_sum_1d"] = working["logret_1d"]
        if window in settings.return_windows:
            working[f"market_excess_ret_{window}d"] = working[f"ret_{window}d"] - grouped[
                "benchmark_ret_1d"
            ].transform(lambda values, w=window: compound_return(values, w, settings))
    for window in settings.short_windows:
        ret_name = f"ret_{window}d"
        if ret_name in working.columns:
            working[f"reversal_ret_{window}d"] = -working[ret_name]
    return working


def add_trend_features(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Add moving-average, rolling high-low, volatility, and drawdown features."""

    working = frame.copy()
    grouped = working.groupby("ts_code", sort=False)
    for window in settings.medium_windows:
        minp = min_periods(window, settings)
        ma = grouped["adj_close"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).mean()
        )
        std = grouped["adj_close"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).std()
        )
        working[f"ma_ratio_{window}d"] = working["adj_close"] / ma - 1.0
        working[f"ma_z_{window}d"] = (working["adj_close"] - ma) / std.replace(0, np.nan)
        working[f"trend_slope_{window}d"] = ma / grouped["adj_close"].shift(window) - 1.0
        working[f"positive_ret_ratio_{window}d"] = grouped["ret_1d"].transform(
            lambda values, w=window, m=minp: (values > 0).rolling(w, min_periods=m).mean()
        )
        working[f"realized_vol_{window}d"] = grouped["ret_1d"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).std()
        )
        working[f"downside_vol_{window}d"] = grouped["ret_1d"].transform(
            lambda values, w=window, m=minp, mar=settings.downside_mar: (
                rolling_downside_deviation(values, w, m, mar)
            )
        )
    for window in settings.long_windows:
        minp = min_periods(window, settings)
        high = grouped["adj_close"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).max()
        )
        low = grouped["adj_close"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).min()
        )
        range_width = (high - low).replace(0, np.nan)
        working[f"dist_high_{window}d"] = working["adj_close"] / high - 1.0
        working[f"dist_low_{window}d"] = working["adj_close"] / low - 1.0
        working[f"range_pos_{window}d"] = (working["adj_close"] - low) / range_width
        working[f"drawdown_{window}d"] = working["adj_close"] / high - 1.0
        working[f"max_drawdown_{window}d"] = grouped[f"drawdown_{window}d"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).min()
        )
    return working


def add_candle_gap_features(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Add intraday candle and open-gap features."""

    working = frame.copy()
    grouped = working.groupby("ts_code", sort=False)
    intraday_range = (working["adj_high"] - working["adj_low"]).replace(0, np.nan)
    body = working["adj_close"] - working["adj_open"]
    prev_close = grouped["adj_close"].shift(1)
    working["intraday_ret"] = working["adj_close"] / working["adj_open"] - 1.0
    working["candle_body_pct"] = body / intraday_range
    working["upper_shadow_pct"] = (
        working["adj_high"] - working[["adj_open", "adj_close"]].max(axis=1)
    ) / intraday_range
    working["lower_shadow_pct"] = (
        working[["adj_open", "adj_close"]].min(axis=1) - working["adj_low"]
    ) / intraday_range
    working["close_location_value"] = (working["adj_close"] - working["adj_low"]) / intraday_range
    working["gap_open_ret"] = working["adj_open"] / prev_close - 1.0
    working["gap_abs"] = working["gap_open_ret"].abs()
    for window in settings.short_windows:
        working[f"gap_mean_{window}d"] = grouped["gap_open_ret"].transform(
            lambda values, w=window: values.rolling(w, min_periods=1).mean()
        )
    return working


def add_liquidity_features(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Add volume, turnover, amount, Amihud, and price-volume features."""

    working = frame.copy()
    grouped = working.groupby("ts_code", sort=False)
    if "turnover_rate" not in working.columns:
        working["turnover_rate"] = np.nan
    working["amihud_raw"] = working["ret_1d"].abs() / working["amount"].replace(0, np.nan)
    for window in settings.medium_windows:
        minp = min_periods(window, settings)
        vol_mean = grouped["vol"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).mean()
        )
        amount_mean = grouped["amount"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).mean()
        )
        turnover_mean = grouped["turnover_rate"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).mean()
        )
        amount_std = grouped["amount"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).std()
        )
        working[f"volume_ratio_{window}d"] = working["vol"] / vol_mean.replace(0, np.nan)
        working[f"amount_ratio_{window}d"] = working["amount"] / amount_mean.replace(0, np.nan)
        working[f"turnover_ratio_{window}d"] = working["turnover_rate"] / turnover_mean.replace(
            0, np.nan
        )
        working[f"amount_cv_{window}d"] = amount_std / amount_mean.replace(0, np.nan)
    for window in settings.long_windows:
        minp = min_periods(window, settings)
        working[f"amihud_{window}d"] = grouped["amihud_raw"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).mean()
        )
        working[f"ret_amount_corr_{window}d"] = rolling_group_corr(
            working, "ret_1d", "amount", window, minp
        )
        working[f"ret_volume_corr_{window}d"] = rolling_group_corr(
            working, "ret_1d", "vol", window, minp
        )
    return working


def add_beta_features(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Add benchmark beta and residual volatility features."""

    working = frame.copy()
    grouped = working.groupby("ts_code", sort=False)
    for window in settings.long_windows:
        minp = min_periods(window, settings)
        cov = rolling_group_cov(working, "ret_1d", "benchmark_ret_1d", window, minp)
        var = grouped["benchmark_ret_1d"].transform(
            lambda values, w=window, m=minp: values.rolling(w, min_periods=m).var()
        )
        beta = cov / var.replace(0, np.nan)
        working[f"beta_{window}d"] = beta
        residual = working["ret_1d"] - beta * working["benchmark_ret_1d"]
        working[f"residual_vol_{window}d"] = residual.groupby(
            working["ts_code"], sort=False
        ).transform(lambda values, w=window, m=minp: values.rolling(w, min_periods=m).std())
    return working


def add_industry_relative_features(frame: DataFrame, settings: FeatureSettings) -> DataFrame:
    """Reject industry-relative computation without a verified PIT source."""

    del frame, settings
    raise DataValidationError(
        "industry-dependent features are disabled because no verified "
        "point-in-time industry source is configured"
    )


def add_valuation_features(frame: DataFrame) -> DataFrame:
    """Add normalized valuation and size features from daily_basic."""

    working = frame.copy()
    working["earnings_yield"] = inverse_numeric(working, "pe")
    working["book_to_market"] = inverse_numeric(working, "pb")
    working["sales_yield"] = inverse_numeric(working, "ps")
    working["pe_ttm_inv"] = inverse_numeric(working, "pe_ttm")
    working["ps_ttm_inv"] = inverse_numeric(working, "ps_ttm")
    if "dv_ttm" not in working.columns:
        working["dv_ttm"] = np.nan
    working["ln_total_mv"] = safe_log(
        working.get("total_mv", pd.Series(np.nan, index=working.index))
    )
    working["ln_circ_mv"] = safe_log(working.get("circ_mv", pd.Series(np.nan, index=working.index)))
    return working


def add_rank_features(frame: DataFrame) -> DataFrame:
    """Add selected cross-sectional and industry-neutral percentile ranks."""

    from ashare_quant.features.registry import FEATURE_REGISTRY

    working = frame.copy()
    if "in_model_universe" not in working.columns:
        working["in_model_universe"] = True
    eligible = working["in_model_universe"].fillna(False).astype(bool)
    for spec in FEATURE_REGISTRY:
        if spec.family == "cross_sectional_percentile_ranks":
            base = spec.required_source_columns[0]
            if base in working.columns:
                ranks = working.loc[eligible].groupby("trade_date")[base].rank(pct=True)
                working[spec.name] = pd.NA
                working.loc[eligible, spec.name] = ranks
    return working


def compound_return(values: pd.Series, window: int, settings: FeatureSettings) -> pd.Series:
    """Compound simple returns over a rolling window."""

    minp = min_periods(window, settings)
    return (1.0 + values).rolling(window, min_periods=minp).apply(np.prod, raw=True) - 1.0


def rolling_downside_deviation(
    values: pd.Series,
    window: int,
    minimum_periods: int,
    mar: float,
) -> pd.Series:
    """Compute rolling downside deviation with non-negative returns contributing zero."""

    downside = (values - mar).clip(upper=0)
    result = np.sqrt((downside**2).rolling(window, min_periods=minimum_periods).mean())
    return pd.Series(result, index=values.index)


def rolling_group_corr(
    frame: DataFrame,
    left_column: str,
    right_column: str,
    window: int,
    minimum_periods: int,
) -> pd.Series:
    """Compute rolling correlation per stock without forward-looking rows."""

    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in frame.groupby("ts_code", sort=False):
        result.loc[group.index] = (
            group[left_column]
            .rolling(window, min_periods=minimum_periods)
            .corr(group[right_column])
        )
    return result


def rolling_group_cov(
    frame: DataFrame,
    left_column: str,
    right_column: str,
    window: int,
    minimum_periods: int,
) -> pd.Series:
    """Compute rolling covariance per stock without forward-looking rows."""

    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, group in frame.groupby("ts_code", sort=False):
        result.loc[group.index] = (
            group[left_column].rolling(window, min_periods=minimum_periods).cov(group[right_column])
        )
    return result


def min_periods(window: int, settings: FeatureSettings) -> int:
    """Return minimum observations for a rolling window."""

    return max(1, int(np.ceil(window * settings.min_traded_observation_fraction)))


def inverse_numeric(frame: DataFrame, column: str) -> pd.Series:
    """Return inverse of a numeric column, preserving missing or non-positive values as NaN."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    return 1.0 / values.where(values > 0)


def safe_log(values: pd.Series) -> pd.Series:
    """Return log of positive values only."""

    numeric = pd.to_numeric(values, errors="coerce")
    logged = np.log(numeric.where(numeric > 0))
    return pd.Series(logged, index=values.index)
