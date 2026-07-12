"""Point-in-time A-share universe construction."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import pandas as pd

from ashare_quant.config.settings import AppSettings, UniverseSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.universe.storage import UNIVERSE_COLUMNS, UniverseStore
from ashare_quant.universe.tradability import add_tradability_flags
from ashare_quant.universe.validation import UniverseValidationResult, validate_universe_frame

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class UniverseBuildResult:
    """Summary of a universe build run."""

    start_date: str
    end_date: str
    rows_written: int
    rows_built: int
    validation: UniverseValidationResult


class UniverseBuilder:
    """Build daily point-in-time base and model universes from canonical raw data."""

    def __init__(
        self,
        raw_store: ParquetDataStore,
        universe_store: UniverseStore,
        settings: AppSettings,
    ) -> None:
        self._raw_store = raw_store
        self._universe_store = universe_store
        self._settings = settings

    def build(self, start_date: str, end_date: str) -> UniverseBuildResult:
        """Build and persist daily universe rows for an inclusive date range."""

        inputs = prepare_universe_inputs(self._load_inputs())
        rows_built = 0
        rows_written = 0
        errors: list[str] = []
        warnings: list[str] = []
        for chunk_start, chunk_end in year_date_ranges(
            open_trade_dates(inputs["trade_cal"], start_date, end_date)
        ):
            frame = build_universe_frame(
                inputs, self._settings.universe, chunk_start, chunk_end
            )
            rows_built += len(frame)
            validation = validate_universe_frame(frame)
            errors.extend(validation.errors)
            warnings.extend(validation.warnings)
            if not validation.ok:
                continue
            rows_written += self._universe_store.write(frame)
        validation = UniverseValidationResult(
            ok=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )
        return UniverseBuildResult(
            start_date=start_date,
            end_date=end_date,
            rows_written=rows_written,
            rows_built=rows_built,
            validation=validation,
        )

    def preview(self, start_date: str, end_date: str) -> DataFrame:
        """Build universe rows in memory without writing."""

        return build_universe_frame(
            self._load_inputs(), self._settings.universe, start_date, end_date
        )

    def _load_inputs(self) -> dict[str, DataFrame]:
        names = ("stock_basic", "trade_cal", "daily", "daily_basic", "suspend_d", "stk_limit")
        return {name: self._raw_store.read_dataset(get_dataset_spec(name)) for name in names}


def build_universe_frame(
    inputs: dict[str, DataFrame],
    settings: UniverseSettings,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Build a daily universe frame from already loaded raw input frames."""

    all_trade_dates = open_trade_dates(inputs["trade_cal"], None, end_date)
    build_dates = [date for date in all_trade_dates if start_date <= date <= end_date]
    if not build_dates:
        return pd.DataFrame(columns=list(UNIVERSE_COLUMNS))

    daily = inputs.get("_daily_normalized", normalize_daily(inputs["daily"]))
    candidates = inputs.get("_candidates", build_candidates(inputs["stock_basic"], daily))
    if candidates.empty:
        raise DataValidationError("cannot build universe without stock_basic or daily candidates")

    calendar = pd.DataFrame({"trade_date": build_dates})
    base = calendar.merge(candidates, how="cross")
    base = add_listing_flags(base, all_trade_dates)
    base = merge_market_data(
        base,
        daily,
        inputs["daily_basic"],
        inputs["suspend_d"],
        inputs["stk_limit"],
        daily_with_liquidity=inputs.get("_daily_with_liquidity"),
        daily_basic_norm=inputs.get("_daily_basic_normalized"),
        suspend_keys=inputs.get("_suspend_keys"),
        limit_prices=inputs.get("_limit_prices"),
    )
    base = add_model_flags(base, settings)
    base = add_tradability_flags(base, settings)
    base = finalize_columns(base)
    return base.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def prepare_universe_inputs(inputs: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Precompute full-history normalized inputs reused by chunked universe builds."""

    prepared = dict(inputs)
    daily = normalize_daily(inputs["daily"])
    prepared["_daily_normalized"] = daily
    prepared["_daily_with_liquidity"] = add_liquidity_features(daily)
    prepared["_daily_basic_normalized"] = normalize_daily_basic(inputs["daily_basic"])
    prepared["_suspend_keys"] = normalize_suspend_keys(inputs["suspend_d"])
    prepared["_limit_prices"] = normalize_limit_prices(inputs["stk_limit"])
    prepared["_candidates"] = build_candidates(inputs["stock_basic"], daily)
    return prepared


def year_date_ranges(trade_dates: list[str]) -> list[tuple[str, str]]:
    """Return inclusive year chunks for sorted YYYYMMDD trade dates."""

    if not trade_dates:
        return []
    ranges: list[tuple[str, str]] = []
    chunk_start = trade_dates[0]
    current_year = chunk_start[:4]
    previous = chunk_start
    for trade_date in trade_dates[1:]:
        if trade_date[:4] != current_year:
            ranges.append((chunk_start, previous))
            chunk_start = trade_date
            current_year = trade_date[:4]
        previous = trade_date
    ranges.append((chunk_start, previous))
    return ranges


def open_trade_dates(trade_cal: DataFrame, start_date: str | None, end_date: str) -> list[str]:
    """Return authoritative open trading dates from trade_cal."""

    required = {"cal_date", "is_open"}
    if trade_cal.empty or not required.issubset(trade_cal.columns):
        raise DataValidationError("trade_cal with cal_date and is_open is required")
    calendar = trade_cal.copy()
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="coerce").fillna(0).astype(int)
    mask = (calendar["is_open"] == 1) & (calendar["cal_date"] <= end_date)
    if start_date is not None:
        mask &= calendar["cal_date"] >= start_date
    dates = calendar.loc[mask, "cal_date"]
    return sorted(dates.drop_duplicates().astype(str).tolist())


def normalize_daily(daily: DataFrame) -> DataFrame:
    """Normalize daily price data and drop unusable pre-2020 invalid OHLC rows."""

    if daily.empty:
        return pd.DataFrame(
            columns=["ts_code", "trade_date", "close", "amount", "price_ohlc_valid"]
        )
    working = daily.copy()
    working["ts_code"] = working["ts_code"].astype(str)
    working["trade_date"] = working["trade_date"].astype(str)
    for column in ("open", "high", "low", "close", "amount"):
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        else:
            working[column] = pd.NA
    working["price_ohlc_valid"] = True
    prices = working[["open", "high", "low", "close"]]
    invalid = (
        (prices["high"] < prices[["open", "close"]].max(axis=1))
        | (prices["low"] > prices[["open", "close"]].min(axis=1))
        | (prices["high"] < prices["low"])
    ).fillna(True)
    pre_2020_invalid = invalid & (working["trade_date"] < "20200101")
    working = working.loc[~pre_2020_invalid].copy()
    working["price_ohlc_valid"] = ~invalid.loc[working.index]
    return working.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")


def build_candidates(stock_basic: DataFrame, daily: DataFrame) -> DataFrame:
    """Build a stock candidate table from stock_basic plus historical daily appearances."""

    stock = normalize_stock_basic(stock_basic)
    daily_bounds = daily_bounds_by_code(daily)
    if stock.empty:
        candidates = daily_bounds.copy()
        candidates["_from_stock_basic"] = False
    else:
        candidates = stock.merge(daily_bounds, on="ts_code", how="outer")
        candidates["_from_stock_basic"] = candidates["name"].notna()

    candidates["list_date"] = candidates["list_date"].fillna(candidates["first_daily_date"])
    candidates["name"] = candidates["name"].fillna(candidates["ts_code"])
    candidates["market"] = candidates["market"].fillna("")
    candidates["industry"] = candidates["industry"].fillna("")
    candidates["exchange"] = candidates["exchange"].fillna(
        candidates["ts_code"].map(exchange_from_ts_code)
    )
    candidates["delist_date"] = clean_date_series(candidates.get("delist_date"))

    daily_only = ~candidates["_from_stock_basic"].astype(bool)
    candidates.loc[daily_only & candidates["delist_date"].isna(), "delist_date"] = candidates.loc[
        daily_only & candidates["delist_date"].isna(), "last_daily_date"
    ]
    return candidates[
        ["ts_code", "name", "market", "exchange", "industry", "list_date", "delist_date"]
    ].drop_duplicates(subset=["ts_code"], keep="last")


def normalize_stock_basic(stock_basic: DataFrame) -> DataFrame:
    """Normalize stock_basic metadata columns used by the universe builder."""

    if stock_basic.empty:
        return pd.DataFrame(
            columns=[
                "ts_code",
                "name",
                "market",
                "exchange",
                "industry",
                "list_date",
                "delist_date",
            ]
        )
    stock = stock_basic.copy()
    stock["ts_code"] = stock["ts_code"].astype(str)
    for column in ("name", "market", "industry"):
        if column not in stock.columns:
            stock[column] = ""
        stock[column] = stock[column].fillna("").astype(str)
    if "list_date" not in stock.columns:
        stock["list_date"] = pd.NA
    stock["list_date"] = clean_date_series(stock["list_date"])
    if "delist_date" not in stock.columns:
        stock["delist_date"] = pd.NA
    stock["delist_date"] = clean_date_series(stock["delist_date"])
    stock["exchange"] = stock["ts_code"].map(exchange_from_ts_code)
    return stock[
        ["ts_code", "name", "market", "exchange", "industry", "list_date", "delist_date"]
    ].drop_duplicates(subset=["ts_code"], keep="last")


def daily_bounds_by_code(daily: DataFrame) -> DataFrame:
    """Return first and last raw daily dates for each stock code."""

    if daily.empty or not {"ts_code", "trade_date"}.issubset(daily.columns):
        return pd.DataFrame(columns=["ts_code", "first_daily_date", "last_daily_date"])
    grouped = daily.groupby("ts_code")["trade_date"]
    first_dates = grouped.min()
    last_dates = grouped.max()
    return pd.DataFrame(
        {
            "ts_code": first_dates.index.astype(str),
            "first_daily_date": first_dates.to_numpy(),
            "last_daily_date": last_dates.to_numpy(),
        }
    )


def add_listing_flags(base: DataFrame, all_trade_dates: list[str]) -> DataFrame:
    """Add point-in-time listed and list-age flags using the full trading calendar."""

    working = base.copy()
    working["list_date"] = clean_date_series(working["list_date"])
    working["delist_date"] = clean_date_series(working["delist_date"])
    working["is_listed"] = (
        working["list_date"].notna()
        & (working["trade_date"] >= working["list_date"])
        & (working["delist_date"].isna() | (working["trade_date"] <= working["delist_date"]))
    )
    working["list_days"] = vectorized_trading_day_counts(
        all_trade_dates,
        working["list_date"],
        working["trade_date"],
        working["is_listed"],
    )
    working["in_base_universe"] = working["is_listed"]
    return working


def vectorized_trading_day_counts(
    all_trade_dates: list[str],
    list_dates: pd.Series,
    trade_dates: pd.Series,
    is_listed: pd.Series,
) -> pd.Series:
    """Count listed open days for many rows using calendar position maps."""

    if not all_trade_dates:
        return pd.Series(0, index=trade_dates.index, dtype="int64")
    calendar = pd.Series(range(1, len(all_trade_dates) + 1), index=pd.Index(all_trade_dates))
    trade_positions = trade_dates.map(calendar).fillna(0).astype(int)

    list_frame = pd.DataFrame({"list_date": clean_date_series(list_dates).drop_duplicates()})
    list_frame = list_frame[list_frame["list_date"].notna()].copy()
    if list_frame.empty:
        counts = pd.Series(0, index=trade_dates.index, dtype="int64")
        return counts
    list_frame["start_position"] = [
        trading_day_count_start_position(all_trade_dates, value)
        for value in list_frame["list_date"]
    ]
    start_positions = clean_date_series(list_dates).map(
        list_frame.set_index("list_date")["start_position"]
    )
    counts = trade_positions - start_positions.fillna(trade_positions + 1).astype(int) + 1
    counts = counts.clip(lower=0)
    counts = counts.where(is_listed.fillna(False).astype(bool), 0)
    return counts.astype(int)


def trading_day_count_start_position(all_trade_dates: list[str], start_date: object) -> int:
    """Return one-based calendar position of the first listed trading day."""

    if is_missing_scalar(start_date):
        return 1
    start = str(start_date)
    position = bisect_left(all_trade_dates, start) + 1
    if start < all_trade_dates[0]:
        start_dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
        first_dt = pd.to_datetime(all_trade_dates[0], format="%Y%m%d", errors="coerce")
        if isinstance(start_dt, pd.Timestamp) and isinstance(first_dt, pd.Timestamp):
            natural_days = max(int((first_dt - start_dt).days), 0)
            return position - natural_days * 5 // 7
    return position


def trading_day_count(all_trade_dates: list[str], start_date: object, end_date: object) -> int:
    """Count listed open days, approximating age before the available calendar."""

    if is_missing_scalar(start_date) or is_missing_scalar(end_date) or not all_trade_dates:
        return 0
    start = str(start_date)
    end = str(end_date)
    left = bisect_left(all_trade_dates, start)
    right = bisect_right(all_trade_dates, end)
    pre_calendar_days = 0
    if start < all_trade_dates[0]:
        start_dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
        first_dt = pd.to_datetime(all_trade_dates[0], format="%Y%m%d", errors="coerce")
        if isinstance(start_dt, pd.Timestamp) and isinstance(first_dt, pd.Timestamp):
            natural_days = max(int((first_dt - start_dt).days), 0)
            pre_calendar_days = natural_days * 5 // 7
    return max(right - left, 0) + pre_calendar_days


def merge_market_data(
    base: DataFrame,
    daily: DataFrame,
    daily_basic: DataFrame,
    suspend_d: DataFrame,
    stk_limit: DataFrame,
    daily_with_liquidity: DataFrame | None = None,
    daily_basic_norm: DataFrame | None = None,
    suspend_keys: DataFrame | None = None,
    limit_prices: DataFrame | None = None,
) -> DataFrame:
    """Merge daily market, liquidity, suspension, and limit-price data."""

    if daily_with_liquidity is None:
        daily_with_liquidity = add_liquidity_features(daily)
    working = base.merge(
        daily_with_liquidity,
        on=["ts_code", "trade_date"],
        how="left",
        suffixes=("", "_daily"),
    )
    price_valid = working["price_ohlc_valid"] if "price_ohlc_valid" in working.columns else False
    working["has_price_data"] = working["close"].notna() & pd.Series(
        price_valid, index=working.index
    ).fillna(False)

    if daily_basic_norm is None:
        daily_basic_norm = normalize_daily_basic(daily_basic)
    if not daily_basic_norm.empty:
        working = working.merge(daily_basic_norm, on=["ts_code", "trade_date"], how="left")

    working = merge_suspension(working, suspend_d, suspend_keys=suspend_keys)
    working = merge_limit_prices(working, stk_limit, limit_prices=limit_prices)
    return working


def normalize_daily_basic(daily_basic: DataFrame) -> DataFrame:
    """Normalize daily_basic data used by later extensions."""

    if daily_basic.empty or not {"ts_code", "trade_date"}.issubset(daily_basic.columns):
        return pd.DataFrame(columns=["ts_code", "trade_date"])
    working = daily_basic.copy()
    working["ts_code"] = working["ts_code"].astype(str)
    working["trade_date"] = working["trade_date"].astype(str)
    keep_columns = [
        column
        for column in ("ts_code", "trade_date", "turnover_rate", "total_mv")
        if column in working
    ]
    return working[keep_columns].drop_duplicates(subset=["ts_code", "trade_date"], keep="last")


def add_liquidity_features(frame: DataFrame) -> DataFrame:
    """Add trailing amount averages using only current and prior rows per stock."""

    working = frame.sort_values(["ts_code", "trade_date"]).copy()
    if "amount" not in working.columns:
        working["amount"] = pd.NA
    working["amount"] = pd.to_numeric(working["amount"], errors="coerce")
    grouped = working.groupby("ts_code", sort=False)["amount"]
    working["avg_amount_20"] = grouped.transform(
        lambda values: values.rolling(20, min_periods=1).mean()
    )
    working["amount_count_20"] = grouped.transform(
        lambda values: values.rolling(20, min_periods=1).count()
    )
    return working


def merge_suspension(
    frame: DataFrame,
    suspend_d: DataFrame,
    suspend_keys: DataFrame | None = None,
) -> DataFrame:
    """Add suspension flags from suspend_d."""

    working = frame.copy()
    if suspend_keys is None:
        suspend_keys = normalize_suspend_keys(suspend_d)
    if suspend_keys.empty:
        working["is_suspended"] = False
        return working
    merged = working.merge(
        suspend_keys.assign(is_suspended=True), on=["ts_code", "trade_date"], how="left"
    )
    merged["is_suspended"] = merged["is_suspended"].fillna(False).astype(bool)
    return merged


def normalize_suspend_keys(suspend_d: DataFrame) -> DataFrame:
    """Normalize suspension keys once for chunked universe builds."""

    if suspend_d.empty or not {"ts_code", "trade_date"}.issubset(suspend_d.columns):
        return pd.DataFrame(columns=["ts_code", "trade_date"])
    suspend_keys = suspend_d[["ts_code", "trade_date"]].copy()
    suspend_keys["ts_code"] = suspend_keys["ts_code"].astype(str)
    suspend_keys["trade_date"] = suspend_keys["trade_date"].astype(str)
    return suspend_keys.drop_duplicates()


def merge_limit_prices(
    frame: DataFrame,
    stk_limit: DataFrame,
    limit_prices: DataFrame | None = None,
) -> DataFrame:
    """Add up_limit and down_limit columns from stk_limit."""

    working = frame.copy()
    if limit_prices is None:
        limit_prices = normalize_limit_prices(stk_limit)
    if limit_prices.empty:
        working["up_limit"] = pd.NA
        working["down_limit"] = pd.NA
        return working
    return working.merge(limit_prices, on=["ts_code", "trade_date"], how="left")


def normalize_limit_prices(stk_limit: DataFrame) -> DataFrame:
    """Normalize limit-price data once for chunked universe builds."""

    if stk_limit.empty or not {"ts_code", "trade_date"}.issubset(stk_limit.columns):
        return pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    limits = stk_limit.copy()
    limits["ts_code"] = limits["ts_code"].astype(str)
    limits["trade_date"] = limits["trade_date"].astype(str)
    keep = [
        column for column in ("ts_code", "trade_date", "up_limit", "down_limit") if column in limits
    ]
    return limits[keep].drop_duplicates(subset=["ts_code", "trade_date"], keep="last")


def add_model_flags(frame: DataFrame, settings: UniverseSettings) -> DataFrame:
    """Add model-universe flags and exclusion reasons separately from tradability."""

    working = frame.copy()
    working["is_new_stock"] = working["in_base_universe"].astype(bool) & (
        working["list_days"] < settings.min_list_trading_days
    )
    working["is_st"] = working["name"].astype(str).str.upper().str.contains("ST", regex=False)

    avg_amount = pd.to_numeric(working["avg_amount_20"], errors="coerce")
    amount_count = pd.to_numeric(working["amount_count_20"], errors="coerce")
    low_amount = avg_amount.isna() | (avg_amount < settings.min_avg_amount)
    insufficient_window = amount_count < settings.liquidity_window_days
    working["is_low_liquidity"] = (
        low_amount | insufficient_window if settings.require_full_liquidity_window else low_amount
    )

    has_price_data = working["has_price_data"].astype(bool)
    if settings.min_price is not None:
        close = pd.to_numeric(working["close"], errors="coerce")
        has_price_data = has_price_data & (close >= settings.min_price)
    working["has_price_data"] = has_price_data

    reasons = pd.Series("", index=working.index, dtype="object")
    reasons = append_reason(reasons, ~working["in_base_universe"].astype(bool), "not_listed")
    reasons = append_reason(reasons, working["is_new_stock"].astype(bool), "new_stock")
    reasons = append_reason(reasons, working["is_st"].astype(bool) & settings.exclude_st, "st")
    reasons = append_reason(reasons, working["is_suspended"].astype(bool), "suspended")
    reasons = append_reason(
        reasons, ~working["has_price_data"].astype(bool), "insufficient_price_data"
    )
    reasons = append_reason(reasons, working["is_low_liquidity"].astype(bool), "low_liquidity")
    working["exclude_reason"] = reasons

    working["in_model_universe"] = (
        working["in_base_universe"].astype(bool)
        & ~working["is_new_stock"].astype(bool)
        & ~(working["is_st"].astype(bool) & settings.exclude_st)
        & ~working["is_suspended"].astype(bool)
        & working["has_price_data"].astype(bool)
        & ~working["is_low_liquidity"].astype(bool)
    )
    return working


def append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    """Append one semicolon-separated exclusion reason where mask is true."""

    updated = reasons.copy()
    selected = mask.fillna(False).astype(bool)
    empty_selected = selected & updated.eq("")
    nonempty_selected = selected & ~updated.eq("")
    updated.loc[empty_selected] = reason
    updated.loc[nonempty_selected] = updated.loc[nonempty_selected] + ";" + reason
    return updated


def finalize_columns(frame: DataFrame) -> DataFrame:
    """Return public universe output columns with stable dtypes."""

    working = frame.copy()
    for column in (
        "is_listed",
        "is_new_stock",
        "is_st",
        "is_suspended",
        "is_low_liquidity",
        "is_limit_up",
        "is_limit_down",
        "can_buy",
        "can_sell",
        "in_base_universe",
        "in_model_universe",
    ):
        working[column] = working[column].fillna(False).astype(bool)
    for column in ("name", "market", "exchange", "industry", "exclude_reason"):
        working[column] = working[column].fillna("").astype(str)
    working["list_date"] = clean_date_series(working["list_date"])
    working["delist_date"] = clean_date_series(working["delist_date"])
    working["list_days"] = (
        pd.to_numeric(working["list_days"], errors="coerce").fillna(0).astype(int)
    )
    return working.loc[:, list(UNIVERSE_COLUMNS)]


def clean_date_series(series: pd.Series | None) -> pd.Series:
    """Normalize Tushare date strings and convert blanks/zeroes to missing values."""

    if series is None:
        return pd.Series(dtype="object")
    cleaned = series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "0", "None", "NaT", "nan", "<NA>"]))
    return cleaned.astype("object")


def is_missing_scalar(value: object) -> bool:
    """Return whether a scalar date-like value should be treated as missing."""

    if value is None or value is pd.NA:
        return True
    return str(value) in {"", "0", "None", "NaT", "nan", "<NA>"}


def exchange_from_ts_code(ts_code: str) -> str:
    """Infer exchange from a Tushare stock code suffix."""

    if ts_code.endswith(".SH"):
        return "SSE"
    if ts_code.endswith(".SZ"):
        return "SZSE"
    if ts_code.endswith(".BJ"):
        return "BSE"
    return ""
