"""Executable forward-return label construction."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from ashare_quant.config.settings import AppSettings, LabelSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.labels.storage import LABEL_COLUMNS, LabelStore
from ashare_quant.labels.validation import LabelValidationResult, validate_label_frame
from ashare_quant.universe import UniverseStore

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class LabelBuildResult:
    """Summary of a label build run."""

    start_date: str
    end_date: str
    horizons: tuple[int, ...]
    rows_built: int
    rows_written: int
    validation: LabelValidationResult


class LabelBuilder:
    """Build executable forward-return labels from raw prices and universe flags."""

    def __init__(
        self,
        raw_store: ParquetDataStore,
        universe_store: UniverseStore,
        label_store: LabelStore,
        settings: AppSettings,
    ) -> None:
        self._raw_store = raw_store
        self._universe_store = universe_store
        self._label_store = label_store
        self._settings = settings

    def build(
        self,
        start_date: str,
        end_date: str,
        horizons: tuple[int, ...] | None = None,
    ) -> LabelBuildResult:
        """Build and persist labels for an inclusive signal-date range."""

        selected_horizons = horizons or self._settings.labels.horizons
        inputs = self._load_inputs(start_date, end_date, selected_horizons)
        frame = build_label_frame(
            inputs, self._settings.labels, start_date, end_date, selected_horizons
        )
        validation = validate_label_frame(
            frame,
            self._settings.labels.quantile_buckets,
            selected_horizons,
        )
        rows_written = self._label_store.write(frame) if validation.ok else 0
        return LabelBuildResult(
            start_date=start_date,
            end_date=end_date,
            horizons=selected_horizons,
            rows_built=len(frame),
            rows_written=rows_written,
            validation=validation,
        )

    def preview(
        self,
        start_date: str,
        end_date: str,
        horizons: tuple[int, ...] | None = None,
    ) -> DataFrame:
        """Build labels in memory without writing."""

        selected_horizons = horizons or self._settings.labels.horizons
        return build_label_frame(
            self._load_inputs(start_date, end_date, selected_horizons),
            self._settings.labels,
            start_date,
            end_date,
            selected_horizons,
        )

    def _load_inputs(
        self,
        start_date: str,
        end_date: str,
        horizons: tuple[int, ...],
    ) -> dict[str, DataFrame]:
        max_horizon = max(horizons) if horizons else 0
        calendar = self._raw_store.read_dataset(get_dataset_spec("trade_cal"))
        future_end = future_calendar_end(
            calendar, end_date, max_horizon + self._settings.labels.max_exit_delay_days + 1
        )
        return {
            "trade_cal": calendar,
            "daily": self._raw_store.read_dataset(get_dataset_spec("daily")),
            "adj_factor": self._raw_store.read_dataset(get_dataset_spec("adj_factor")),
            "stk_limit": self._raw_store.read_dataset(get_dataset_spec("stk_limit")),
            "index_daily": self._raw_store.read_dataset(get_dataset_spec("index_daily")),
            "universe": self._universe_store.read(start_date, future_end),
        }


def build_label_frame(
    inputs: dict[str, DataFrame],
    settings: LabelSettings,
    start_date: str,
    end_date: str,
    horizons: tuple[int, ...],
) -> DataFrame:
    """Build executable label rows from loaded inputs."""

    if settings.delay_unsellable_exit:
        return build_label_frame_iterative(inputs, settings, start_date, end_date, horizons)

    return build_label_frame_vectorized(inputs, settings, start_date, end_date, horizons)


def build_label_frame_vectorized(
    inputs: dict[str, DataFrame],
    settings: LabelSettings,
    start_date: str,
    end_date: str,
    horizons: tuple[int, ...],
) -> DataFrame:
    """Build label rows with vectorized joins for the default non-delayed exit mode."""

    trade_dates = open_trade_dates(inputs["trade_cal"], None, None)
    signal_dates = [date for date in trade_dates if start_date <= date <= end_date]
    if not signal_dates or not horizons:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))

    prices = adjusted_stock_open_prices(inputs["daily"], inputs["adj_factor"])
    limits = limit_prices(inputs.get("stk_limit", pd.DataFrame()))
    benchmark = benchmark_open_prices(inputs["index_daily"], settings.benchmark_index_code)
    universe = normalize_universe(inputs["universe"])
    signal_universe = universe[
        universe["trade_date"].isin(signal_dates) & universe["in_base_universe"].astype(bool)
    ].copy()
    if signal_universe.empty:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))

    frame = build_signal_horizon_grid(signal_universe, trade_dates, signal_dates, horizons)
    frame = attach_label_inputs_duckdb(frame, universe, prices, limits, benchmark)
    frame = assign_label_availability(frame, settings)
    frame = add_ranking_labels(frame[list(LABEL_COLUMNS)], settings.quantile_buckets)
    return frame.sort_values(["trade_date", "horizon", "ts_code"]).reset_index(drop=True)


def build_label_frame_iterative(
    inputs: dict[str, DataFrame],
    settings: LabelSettings,
    start_date: str,
    end_date: str,
    horizons: tuple[int, ...],
) -> DataFrame:
    """Build executable label rows with the original row-wise delayed-exit logic."""

    trade_dates = open_trade_dates(inputs["trade_cal"], None, None)
    signal_dates = [date for date in trade_dates if start_date <= date <= end_date]
    if not signal_dates or not horizons:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))

    prices = adjusted_stock_open_prices(inputs["daily"], inputs["adj_factor"])
    limits = limit_prices(inputs.get("stk_limit", pd.DataFrame()))
    benchmark = benchmark_open_prices(inputs["index_daily"], settings.benchmark_index_code)
    universe = normalize_universe(inputs["universe"])
    signal_universe = universe[
        universe["trade_date"].isin(signal_dates) & universe["in_base_universe"].astype(bool)
    ].copy()
    if signal_universe.empty:
        return pd.DataFrame(columns=list(LABEL_COLUMNS))

    rows: list[dict[str, object]] = []
    for signal_row in signal_universe.itertuples(index=False):
        trade_date = str(signal_row.trade_date)
        ts_code = str(signal_row.ts_code)
        for horizon in horizons:
            rows.append(
                build_one_label(
                    trade_date=trade_date,
                    ts_code=ts_code,
                    horizon=horizon,
                    trade_dates=trade_dates,
                    prices=prices,
                    limits=limits,
                    benchmark=benchmark,
                    universe=universe,
                    settings=settings,
                )
            )

    frame = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    frame = add_ranking_labels(frame, settings.quantile_buckets)
    return frame.sort_values(["trade_date", "horizon", "ts_code"]).reset_index(drop=True)


def build_signal_horizon_grid(
    signal_universe: DataFrame,
    trade_dates: list[str],
    signal_dates: list[str],
    horizons: tuple[int, ...],
) -> DataFrame:
    """Return one row per signal stock and requested horizon with entry/exit dates."""

    calendar = calendar_horizon_map(trade_dates, signal_dates, horizons)
    base_columns = ["trade_date", "ts_code"]
    base = signal_universe[base_columns].drop_duplicates(subset=base_columns, keep="last")
    return base.merge(calendar, on="trade_date", how="inner")


def calendar_horizon_map(
    trade_dates: list[str],
    signal_dates: list[str],
    horizons: tuple[int, ...],
) -> DataFrame:
    """Map signal dates and horizons to entry and planned exit dates."""

    date_position = {date: position for position, date in enumerate(trade_dates)}
    rows: list[dict[str, object]] = []
    for trade_date in signal_dates:
        position = date_position.get(trade_date)
        if position is None:
            continue
        entry_position = position + 1
        entry_date = trade_dates[entry_position] if entry_position < len(trade_dates) else ""
        for horizon in horizons:
            exit_position = entry_position + horizon
            exit_date = trade_dates[exit_position] if exit_position < len(trade_dates) else ""
            rows.append(
                {
                    "trade_date": trade_date,
                    "horizon": horizon,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                }
            )
    return pd.DataFrame(rows, columns=["trade_date", "horizon", "entry_date", "exit_date"])


def attach_entry_exit_universe(frame: DataFrame, universe: DataFrame) -> DataFrame:
    """Attach entry-date and exit-date universe/tradability flags by vectorized joins."""

    universe_flags = universe[
        [
            "trade_date",
            "ts_code",
            "in_base_universe",
            "can_buy",
            "can_sell",
            "is_limit_up",
            "is_limit_down",
            "is_suspended",
        ]
    ].drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    entry_flags = universe_flags.rename(
        columns={
            "trade_date": "entry_date",
            "in_base_universe": "entry_in_base_universe",
            "can_buy": "entry_can_buy",
            "is_limit_up": "entry_is_limit_up",
            "is_suspended": "entry_is_suspended",
        }
    )[[
        "entry_date",
        "ts_code",
        "entry_in_base_universe",
        "entry_can_buy",
        "entry_is_limit_up",
        "entry_is_suspended",
    ]]
    exit_flags = universe_flags.rename(
        columns={
            "trade_date": "exit_date",
            "in_base_universe": "exit_in_base_universe",
            "can_sell": "exit_can_sell",
            "is_limit_down": "exit_is_limit_down",
            "is_suspended": "exit_is_suspended",
        }
    )[[
        "exit_date",
        "ts_code",
        "exit_in_base_universe",
        "exit_can_sell",
        "exit_is_limit_down",
        "exit_is_suspended",
    ]]
    merged = frame.merge(entry_flags, on=["entry_date", "ts_code"], how="left")
    return merged.merge(exit_flags, on=["exit_date", "ts_code"], how="left")


def attach_prices_and_benchmark(
    frame: DataFrame, prices: DataFrame, benchmark: DataFrame
) -> DataFrame:
    """Attach stock entry/exit prices and benchmark entry/exit prices."""

    entry_prices = prices.rename(
        columns={"trade_date": "entry_date", "adjusted_open": "entry_price"}
    )[["ts_code", "entry_date", "entry_price"]]
    exit_prices = prices.rename(
        columns={"trade_date": "exit_date", "adjusted_open": "exit_price"}
    )[["ts_code", "exit_date", "exit_price"]]
    benchmark_entry = benchmark.rename(
        columns={"trade_date": "entry_date", "benchmark_open": "benchmark_entry_price"}
    )[["entry_date", "benchmark_entry_price"]]
    benchmark_exit = benchmark.rename(
        columns={"trade_date": "exit_date", "benchmark_open": "benchmark_exit_price"}
    )[["exit_date", "benchmark_exit_price"]]

    merged = frame.merge(entry_prices, on=["ts_code", "entry_date"], how="left")
    merged = merged.merge(exit_prices, on=["ts_code", "exit_date"], how="left")
    merged = merged.merge(benchmark_entry, on="entry_date", how="left")
    return merged.merge(benchmark_exit, on="exit_date", how="left")


def attach_label_inputs_duckdb(
    frame: DataFrame,
    universe: DataFrame,
    prices: DataFrame,
    limits: DataFrame,
    benchmark: DataFrame,
) -> DataFrame:
    """Attach universe flags, prices, and benchmark prices using DuckDB joins."""

    universe_flags = universe[
        [
            "trade_date",
            "ts_code",
            "in_base_universe",
            "is_suspended",
        ]
    ].drop_duplicates(subset=["trade_date", "ts_code"], keep="last")
    entry_flags = universe_flags.rename(
        columns={
            "trade_date": "entry_date",
            "in_base_universe": "entry_in_base_universe",
            "is_suspended": "entry_is_suspended",
        }
    )[[
        "entry_date",
        "ts_code",
        "entry_in_base_universe",
        "entry_is_suspended",
    ]]
    exit_flags = universe_flags.rename(
        columns={
            "trade_date": "exit_date",
            "in_base_universe": "exit_in_base_universe",
            "is_suspended": "exit_is_suspended",
        }
    )[[
        "exit_date",
        "ts_code",
        "exit_in_base_universe",
        "exit_is_suspended",
    ]]
    entry_prices = prices.rename(
        columns={
            "trade_date": "entry_date",
            "adjusted_open": "entry_price",
            "open": "entry_open_price",
        }
    )[["ts_code", "entry_date", "entry_price", "entry_open_price"]]
    exit_prices = prices.rename(
        columns={
            "trade_date": "exit_date",
            "adjusted_open": "exit_price",
            "open": "exit_open_price",
        }
    )[["ts_code", "exit_date", "exit_price", "exit_open_price"]]
    entry_limits = limits.rename(
        columns={
            "trade_date": "entry_date",
            "up_limit": "entry_up_limit",
            "down_limit": "entry_down_limit",
        }
    )[["ts_code", "entry_date", "entry_up_limit", "entry_down_limit"]]
    exit_limits = limits.rename(
        columns={
            "trade_date": "exit_date",
            "up_limit": "exit_up_limit",
            "down_limit": "exit_down_limit",
        }
    )[["ts_code", "exit_date", "exit_up_limit", "exit_down_limit"]]
    benchmark_entry = benchmark.rename(
        columns={"trade_date": "entry_date", "benchmark_open": "benchmark_entry_price"}
    )[["entry_date", "benchmark_entry_price"]]
    benchmark_exit = benchmark.rename(
        columns={"trade_date": "exit_date", "benchmark_open": "benchmark_exit_price"}
    )[["exit_date", "benchmark_exit_price"]]

    connection = duckdb.connect(":memory:")
    try:
        connection.register("base_frame", frame)
        connection.register("entry_flags", entry_flags)
        connection.register("exit_flags", exit_flags)
        connection.register("entry_prices", entry_prices)
        connection.register("exit_prices", exit_prices)
        connection.register("entry_limits", entry_limits)
        connection.register("exit_limits", exit_limits)
        connection.register("benchmark_entry", benchmark_entry)
        connection.register("benchmark_exit", benchmark_exit)
        return connection.execute(
            """
            SELECT
                base_frame.*,
                entry_flags.entry_in_base_universe,
                entry_flags.entry_is_suspended,
                exit_flags.exit_in_base_universe,
                exit_flags.exit_is_suspended,
                entry_prices.entry_price,
                entry_prices.entry_open_price,
                exit_prices.exit_price,
                exit_prices.exit_open_price,
                entry_limits.entry_up_limit,
                entry_limits.entry_down_limit,
                exit_limits.exit_up_limit,
                exit_limits.exit_down_limit,
                benchmark_entry.benchmark_entry_price,
                benchmark_exit.benchmark_exit_price
            FROM base_frame
            LEFT JOIN entry_flags
              ON base_frame.ts_code = entry_flags.ts_code
             AND base_frame.entry_date = entry_flags.entry_date
            LEFT JOIN exit_flags
              ON base_frame.ts_code = exit_flags.ts_code
             AND base_frame.exit_date = exit_flags.exit_date
            LEFT JOIN entry_prices
              ON base_frame.ts_code = entry_prices.ts_code
             AND base_frame.entry_date = entry_prices.entry_date
            LEFT JOIN exit_prices
              ON base_frame.ts_code = exit_prices.ts_code
             AND base_frame.exit_date = exit_prices.exit_date
            LEFT JOIN entry_limits
              ON base_frame.ts_code = entry_limits.ts_code
             AND base_frame.entry_date = entry_limits.entry_date
            LEFT JOIN exit_limits
              ON base_frame.ts_code = exit_limits.ts_code
             AND base_frame.exit_date = exit_limits.exit_date
            LEFT JOIN benchmark_entry
              ON base_frame.entry_date = benchmark_entry.entry_date
            LEFT JOIN benchmark_exit
              ON base_frame.exit_date = benchmark_exit.exit_date
            """
        ).df()
    finally:
        connection.close()


def assign_label_availability(frame: DataFrame, settings: LabelSettings) -> DataFrame:
    """Assign label availability, unavailable reason, and return columns."""

    working = frame.copy()
    working["label_unavailable_reason"] = ""

    set_first_reason(working, working["entry_date"].eq(""), "insufficient_future_calendar")
    set_first_reason(working, working["exit_date"].eq(""), "insufficient_future_calendar")
    set_first_reason(
        working,
        working["entry_in_base_universe"].fillna(False).astype(bool).eq(False),
        "entry_not_in_base_universe",
    )
    set_first_reason(working, invalid_positive_price(working["entry_price"]), "missing_entry_price")
    set_first_reason(working, invalid_positive_price(working["exit_price"]), "missing_exit_price")
    set_first_reason(
        working,
        working["entry_is_suspended"].fillna(False).astype(bool),
        "entry_suspended",
    )
    if settings.skip_unbuyable_entry:
        entry_buyable = entry_open_buyable(working, settings)
        set_first_reason(working, entry_buyable.eq(False), "entry_not_buyable")

    set_first_reason(
        working,
        working["exit_in_base_universe"].fillna(False).astype(bool).eq(False),
        "exit_not_in_base_universe",
    )
    set_first_reason(
        working,
        working["exit_is_suspended"].fillna(False).astype(bool),
        "exit_suspended",
    )
    exit_sellable = exit_open_sellable(working, settings)
    set_first_reason(working, exit_sellable.eq(False), "exit_not_sellable")

    benchmark_missing = invalid_positive_price(
        working["benchmark_entry_price"]
    ) | invalid_positive_price(working["benchmark_exit_price"])
    set_first_reason(working, benchmark_missing, "missing_benchmark_price")

    available = working["label_unavailable_reason"].eq("")
    working["is_label_available"] = available
    working["stock_forward_ret"] = pd.NA
    working["benchmark_forward_ret"] = pd.NA
    working["future_excess_ret"] = pd.NA
    if bool(available.any()):
        stock_forward_ret = working.loc[available, "exit_price"] / working.loc[
            available, "entry_price"
        ] - 1.0
        benchmark_forward_ret = working.loc[available, "benchmark_exit_price"] / working.loc[
            available, "benchmark_entry_price"
        ] - 1.0
        working.loc[available, "stock_forward_ret"] = stock_forward_ret
        working.loc[available, "benchmark_forward_ret"] = benchmark_forward_ret
        working.loc[available, "future_excess_ret"] = stock_forward_ret - benchmark_forward_ret

    working.loc[~available, ["entry_price", "exit_price"]] = pd.NA
    working["future_rank_pct"] = pd.NA
    working["future_quantile"] = pd.NA
    return working


def set_first_reason(frame: DataFrame, mask: pd.Series, reason: str) -> None:
    """Set an unavailable reason only for rows that do not already have one."""

    frame.loc[mask & frame["label_unavailable_reason"].eq(""), "label_unavailable_reason"] = reason


def invalid_positive_price(values: pd.Series) -> pd.Series:
    """Return true where a price is missing, non-numeric, or non-positive."""

    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.isna() | numeric.le(0)


def entry_open_buyable(frame: DataFrame, settings: LabelSettings) -> pd.Series:
    """Return whether entry can be bought at the entry-date open."""

    open_price = pd.to_numeric(frame["entry_open_price"], errors="coerce")
    up_limit = pd.to_numeric(frame["entry_up_limit"], errors="coerce")
    down_limit = pd.to_numeric(frame["entry_down_limit"], errors="coerce")
    suspended = frame["entry_is_suspended"].fillna(False).astype(bool)
    has_open = open_price.notna() & open_price.gt(0)
    usable_limit = usable_limit_prices(up_limit, down_limit)
    at_limit_up_open = usable_limit & open_price.ge(up_limit - settings.price_tolerance)
    if settings.allow_limit_up_entry:
        at_limit_up_open = pd.Series(False, index=frame.index)
    return has_open & ~suspended & ~at_limit_up_open


def exit_open_sellable(frame: DataFrame, settings: LabelSettings) -> pd.Series:
    """Return whether exit can be sold at the exit-date open."""

    open_price = pd.to_numeric(frame["exit_open_price"], errors="coerce")
    up_limit = pd.to_numeric(frame["exit_up_limit"], errors="coerce")
    down_limit = pd.to_numeric(frame["exit_down_limit"], errors="coerce")
    suspended = frame["exit_is_suspended"].fillna(False).astype(bool)
    has_open = open_price.notna() & open_price.gt(0)
    usable_limit = usable_limit_prices(up_limit, down_limit)
    at_limit_down_open = usable_limit & open_price.le(down_limit + settings.price_tolerance)
    if settings.allow_limit_down_exit:
        at_limit_down_open = pd.Series(False, index=frame.index)
    return has_open & ~suspended & ~at_limit_down_open


def usable_limit_prices(up_limit: pd.Series, down_limit: pd.Series) -> pd.Series:
    """Return rows with meaningful daily limit prices."""

    no_price_limit = up_limit.eq(99999.99) & down_limit.eq(0)
    return (
        up_limit.notna()
        & down_limit.notna()
        & up_limit.gt(0)
        & down_limit.gt(0)
        & ~no_price_limit
    )


def build_one_label(
    trade_date: str,
    ts_code: str,
    horizon: int,
    trade_dates: list[str],
    prices: DataFrame,
    limits: DataFrame,
    benchmark: DataFrame,
    universe: DataFrame,
    settings: LabelSettings,
) -> dict[str, object]:
    """Build one stock/date/horizon label row."""

    entry_date = nth_next_trade_date(trade_dates, trade_date, 1)
    if entry_date is None:
        return unavailable_row(trade_date, ts_code, horizon, "", "", "insufficient_future_calendar")
    exit_date = nth_next_trade_date(trade_dates, entry_date, horizon)
    if exit_date is None:
        return unavailable_row(
            trade_date, ts_code, horizon, entry_date, "", "insufficient_future_calendar"
        )

    entry_check = check_entry_tradable(universe, prices, limits, ts_code, entry_date, settings)
    if entry_check is not None:
        return unavailable_row(trade_date, ts_code, horizon, entry_date, exit_date, entry_check)

    final_exit_date = exit_date
    exit_check = check_exit_tradable(universe, prices, limits, ts_code, final_exit_date, settings)
    if exit_check is not None and settings.delay_unsellable_exit:
        delayed = find_delayed_exit_date(
            trade_dates=trade_dates,
            ts_code=ts_code,
            start_exit_date=exit_date,
            prices=prices,
            limits=limits,
            universe=universe,
            settings=settings,
        )
        if delayed is not None:
            final_exit_date = delayed
            exit_check = None
    if exit_check is not None:
        return unavailable_row(
            trade_date, ts_code, horizon, entry_date, final_exit_date, exit_check
        )

    entry_price = lookup_price(prices, ts_code, entry_date)
    exit_price = lookup_price(prices, ts_code, final_exit_date)
    if entry_price is None:
        return unavailable_row(
            trade_date, ts_code, horizon, entry_date, final_exit_date, "missing_entry_price"
        )
    if exit_price is None:
        return unavailable_row(
            trade_date, ts_code, horizon, entry_date, final_exit_date, "missing_exit_price"
        )

    benchmark_entry = lookup_benchmark_price(benchmark, entry_date)
    benchmark_exit = lookup_benchmark_price(benchmark, final_exit_date)
    if benchmark_entry is None or benchmark_exit is None:
        return unavailable_row(
            trade_date, ts_code, horizon, entry_date, final_exit_date, "missing_benchmark_price"
        )

    stock_forward_ret = exit_price / entry_price - 1.0
    benchmark_forward_ret = benchmark_exit / benchmark_entry - 1.0
    return {
        "trade_date": trade_date,
        "ts_code": ts_code,
        "horizon": horizon,
        "entry_date": entry_date,
        "exit_date": final_exit_date,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stock_forward_ret": stock_forward_ret,
        "benchmark_forward_ret": benchmark_forward_ret,
        "future_excess_ret": stock_forward_ret - benchmark_forward_ret,
        "future_rank_pct": pd.NA,
        "future_quantile": pd.NA,
        "is_label_available": True,
        "label_unavailable_reason": "",
    }


def open_trade_dates(
    trade_cal: DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> list[str]:
    """Return authoritative open trading dates from trade_cal."""

    required = {"cal_date", "is_open"}
    if trade_cal.empty or not required.issubset(trade_cal.columns):
        raise DataValidationError("trade_cal with cal_date and is_open is required")
    calendar = trade_cal.copy()
    calendar["cal_date"] = calendar["cal_date"].astype(str)
    calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="coerce").fillna(0).astype(int)
    mask = calendar["is_open"] == 1
    if start_date is not None:
        mask &= calendar["cal_date"] >= start_date
    if end_date is not None:
        mask &= calendar["cal_date"] <= end_date
    return sorted(calendar.loc[mask, "cal_date"].drop_duplicates().astype(str).tolist())


def future_calendar_end(trade_cal: DataFrame, end_date: str, future_open_days: int) -> str:
    """Return the latest date needed to cover future horizons if available."""

    dates = open_trade_dates(trade_cal, None, None)
    if not dates:
        return end_date
    if end_date not in dates:
        candidates = [date for date in dates if date > end_date]
        if not candidates:
            return dates[-1]
        end_date = candidates[0]
    position = dates.index(end_date)
    return dates[min(position + future_open_days, len(dates) - 1)]


def nth_next_trade_date(trade_dates: list[str], date: str, n: int) -> str | None:
    """Return the nth open trading day after a date."""

    later_dates = [trade_date for trade_date in trade_dates if trade_date > date]
    if len(later_dates) < n:
        return None
    return later_dates[n - 1]


def adjusted_stock_open_prices(daily: DataFrame, adj_factor: DataFrame) -> DataFrame:
    """Return stock open prices adjusted as open * adj_factor."""

    if daily.empty or adj_factor.empty:
        return pd.DataFrame(columns=["ts_code", "trade_date", "adjusted_open"])
    required_daily = {"ts_code", "trade_date", "open"}
    required_adj = {"ts_code", "trade_date", "adj_factor"}
    if not required_daily.issubset(daily.columns) or not required_adj.issubset(adj_factor.columns):
        raise DataValidationError("daily open and adj_factor are required for labels")
    daily_work = daily[["ts_code", "trade_date", "open"]].copy()
    daily_work["ts_code"] = daily_work["ts_code"].astype(str)
    daily_work["trade_date"] = daily_work["trade_date"].astype(str)
    daily_work["open"] = pd.to_numeric(daily_work["open"], errors="coerce")
    adj_work = adj_factor[["ts_code", "trade_date", "adj_factor"]].copy()
    adj_work["ts_code"] = adj_work["ts_code"].astype(str)
    adj_work["trade_date"] = adj_work["trade_date"].astype(str)
    adj_work["adj_factor"] = pd.to_numeric(adj_work["adj_factor"], errors="coerce")
    merged = daily_work.merge(adj_work, on=["ts_code", "trade_date"], how="left")
    merged["adjusted_open"] = merged["open"] * merged["adj_factor"]
    return merged[["ts_code", "trade_date", "open", "adjusted_open"]].drop_duplicates(
        subset=["ts_code", "trade_date"], keep="last"
    )


def limit_prices(stk_limit: DataFrame) -> DataFrame:
    """Return raw daily limit prices used for open execution checks."""

    if stk_limit.empty or not {"ts_code", "trade_date"}.issubset(stk_limit.columns):
        return pd.DataFrame(columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    frame = stk_limit.copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = frame["trade_date"].astype(str)
    for column in ("up_limit", "down_limit"):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["ts_code", "trade_date", "up_limit", "down_limit"]].drop_duplicates(
        subset=["ts_code", "trade_date"], keep="last"
    )


def benchmark_open_prices(index_daily: DataFrame, benchmark_index_code: str) -> DataFrame:
    """Return benchmark open prices for one configured index code."""

    if index_daily.empty or not {"ts_code", "trade_date", "open"}.issubset(index_daily.columns):
        raise DataValidationError(
            "index_daily with ts_code/trade_date/open is required for benchmark labels"
        )
    frame = index_daily[index_daily["ts_code"].astype(str) == benchmark_index_code].copy()
    if frame.empty:
        available = sorted(index_daily["ts_code"].dropna().astype(str).unique().tolist())
        raise DataValidationError(
            f"benchmark index {benchmark_index_code} is missing from index_daily; "
            f"available_index_codes={available}"
        )
    frame["trade_date"] = frame["trade_date"].astype(str)
    frame["benchmark_open"] = pd.to_numeric(frame["open"], errors="coerce")
    return frame[["trade_date", "benchmark_open"]].drop_duplicates(
        subset=["trade_date"], keep="last"
    )


def normalize_universe(universe: DataFrame) -> DataFrame:
    """Normalize universe flags needed by label execution checks."""

    required = {"trade_date", "ts_code", "in_base_universe", "can_buy", "can_sell"}
    if universe.empty or not required.issubset(universe.columns):
        raise DataValidationError(
            "universe_daily with in_base_universe/can_buy/can_sell is required"
        )
    working = universe.copy()
    working["trade_date"] = working["trade_date"].astype(str)
    working["ts_code"] = working["ts_code"].astype(str)
    for column in ("in_base_universe", "can_buy", "can_sell"):
        working[column] = working[column].fillna(False).astype(bool)
    for column in ("is_limit_up", "is_limit_down", "is_suspended"):
        if column not in working.columns:
            working[column] = False
        working[column] = working[column].fillna(False).astype(bool)
    return working.drop_duplicates(subset=["trade_date", "ts_code"], keep="last")


def check_entry_tradable(
    universe: DataFrame,
    prices: DataFrame,
    limits: DataFrame,
    ts_code: str,
    entry_date: str,
    settings: LabelSettings,
) -> str | None:
    """Return an entry unavailable reason, or None when entry is executable."""

    row = universe_row(universe, ts_code, entry_date)
    if row is None or not bool(row["in_base_universe"]):
        return "entry_not_in_base_universe"
    if not settings.skip_unbuyable_entry:
        return None
    if bool(row["is_suspended"]):
        return "entry_suspended"
    open_price = lookup_raw_open(prices, ts_code, entry_date)
    if open_price is None:
        return "missing_entry_price"
    up_limit, down_limit = usable_scalar_limit_prices(limits, ts_code, entry_date)
    if (
        not settings.allow_limit_up_entry
        and up_limit is not None
        and open_price >= up_limit - settings.price_tolerance
    ):
        return "entry_not_buyable"
    return None


def check_exit_tradable(
    universe: DataFrame,
    prices: DataFrame,
    limits: DataFrame,
    ts_code: str,
    exit_date: str,
    settings: LabelSettings,
) -> str | None:
    """Return an exit unavailable reason, or None when exit is executable."""

    row = universe_row(universe, ts_code, exit_date)
    if row is None or not bool(row["in_base_universe"]):
        return "exit_not_in_base_universe"
    if bool(row["is_suspended"]):
        return "exit_suspended"
    open_price = lookup_raw_open(prices, ts_code, exit_date)
    if open_price is None:
        return "missing_exit_price"
    _, down_limit = usable_scalar_limit_prices(limits, ts_code, exit_date)
    if (
        not settings.allow_limit_down_exit
        and down_limit is not None
        and open_price <= down_limit + settings.price_tolerance
    ):
        return "exit_not_sellable"
    return None


def find_delayed_exit_date(
    trade_dates: list[str],
    ts_code: str,
    start_exit_date: str,
    prices: DataFrame,
    limits: DataFrame,
    universe: DataFrame,
    settings: LabelSettings,
) -> str | None:
    """Find the next executable sell date after an untradeable planned exit."""

    candidates = [date for date in trade_dates if date > start_exit_date]
    for delay, candidate in enumerate(candidates, start=1):
        if delay > settings.max_exit_delay_days:
            return None
        if check_exit_tradable(universe, prices, limits, ts_code, candidate, settings) is None:
            return candidate
    return None


def universe_row(universe: DataFrame, ts_code: str, trade_date: str) -> pd.Series | None:
    """Return one universe row for a stock/date, if available."""

    match = universe[(universe["ts_code"] == ts_code) & (universe["trade_date"] == trade_date)]
    if match.empty:
        return None
    return match.iloc[-1]


def lookup_price(prices: DataFrame, ts_code: str, trade_date: str) -> float | None:
    """Lookup one adjusted stock open price."""

    match = prices[(prices["ts_code"] == ts_code) & (prices["trade_date"] == trade_date)]
    if match.empty:
        return None
    value = pd.to_numeric(match.iloc[-1]["adjusted_open"], errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return None
    return float(value)


def lookup_raw_open(prices: DataFrame, ts_code: str, trade_date: str) -> float | None:
    """Lookup one unadjusted stock open price."""

    match = prices[(prices["ts_code"] == ts_code) & (prices["trade_date"] == trade_date)]
    if match.empty or "open" not in match.columns:
        return None
    value = pd.to_numeric(match.iloc[-1]["open"], errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return None
    return float(value)


def lookup_limit_prices(
    limits: DataFrame, ts_code: str, trade_date: str
) -> tuple[float | None, float | None]:
    """Lookup raw up/down limit prices for one stock/date."""

    match = limits[(limits["ts_code"] == ts_code) & (limits["trade_date"] == trade_date)]
    if match.empty:
        return None, None
    up_limit = pd.to_numeric(pd.Series([match.iloc[-1].get("up_limit")]), errors="coerce").iloc[0]
    down_limit = pd.to_numeric(
        pd.Series([match.iloc[-1].get("down_limit")]), errors="coerce"
    ).iloc[0]
    return (
        None if pd.isna(up_limit) else float(up_limit),
        None if pd.isna(down_limit) else float(down_limit),
    )


def limit_prices_are_usable(up_limit: float | None, down_limit: float | None) -> bool:
    """Return whether scalar limit prices are meaningful execution constraints."""

    if up_limit is None or down_limit is None:
        return False
    return up_limit > 0 and down_limit > 0 and not (up_limit == 99999.99 and down_limit == 0)


def usable_scalar_limit_prices(
    limits: DataFrame, ts_code: str, trade_date: str
) -> tuple[float | None, float | None]:
    """Lookup scalar limit prices and discard non-usable sentinels."""

    up_limit, down_limit = lookup_limit_prices(limits, ts_code, trade_date)
    if not limit_prices_are_usable(up_limit, down_limit):
        return None, None
    return up_limit, down_limit


def lookup_benchmark_price(benchmark: DataFrame, trade_date: str) -> float | None:
    """Lookup one benchmark open price."""

    match = benchmark[benchmark["trade_date"] == trade_date]
    if match.empty:
        return None
    value = pd.to_numeric(match.iloc[-1]["benchmark_open"], errors="coerce")
    if pd.isna(value) or float(value) <= 0:
        return None
    return float(value)


def unavailable_row(
    trade_date: str,
    ts_code: str,
    horizon: int,
    entry_date: str,
    exit_date: str,
    reason: str,
) -> dict[str, object]:
    """Return a label row with unavailable target values."""

    return {
        "trade_date": trade_date,
        "ts_code": ts_code,
        "horizon": horizon,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": pd.NA,
        "exit_price": pd.NA,
        "stock_forward_ret": pd.NA,
        "benchmark_forward_ret": pd.NA,
        "future_excess_ret": pd.NA,
        "future_rank_pct": pd.NA,
        "future_quantile": pd.NA,
        "is_label_available": False,
        "label_unavailable_reason": reason,
    }


def add_ranking_labels(frame: DataFrame, quantile_buckets: int) -> DataFrame:
    """Add cross-sectional rank percentiles and integer quantile buckets."""

    if frame.empty:
        return frame
    working = frame.copy()
    available = working["is_label_available"].astype(bool)
    working["future_rank_pct"] = pd.NA
    working["future_quantile"] = pd.NA
    for _, index in working.loc[available].groupby(["trade_date", "horizon"]).groups.items():
        values = pd.to_numeric(working.loc[index, "future_excess_ret"], errors="coerce")
        valid_index = values.dropna().index
        if valid_index.empty:
            continue
        rank_pct = values.loc[valid_index].rank(method="average", pct=True)
        quantile = ((rank_pct - 1e-12) * quantile_buckets).astype(int).clip(0, quantile_buckets - 1)
        working.loc[valid_index, "future_rank_pct"] = rank_pct
        working.loc[valid_index, "future_quantile"] = quantile
    return working
