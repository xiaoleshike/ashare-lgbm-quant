"""Data loading for executable backtests."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from ashare_quant.backtest.engine import BacktestInputs
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame


def load_model_and_features(model_dir: Path) -> tuple[lgb.Booster, tuple[str, ...], str]:
    """Load one saved Ranker model and its feature list."""

    model_path = model_dir / "model.txt"
    feature_path = model_dir / "feature_list.json"
    if not model_path.exists() or not feature_path.exists():
        raise DataValidationError(f"model.txt and feature_list.json are required in {model_dir}")
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    features = tuple(str(value) for value in payload.get("features", ()))
    if not features:
        raise DataValidationError(f"feature_list.json does not contain features: {feature_path}")
    feature_hash = str(payload.get("feature_hash", ""))
    return lgb.Booster(model_file=str(model_path)), features, feature_hash


def load_backtest_inputs(
    *,
    raw_root: Path,
    processed_root: Path,
    model: lgb.Booster,
    feature_names: tuple[str, ...],
    start_date: str,
    end_date: str,
    settings: AppSettings,
) -> BacktestInputs:
    """Load scores, execution prices, constraints, calendar, and benchmark data."""

    calendar = load_calendar(
        raw_root,
        start_date,
        end_date,
        settings.backtest.holding_period_days + settings.backtest.sell_delay_max_days,
    )
    if not calendar:
        raise DataValidationError(f"no open trading calendar for {start_date}..{end_date}")
    price_start = min(start_date, calendar[0])
    price_end = calendar[-1]
    signals = load_scored_signals(processed_root, model, feature_names, start_date, end_date)
    prices = load_execution_prices(
        raw_root,
        processed_root,
        price_start,
        price_end,
        settings.universe.price_tolerance,
    )
    benchmark = load_benchmark(
        raw_root, settings.backtest.benchmark_index_code, price_start, price_end
    )
    return BacktestInputs(
        signals=signals,
        prices=prices,
        calendar=tuple(calendar),
        benchmark=benchmark,
    )


def load_calendar(raw_root: Path, start_date: str, end_date: str, holding_period: int) -> list[str]:
    """Return open trading dates from start through the required exit buffer."""

    glob = raw_root / "trade_cal" / "**" / "*.parquet"
    query = f"""
        SELECT CAST(cal_date AS VARCHAR) AS trade_date
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(is_open AS INTEGER) = 1
          AND CAST(cal_date AS VARCHAR) >= ?
        ORDER BY cal_date
    """  # noqa: S608 -- local configured Parquet path
    with duckdb.connect() as connection:
        frame = connection.execute(query, [start_date]).fetch_df()
    dates = frame["trade_date"].astype(str).tolist()
    if end_date not in dates:
        dates = [date for date in dates if date <= end_date]
        if not dates:
            return []
        end_index = len(dates) - 1
    else:
        end_index = dates.index(end_date)
    return dates[: min(len(dates), end_index + holding_period + 2)]


def load_scored_signals(
    processed_root: Path,
    model: lgb.Booster,
    feature_names: tuple[str, ...],
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Score in-model-universe stocks for every signal date."""

    selected = ",\n".join(f'f."{name}"' for name in feature_names)
    feature_glob = processed_root / "features_daily" / "**" / "*.parquet"
    universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
    query = f"""
        SELECT
            CAST(f.trade_date AS VARCHAR) AS trade_date,
            CAST(f.ts_code AS VARCHAR) AS ts_code,
            {selected}
        FROM read_parquet('{feature_glob.as_posix()}', hive_partitioning=false) AS f
        INNER JOIN read_parquet('{universe_glob.as_posix()}', hive_partitioning=false) AS u
            ON CAST(f.trade_date AS VARCHAR) = CAST(u.trade_date AS VARCHAR)
           AND CAST(f.ts_code AS VARCHAR) = CAST(u.ts_code AS VARCHAR)
        WHERE CAST(f.trade_date AS VARCHAR) BETWEEN ? AND ?
          AND CAST(u.in_model_universe AS BOOLEAN)
        ORDER BY f.trade_date, f.ts_code
    """  # noqa: S608 -- feature identifiers are validated by model feature_list.json
    with duckdb.connect() as connection:
        frame = connection.execute(query, [start_date, end_date]).fetch_df()
    if frame.empty:
        raise DataValidationError(f"no backtest signals for {start_date}..{end_date}")
    matrix = frame.loc[:, list(feature_names)].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).astype("float32")
    frame["score"] = model.predict(matrix)
    return frame[["trade_date", "ts_code", "score"]].reset_index(drop=True)


def load_execution_prices(
    raw_root: Path,
    processed_root: Path,
    start_date: str,
    end_date: str,
    tolerance: float,
) -> DataFrame:
    """Load next-open tradability fields without using label outputs."""

    daily_glob = raw_root / "daily" / "**" / "*.parquet"
    limit_glob = raw_root / "stk_limit" / "**" / "*.parquet"
    universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
    query = f"""
        SELECT
            CAST(u.trade_date AS VARCHAR) AS trade_date,
            CAST(u.ts_code AS VARCHAR) AS ts_code,
            CAST(d.open AS DOUBLE) AS open,
            CAST(d.close AS DOUBLE) AS close,
            CAST(s.up_limit AS DOUBLE) AS up_limit,
            CAST(s.down_limit AS DOUBLE) AS down_limit,
            CAST(u.is_suspended AS BOOLEAN) AS is_suspended,
            CAST(u.is_st AS BOOLEAN) AS is_st,
            CAST(u.is_listed AS BOOLEAN) AS is_listed,
            CAST(u.delist_date AS VARCHAR) AS delist_date
        FROM read_parquet('{universe_glob.as_posix()}', hive_partitioning=false) AS u
        LEFT JOIN read_parquet('{daily_glob.as_posix()}', hive_partitioning=false) AS d
            ON CAST(u.trade_date AS VARCHAR) = CAST(d.trade_date AS VARCHAR)
           AND CAST(u.ts_code AS VARCHAR) = CAST(d.ts_code AS VARCHAR)
        LEFT JOIN read_parquet('{limit_glob.as_posix()}', hive_partitioning=false) AS s
            ON CAST(u.trade_date AS VARCHAR) = CAST(s.trade_date AS VARCHAR)
           AND CAST(u.ts_code AS VARCHAR) = CAST(s.ts_code AS VARCHAR)
        WHERE CAST(u.trade_date AS VARCHAR) BETWEEN ? AND ?
        ORDER BY u.trade_date, u.ts_code
    """  # noqa: S608 -- local configured Parquet path
    with duckdb.connect() as connection:
        frame = connection.execute(query, [start_date, end_date]).fetch_df()
    if frame.empty:
        raise DataValidationError(f"no daily prices for backtest {start_date}..{end_date}")
    frame["is_suspended"] = frame["is_suspended"].fillna(False).astype(bool)
    frame["is_st"] = frame["is_st"].fillna(False).astype(bool)
    frame["is_listed"] = frame["is_listed"].fillna(False).astype(bool)
    frame["can_buy"] = (
        frame["is_listed"]
        & ~frame["is_suspended"]
        & ~frame["is_st"]
        & frame["open"].notna()
        & (frame["open"] > 0)
        & (frame["up_limit"].isna() | (frame["open"] < frame["up_limit"] - tolerance))
    )
    frame["can_sell"] = (
        frame["is_listed"]
        & ~frame["is_suspended"]
        & frame["open"].notna()
        & (frame["open"] > 0)
        & (frame["down_limit"].isna() | (frame["open"] > frame["down_limit"] + tolerance))
    )
    return frame[
        [
            "trade_date",
            "ts_code",
            "open",
            "close",
            "can_buy",
            "can_sell",
            "is_suspended",
            "is_listed",
            "delist_date",
        ]
    ]


def load_benchmark(raw_root: Path, index_code: str, start_date: str, end_date: str) -> DataFrame:
    """Load benchmark daily closes for relative reporting."""

    glob = raw_root / "index_daily" / "**" / "*.parquet"
    query = f"""
        SELECT CAST(trade_date AS VARCHAR) AS trade_date, CAST(close AS DOUBLE) AS close
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(ts_code AS VARCHAR) = ?
          AND CAST(trade_date AS VARCHAR) BETWEEN ? AND ?
        ORDER BY trade_date
    """  # noqa: S608 -- local configured Parquet path
    with duckdb.connect() as connection:
        return connection.execute(query, [index_code, start_date, end_date]).fetch_df()
