"""Data loading for executable backtests."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from ashare_quant.backtest.engine import BacktestInputs
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import (
    SecurityIdentityTransitionResolver,
    load_identity_transition_contract,
)
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver

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
    transitions = load_identity_transition_contract(
        mode=settings.security_identity.identity_transition_mode,
        artifact_path=settings.security_identity.identity_transition_path,
    )
    prices = load_execution_prices(
        raw_root,
        processed_root,
        price_start,
        price_end,
        settings.universe.price_tolerance,
        identity_resolver=SecurityIdentityResolver.from_path(
            settings.security_identity.mapping_path
        ),
        lifecycle_resolver=SecurityLifecycleResolver.from_path(
            settings.security_identity.lifecycle_path
        ),
        identity_transitions=transitions,
    )
    benchmark = load_benchmark(
        raw_root, settings.backtest.benchmark_index_code, price_start, price_end
    )
    return BacktestInputs(
        signals=signals,
        prices=prices,
        calendar=tuple(calendar),
        benchmark=benchmark,
        identity_transitions=transitions.transition_records(),
        identity_transition_version=transitions.artifact_version,
        identity_transition_hash=transitions.artifact_hash,
    )


def load_calendar(
    raw_root: Path,
    start_date: str,
    end_date: str,
    holding_period: int | None,
    *,
    maximum_date: str | None = None,
) -> list[str]:
    """Return open trading dates from start through the required exit buffer."""

    glob = raw_root / "trade_cal" / "**" / "*.parquet"
    maximum_clause = "AND CAST(cal_date AS VARCHAR) <= ?" if maximum_date is not None else ""
    query = f"""
        SELECT CAST(cal_date AS VARCHAR) AS trade_date
        FROM read_parquet('{glob.as_posix()}', hive_partitioning=false)
        WHERE CAST(is_open AS INTEGER) = 1
          AND CAST(cal_date AS VARCHAR) >= ?
          {maximum_clause}
        ORDER BY cal_date
    """  # noqa: S608 -- local configured Parquet path
    parameters = [start_date] if maximum_date is None else [start_date, maximum_date]
    with duckdb.connect() as connection:
        frame = connection.execute(query, parameters).fetch_df()
    dates = frame["trade_date"].astype(str).tolist()
    if end_date not in dates:
        dates = [date for date in dates if date <= end_date]
        if not dates:
            return []
        end_index = len(dates) - 1
    else:
        end_index = dates.index(end_date)
    if holding_period is None:
        if maximum_date is None:
            raise DataValidationError(
                "BACKTEST_EXECUTION_DATA_CUTOFF_INVALID: unbounded calendar extension"
            )
        return dates
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
    *,
    identity_resolver: SecurityIdentityResolver | None = None,
    lifecycle_resolver: SecurityLifecycleResolver | None = None,
    identity_transitions: SecurityIdentityTransitionResolver,
    ts_codes: Collection[str] | None = None,
) -> DataFrame:
    """Load next-open tradability fields without using label outputs."""

    daily_glob = raw_root / "daily" / "**" / "*.parquet"
    limit_glob = raw_root / "stk_limit" / "**" / "*.parquet"
    universe_glob = processed_root / "universe_daily" / "**" / "*.parquet"
    resolver = identity_resolver or SecurityIdentityResolver.empty()
    canonical_codes: tuple[str, ...] = ()
    if ts_codes is not None:
        canonical_codes = identity_transitions.execution_code_closure(
            {str(code).strip().upper() for code in ts_codes},
            end_date=end_date,
        )
    if ts_codes is not None and not canonical_codes:
        raise DataValidationError("BACKTEST_MARKET_DATA_INCOMPLETE: execution code set is empty")
    universe_code_join = (
        "INNER JOIN selected_canonical_codes AS selected "
        "ON CAST(u.ts_code AS VARCHAR) = selected.ts_code"
        if canonical_codes
        else ""
    )
    raw_code_join = (
        "INNER JOIN selected_source_codes AS selected "
        "ON CAST(source.ts_code AS VARCHAR) = selected.ts_code"
        if canonical_codes
        else ""
    )
    universe_query = f"""
        SELECT
            CAST(u.trade_date AS VARCHAR) AS trade_date,
            CAST(u.ts_code AS VARCHAR) AS ts_code,
            CAST(u.is_suspended AS BOOLEAN) AS is_suspended,
            CAST(u.is_st AS BOOLEAN) AS is_st,
            CAST(u.is_listed AS BOOLEAN) AS is_listed,
            CAST(u.delist_date AS VARCHAR) AS delist_date
        FROM read_parquet('{universe_glob.as_posix()}', hive_partitioning=false) AS u
        {universe_code_join}
        WHERE CAST(u.trade_date AS VARCHAR) BETWEEN ? AND ?
        ORDER BY u.trade_date, u.ts_code
    """  # noqa: S608 -- local configured Parquet path
    daily_query = f"""
        SELECT CAST(source.trade_date AS VARCHAR) AS trade_date,
               CAST(source.ts_code AS VARCHAR) AS ts_code,
               CAST(source.open AS DOUBLE) AS open,
               CAST(source.close AS DOUBLE) AS close
        FROM read_parquet('{daily_glob.as_posix()}', hive_partitioning=false) AS source
        {raw_code_join}
        WHERE CAST(source.trade_date AS VARCHAR) BETWEEN ? AND ?
    """  # noqa: S608 -- local configured Parquet path
    limit_query = f"""
        SELECT CAST(source.trade_date AS VARCHAR) AS trade_date,
               CAST(source.ts_code AS VARCHAR) AS ts_code,
               CAST(source.up_limit AS DOUBLE) AS up_limit,
               CAST(source.down_limit AS DOUBLE) AS down_limit
        FROM read_parquet('{limit_glob.as_posix()}', hive_partitioning=false) AS source
        {raw_code_join}
        WHERE CAST(source.trade_date AS VARCHAR) BETWEEN ? AND ?
    """  # noqa: S608 -- local configured Parquet path
    with duckdb.connect() as connection:
        if canonical_codes:
            connection.register(
                "selected_canonical_codes", pd.DataFrame({"ts_code": canonical_codes})
            )
            connection.register(
                "selected_source_codes",
                pd.DataFrame({"ts_code": resolver.source_codes_for(set(canonical_codes))}),
            )
        frame = connection.execute(universe_query, [start_date, end_date]).fetch_df()
        daily = connection.execute(daily_query, [start_date, end_date]).fetch_df()
        limits = connection.execute(limit_query, [start_date, end_date]).fetch_df()
    if frame.empty:
        raise DataValidationError(f"no daily prices for backtest {start_date}..{end_date}")
    daily = resolver.canonicalize_frame(daily, "daily")
    limits = resolver.canonicalize_frame(limits, "stk_limit")
    frame = frame.merge(
        daily[["trade_date", "ts_code", "open", "close"]],
        on=["trade_date", "ts_code"],
        how="left",
    ).merge(
        limits[["trade_date", "ts_code", "up_limit", "down_limit"]],
        on=["trade_date", "ts_code"],
        how="left",
    )
    frame["is_suspended"] = frame["is_suspended"].fillna(False).astype(bool)
    frame = (lifecycle_resolver or SecurityLifecycleResolver.empty()).apply_suspension(frame)
    frame["is_st"] = frame["is_st"].fillna(False).astype(bool)
    frame["is_listed"] = frame["is_listed"].fillna(False).astype(bool)
    delist_dates = frame["delist_date"].astype("string")
    terminal_effective = delist_dates.notna() & (frame["trade_date"] >= delist_dates)
    frame.loc[terminal_effective, "is_listed"] = False
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
