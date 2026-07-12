"""Point-in-time candidate feature matrix construction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.features.fundamentals import build_fundamental_features
from ashare_quant.features.market import build_market_features
from ashare_quant.features.registry import FEATURE_REGISTRY, FeatureSpec
from ashare_quant.features.storage import FeatureStore
from ashare_quant.universe import UniverseStore

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    """Summary of one feature build run."""

    start_date: str
    end_date: str
    rows_built: int
    rows_written: int
    feature_count: int
    elapsed_seconds: float
    missing_value_stats: dict[str, float]


class FeatureBuilder:
    """Build point-in-time daily features from raw data and daily universe rows."""

    def __init__(
        self,
        raw_store: ParquetDataStore,
        universe_store: UniverseStore,
        feature_store: FeatureStore,
        settings: AppSettings,
    ) -> None:
        self._raw_store = raw_store
        self._universe_store = universe_store
        self._feature_store = feature_store
        self._settings = settings

    def build(self, start_date: str, end_date: str) -> FeatureBuildResult:
        """Build and persist daily feature rows."""

        started = time.perf_counter()
        rows_built = 0
        rows_written = 0
        missing_counts: dict[str, int] = {}
        observed_counts: dict[str, int] = {}
        feature_count = len(registered_feature_names(FEATURE_REGISTRY))
        for chunk_start, chunk_end in month_date_ranges(start_date, end_date):
            frame = self.preview(chunk_start, chunk_end)
            rows_built += len(frame)
            rows_written += self._feature_store.write(frame)
            feature_columns = feature_column_names(frame)
            feature_count = len(feature_columns) if feature_columns else feature_count
            for column in feature_columns:
                missing_counts[column] = missing_counts.get(column, 0) + int(
                    frame[column].isna().sum()
                )
                observed_counts[column] = observed_counts.get(column, 0) + len(frame)
        elapsed = time.perf_counter() - started
        missing_stats = {
            column: missing_counts[column] / observed_counts[column]
            for column in sorted(missing_counts)
            if observed_counts[column] > 0
        }
        return FeatureBuildResult(
            start_date=start_date,
            end_date=end_date,
            rows_built=rows_built,
            rows_written=rows_written,
            feature_count=feature_count,
            elapsed_seconds=elapsed,
            missing_value_stats=missing_stats,
        )

    def preview(self, start_date: str, end_date: str) -> DataFrame:
        """Build feature rows in memory without writing."""

        inputs = self._load_inputs(start_date, end_date)
        return build_feature_frame(inputs, self._settings, start_date, end_date)

    def _load_inputs(self, start_date: str, end_date: str) -> dict[str, DataFrame]:
        history_start = feature_history_start(start_date)
        dated_names = ("daily", "adj_factor", "daily_basic", "index_daily", "trade_cal")
        inputs = {
            name: self._raw_store.read_dataset(get_dataset_spec(name), history_start, end_date)
            for name in dated_names
        }
        financial_names = ("fina_indicator", "income", "balancesheet", "cashflow")
        inputs.update(
            {
                name: self._raw_store.read_dataset(get_dataset_spec(name), None, end_date)
                for name in financial_names
            }
        )
        inputs["universe"] = self._universe_store.read(history_start, end_date)
        return inputs


def build_feature_frame(
    inputs: dict[str, DataFrame],
    settings: AppSettings,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Build a stable feature matrix for an inclusive trade-date range."""

    universe = inputs["universe"]
    market = build_market_features(
        daily=inputs["daily"],
        adj_factor=inputs["adj_factor"],
        daily_basic=inputs["daily_basic"],
        index_daily=inputs["index_daily"],
        trade_cal=inputs["trade_cal"],
        universe=universe,
        settings=settings.features,
    )
    if market.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code"])
    market["trade_date"] = market["trade_date"].astype(str)
    market = market[
        (market["trade_date"] >= start_date) & (market["trade_date"] <= end_date)
    ].copy()

    keys = market[["trade_date", "ts_code"]].drop_duplicates()
    if settings.features.include_fundamentals:
        fundamentals = build_fundamental_features(
            keys,
            fina_indicator=inputs["fina_indicator"],
            income=inputs["income"],
            balancesheet=inputs["balancesheet"],
            cashflow=inputs["cashflow"],
        )
        market = market.merge(
            fundamentals, on=["trade_date", "ts_code"], how="left", suffixes=("", "_fin")
        )
        for column in fundamentals.columns:
            if column.endswith("_fin"):
                base = column.removesuffix("_fin")
                if base not in market.columns:
                    market[base] = market[column]

    output_columns = ["trade_date", "ts_code", *registered_feature_names(FEATURE_REGISTRY)]
    for column in output_columns:
        if column not in market.columns:
            market[column] = pd.NA
    return market[output_columns].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def registered_feature_names(registry: tuple[FeatureSpec, ...]) -> list[str]:
    """Return registered feature names in declaration order."""

    return [spec.name for spec in registry]


def feature_column_names(frame: DataFrame) -> list[str]:
    """Return non-key feature columns."""

    return [column for column in frame.columns if column not in {"trade_date", "ts_code"}]


def missing_value_stats(frame: DataFrame, feature_columns: list[str]) -> dict[str, float]:
    """Return missing ratios for feature columns."""

    if frame.empty:
        return {}
    return {column: float(frame[column].isna().mean()) for column in feature_columns}


def month_date_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Return inclusive calendar-month chunks covering YYYYMMDD dates."""

    start = pd.Period(start_date[:6], freq="M")
    end = pd.Period(end_date[:6], freq="M")
    ranges: list[tuple[str, str]] = []
    for period in pd.period_range(start, end, freq="M"):
        first = max(start_date, period.start_time.strftime("%Y%m%d"))
        last = min(end_date, period.end_time.strftime("%Y%m%d"))
        ranges.append((first, last))
    return ranges


def feature_history_start(start_date: str) -> str:
    """Return a conservative raw-data start date for rolling feature warm-up."""

    start = pd.to_datetime(start_date, format="%Y%m%d")
    return (start - timedelta(days=550)).strftime("%Y%m%d")
