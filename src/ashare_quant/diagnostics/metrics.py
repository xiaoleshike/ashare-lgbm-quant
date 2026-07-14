"""Statistical diagnostics for cross-sectional feature research."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

type DataFrame = pd.DataFrame


def daily_ic_table(
    frame: DataFrame, feature_names: Sequence[str], minimum_cross_section: int
) -> DataFrame:
    """Compute daily Pearson and Spearman IC without pooling dates."""

    records: list[dict[str, object]] = []
    for trade_date, daily in frame.groupby("trade_date", sort=True):
        target = pd.to_numeric(daily["target"], errors="coerce")
        for feature in feature_names:
            values = pd.to_numeric(daily[feature], errors="coerce")
            valid = values.notna() & target.notna() & np.isfinite(values) & np.isfinite(target)
            count = int(valid.sum())
            if count < minimum_cross_section:
                continue
            x = values[valid]
            y = target[valid]
            pearson = x.corr(y, method="pearson")
            rank_ic = x.corr(y, method="spearman")
            records.append(
                {
                    "trade_date": str(trade_date),
                    "feature": feature,
                    "observations": count,
                    "pearson_ic": pearson,
                    "rank_ic": rank_ic,
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_ic(daily_ic: DataFrame, coverage: dict[str, float]) -> DataFrame:
    """Aggregate daily IC observations into robust feature-level statistics."""

    columns = [
        "feature",
        "coverage",
        "missing_ratio",
        "ic_days",
        "pearson_ic_mean",
        "pearson_ic_std",
        "pearson_icir",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_icir",
        "positive_ic_ratio",
    ]
    if daily_ic.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, object]] = []
    for feature, group in daily_ic.groupby("feature", sort=True):
        pearson_mean = float(group["pearson_ic"].mean())
        pearson_std = float(group["pearson_ic"].std(ddof=1))
        rank_mean = float(group["rank_ic"].mean())
        rank_std = float(group["rank_ic"].std(ddof=1))
        records.append(
            {
                "feature": feature,
                "coverage": coverage.get(str(feature), 0.0),
                "missing_ratio": 1.0 - coverage.get(str(feature), 0.0),
                "ic_days": len(group),
                "pearson_ic_mean": pearson_mean,
                "pearson_ic_std": pearson_std,
                "pearson_icir": safe_ratio(pearson_mean, pearson_std),
                "rank_ic_mean": rank_mean,
                "rank_ic_std": rank_std,
                "rank_icir": safe_ratio(rank_mean, rank_std),
                "positive_ic_ratio": float((group["rank_ic"] > 0).mean()),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def yearly_ic_statistics(daily_ic: DataFrame) -> DataFrame:
    """Report IC stability by calendar year."""

    if daily_ic.empty:
        return pd.DataFrame(
            columns=["feature", "year", "days", "pearson_ic_mean", "rank_ic_mean", "rank_icir"]
        )
    working = daily_ic.copy()
    working["year"] = working["trade_date"].astype(str).str[:4]
    records = []
    for (feature, year), group in working.groupby(["feature", "year"], sort=True):
        rank_mean = float(group["rank_ic"].mean())
        rank_std = float(group["rank_ic"].std(ddof=1))
        records.append(
            {
                "feature": feature,
                "year": year,
                "days": len(group),
                "pearson_ic_mean": float(group["pearson_ic"].mean()),
                "rank_ic_mean": rank_mean,
                "rank_icir": safe_ratio(rank_mean, rank_std),
            }
        )
    return pd.DataFrame.from_records(records)


def regime_ic_statistics(
    daily_ic: DataFrame, daily_benchmark: DataFrame, threshold: float
) -> DataFrame:
    """Summarize IC by ex-post benchmark-return regime for stability analysis."""

    if daily_ic.empty or daily_benchmark.empty:
        return pd.DataFrame(columns=["feature", "regime", "days", "rank_ic_mean", "rank_icir"])
    benchmark = daily_benchmark.copy()
    values = benchmark["benchmark_forward_ret"]
    benchmark["regime"] = np.select(
        [values > threshold, values < -threshold], ["bull", "bear"], default="neutral"
    )
    merged = daily_ic.merge(benchmark[["trade_date", "regime"]], on="trade_date", how="inner")
    records = []
    for (feature, regime), group in merged.groupby(["feature", "regime"], sort=True):
        rank_mean = float(group["rank_ic"].mean())
        rank_std = float(group["rank_ic"].std(ddof=1))
        records.append(
            {
                "feature": feature,
                "regime": regime,
                "days": len(group),
                "rank_ic_mean": rank_mean,
                "rank_icir": safe_ratio(rank_mean, rank_std),
            }
        )
    return pd.DataFrame.from_records(records)


def pairwise_correlations(frame: DataFrame, feature_names: Sequence[str]) -> DataFrame:
    """Return all finite upper-triangle Pearson feature correlations."""

    correlation = frame[list(feature_names)].corr(method="pearson", min_periods=3)
    records = []
    for left_index, left in enumerate(feature_names):
        for right in feature_names[left_index + 1 :]:
            value = correlation.loc[left, right]
            if pd.notna(value):
                records.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "correlation": float(str(value)),
                    }
                )
    return pd.DataFrame.from_records(records)


def greedy_correlation_prune(
    ordered_features: Sequence[str], correlations: DataFrame, threshold: float
) -> tuple[list[str], DataFrame]:
    """Greedily retain stronger features and document correlated removals."""

    lookup: dict[frozenset[str], float] = {}
    for _, row in correlations.iterrows():
        left = str(row["feature_left"])
        right = str(row["feature_right"])
        lookup[frozenset((left, right))] = float(str(row["correlation"]))
    kept: list[str] = []
    removals: list[dict[str, object]] = []
    for feature in ordered_features:
        conflict = next(
            (
                (existing, lookup.get(frozenset((feature, existing))))
                for existing in kept
                if abs(lookup.get(frozenset((feature, existing)), 0.0)) >= threshold
            ),
            None,
        )
        if conflict is None:
            kept.append(feature)
        else:
            removals.append(
                {
                    "removed_feature": feature,
                    "kept_feature": conflict[0],
                    "correlation": conflict[1],
                }
            )
    return kept, pd.DataFrame.from_records(removals)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or zero when dispersion is unavailable."""

    if not math.isfinite(denominator) or denominator == 0:
        return 0.0
    return numerator / denominator
