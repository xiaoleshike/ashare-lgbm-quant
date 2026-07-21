"""Pure statistical metrics for read-only model drift diagnostics."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

type DataFrame = pd.DataFrame
type FloatArray = NDArray[np.float64]


def fit_psi_edges(values: pd.Series, bins: int) -> np.ndarray:
    """Fit deterministic quantile edges from reference data only."""

    finite = _finite_values(values)
    if len(finite) == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    quantiles = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    internal = np.unique(np.quantile(finite, quantiles))
    return np.concatenate(([-np.inf], internal, [np.inf])).astype(float)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    edges: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Calculate PSI with an explicit final bucket for missing/non-finite values."""

    reference_distribution = _binned_distribution(reference, edges)
    current_distribution = _binned_distribution(current, edges)
    reference_smoothed = np.clip(reference_distribution, epsilon, None)
    current_smoothed = np.clip(current_distribution, epsilon, None)
    return float(
        np.sum(
            (current_smoothed - reference_smoothed) * np.log(current_smoothed / reference_smoothed)
        )
    )


def ks_statistic(reference: pd.Series, current: pd.Series) -> float | None:
    """Return two-sample Kolmogorov-Smirnov D without depending on SciPy."""

    left = np.sort(_finite_values(reference))
    right = np.sort(_finite_values(current))
    if len(left) == 0 or len(right) == 0:
        return None
    support = np.sort(np.concatenate((left, right)))
    left_cdf = np.searchsorted(left, support, side="right") / len(left)
    right_cdf = np.searchsorted(right, support, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def build_feature_drift(
    reference_sample: DataFrame,
    evaluation_sample: DataFrame,
    reference_coverage: DataFrame,
    evaluation_coverage: DataFrame,
    feature_names: Sequence[str],
    *,
    psi_bins: int,
) -> DataFrame:
    """Build monthly feature PSI, KS, missingness, and coverage drift rows."""

    records: list[dict[str, Any]] = []
    reference_lookup = reference_coverage.set_index("feature")
    evaluation_lookup = evaluation_coverage.set_index(["month", "feature"])
    months = sorted(evaluation_sample["month"].astype(str).unique())
    for feature in feature_names:
        reference_values = pd.to_numeric(reference_sample[feature], errors="coerce")
        edges = fit_psi_edges(reference_values, psi_bins)
        reference_row = cast(pd.Series, reference_lookup.loc[feature])
        for month in months:
            current_values = pd.to_numeric(
                evaluation_sample.loc[evaluation_sample["month"] == month, feature],
                errors="coerce",
            )
            coverage_row = cast(pd.Series, evaluation_lookup.loc[(month, feature)])
            reference_feature_coverage = float(reference_row["coverage"])
            current_coverage = float(coverage_row["coverage"])
            reference_missing = float(reference_row["missing_ratio"])
            current_missing = float(coverage_row["missing_ratio"])
            records.append(
                {
                    "month": month,
                    "feature": feature,
                    "reference_rows": int(reference_row["rows"]),
                    "current_rows": int(coverage_row["rows"]),
                    "reference_coverage": reference_feature_coverage,
                    "coverage": current_coverage,
                    "coverage_drift": current_coverage - reference_feature_coverage,
                    "reference_missing_ratio": reference_missing,
                    "missing_ratio": current_missing,
                    "missing_ratio_drift": current_missing - reference_missing,
                    "psi": population_stability_index(reference_values, current_values, edges),
                    "ks_statistic": ks_statistic(reference_values, current_values),
                    "reference_sample_rows": len(reference_values),
                    "current_sample_rows": len(current_values),
                    "psi_edges": json.dumps(edges.tolist(), separators=(",", ":")),
                }
            )
    return (
        pd.DataFrame.from_records(records).sort_values(["month", "feature"]).reset_index(drop=True)
    )


def build_score_drift(
    predictions: DataFrame,
    *,
    reference_months: int,
    psi_bins: int,
) -> tuple[DataFrame, tuple[str, ...]]:
    """Build monthly raw-score, percentile, concentration, and breadth diagnostics."""

    frame = predictions.copy()
    frame["month"] = frame["trade_date"].astype(str).str[:6]
    months = tuple(sorted(frame["month"].unique()))
    selected_reference_months = months[: min(reference_months, len(months))]
    reference = frame.loc[frame["month"].isin(selected_reference_months)]
    score_edges = fit_psi_edges(reference["prediction_score"], psi_bins)
    percentile_edges = fit_psi_edges(reference["score_percentile"], psi_bins)
    daily = _daily_score_concentration(frame)
    records: list[dict[str, Any]] = []
    for month in months:
        current = frame.loc[frame["month"] == month]
        current_daily = daily.loc[daily["month"] == month]
        score = pd.to_numeric(current["prediction_score"], errors="coerce")
        percentile = pd.to_numeric(current["score_percentile"], errors="coerce")
        records.append(
            {
                "month": month,
                "rows": len(current),
                "dates": int(current["trade_date"].nunique()),
                "score_mean": float(score.mean()),
                "score_std": float(score.std(ddof=1)),
                "score_p01": float(score.quantile(0.01)),
                "score_p10": float(score.quantile(0.10)),
                "score_p50": float(score.quantile(0.50)),
                "score_p90": float(score.quantile(0.90)),
                "score_p99": float(score.quantile(0.99)),
                "score_psi": population_stability_index(
                    reference["prediction_score"], score, score_edges
                ),
                "score_ks_statistic": ks_statistic(reference["prediction_score"], score),
                "percentile_mean": float(percentile.mean()),
                "percentile_std": float(percentile.std(ddof=1)),
                "percentile_psi": population_stability_index(
                    reference["score_percentile"], percentile, percentile_edges
                ),
                "percentile_ks_statistic": ks_statistic(reference["score_percentile"], percentile),
                "top1_concentration": float(current_daily["top1_concentration"].mean()),
                "top10_concentration": float(current_daily["top10_concentration"].mean()),
                "effective_breadth": float(current_daily["effective_breadth"].mean()),
                "normalized_breadth": float(current_daily["normalized_breadth"].mean()),
                "top1_score_spread": float(current_daily["top1_score_spread"].mean()),
                "top10_score_spread": float(current_daily["top10_score_spread"].mean()),
            }
        )
    return pd.DataFrame.from_records(records), selected_reference_months


def build_feature_response_drift(
    reference: DataFrame,
    evaluation: DataFrame,
    feature_names: Sequence[str],
    *,
    bucket_counts: Sequence[int],
    minimum_cross_section: int,
) -> DataFrame:
    """Measure monthly feature Rank IC and disjoint bucket-return response changes."""

    records: list[dict[str, Any]] = []
    months = sorted(evaluation["month"].astype(str).unique())
    for feature in feature_names:
        reference_ic = _mean_daily_rank_ic(reference, feature, minimum_cross_section)
        for month in months:
            current = evaluation.loc[evaluation["month"] == month]
            current_ic = _mean_daily_rank_ic(current, feature, minimum_cross_section)
            sign_change = (
                reference_ic is not None
                and current_ic is not None
                and np.sign(reference_ic) != 0
                and np.sign(current_ic) != 0
                and np.sign(reference_ic) != np.sign(current_ic)
            )
            for bucket_count in bucket_counts:
                bucket_returns = _daily_bucket_returns(current, feature, bucket_count)
                reference_bucket_returns = _daily_bucket_returns(reference, feature, bucket_count)
                spread = _bucket_spread(bucket_returns, bucket_count)
                reference_spread = _bucket_spread(reference_bucket_returns, bucket_count)
                monotonicity = _bucket_monotonicity(bucket_returns)
                for bucket in range(bucket_count):
                    records.append(
                        {
                            "month": month,
                            "feature": feature,
                            "bucket_count": bucket_count,
                            "bucket": bucket,
                            "rank_ic": current_ic,
                            "reference_rank_ic": reference_ic,
                            "ic_sign_change": bool(sign_change),
                            "mean_forward_excess_return": bucket_returns.get(bucket),
                            "reference_bucket_return": reference_bucket_returns.get(bucket),
                            "top_minus_bottom": spread,
                            "reference_top_minus_bottom": reference_spread,
                            "bucket_monotonicity": monotonicity,
                        }
                    )
    return (
        pd.DataFrame.from_records(records)
        .sort_values(["month", "feature", "bucket_count", "bucket"])
        .reset_index(drop=True)
    )


def _daily_score_concentration(frame: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for trade_date, daily in frame.groupby("trade_date", sort=True):
        ordered = daily.sort_values(
            ["prediction_score", "ts_code"], ascending=[False, True], kind="mergesort"
        )
        score = ordered["prediction_score"].to_numpy(dtype=float)
        weights = np.exp(score - np.max(score))
        weights /= weights.sum()
        count = len(ordered)
        top1 = max(1, math.ceil(count * 0.01))
        top10 = max(1, math.ceil(count * 0.10))
        median = float(np.median(score))
        effective_breadth = float(1.0 / np.sum(weights**2))
        records.append(
            {
                "trade_date": str(trade_date),
                "month": str(trade_date)[:6],
                "top1_concentration": float(weights[:top1].sum()),
                "top10_concentration": float(weights[:top10].sum()),
                "effective_breadth": effective_breadth,
                "normalized_breadth": effective_breadth / count,
                "top1_score_spread": float(np.mean(score[:top1]) - median),
                "top10_score_spread": float(np.mean(score[:top10]) - median),
            }
        )
    return pd.DataFrame.from_records(records)


def _mean_daily_rank_ic(
    frame: DataFrame,
    feature: str,
    minimum_cross_section: int,
) -> float | None:
    values: list[float] = []
    for _, daily in frame.groupby("trade_date", sort=True):
        factor = pd.to_numeric(daily[feature], errors="coerce")
        target = pd.to_numeric(daily["future_excess_ret"], errors="coerce")
        valid = np.isfinite(factor) & np.isfinite(target)
        if int(valid.sum()) < minimum_cross_section:
            continue
        correlation = factor[valid].corr(target[valid], method="spearman")
        if pd.notna(correlation):
            values.append(float(correlation))
    return None if not values else float(np.mean(values))


def _daily_bucket_returns(frame: DataFrame, feature: str, buckets: int) -> dict[int, float]:
    daily_records: list[DataFrame] = []
    for _, daily in frame.groupby("trade_date", sort=True):
        factor = pd.to_numeric(daily[feature], errors="coerce")
        target = pd.to_numeric(daily["future_excess_ret"], errors="coerce")
        valid = np.isfinite(factor) & np.isfinite(target)
        if int(valid.sum()) < buckets:
            continue
        selected = daily.loc[valid, ["trade_date"]].copy()
        percentile = factor[valid].rank(method="average", pct=True)
        selected["bucket"] = np.minimum(
            np.floor((percentile.to_numpy() - np.finfo(float).eps) * buckets), buckets - 1
        ).astype(int)
        selected["future_excess_ret"] = target[valid].to_numpy(dtype=float)
        daily_records.append(
            selected.groupby("bucket", sort=True)["future_excess_ret"].mean().reset_index()
        )
    if not daily_records:
        return {}
    aggregated = (
        pd.concat(daily_records, ignore_index=True).groupby("bucket")["future_excess_ret"].mean()
    )
    return {int(cast(int, bucket)): float(value) for bucket, value in aggregated.items()}


def _bucket_spread(values: dict[int, float], buckets: int) -> float | None:
    if 0 not in values or buckets - 1 not in values:
        return None
    return values[buckets - 1] - values[0]


def _bucket_monotonicity(values: dict[int, float]) -> float | None:
    if len(values) < 2:
        return None
    series = pd.Series(values).sort_index()
    correlation = series.index.to_series().corr(series, method="spearman")
    return None if pd.isna(correlation) else float(correlation)


def _binned_distribution(values: pd.Series, edges: np.ndarray) -> FloatArray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(numeric)
    counts, _ = np.histogram(numeric[finite_mask], bins=edges)
    combined = np.append(counts.astype(float), float((~finite_mask).sum()))
    if combined.sum() == 0:
        return cast(FloatArray, np.full(len(combined), 1.0 / len(combined)))
    return cast(FloatArray, combined / combined.sum())


def _finite_values(values: pd.Series) -> FloatArray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return cast(FloatArray, numeric[np.isfinite(numeric)])
