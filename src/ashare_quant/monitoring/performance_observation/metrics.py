"""Cross-sectional prospective model-performance metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

type DataFrame = pd.DataFrame


def build_performance_metrics(observations: DataFrame) -> dict[str, Any]:
    """Calculate daily, aggregate, Top-N, monotonicity, and rolling IC metrics."""

    available = observations.loc[
        observations["label_status"].astype(str).eq("available")
        & observations["future_excess_ret"].notna()
    ].copy()
    if available.empty:
        return {
            "available_rows": 0,
            "daily": [],
            "models": [],
            "rolling_windows": [20, 60, 120],
        }
    daily = _daily_metrics(available)
    models = _aggregate_metrics(available, daily)
    return {
        "available_rows": len(available),
        "daily": daily.replace({np.nan: None}).to_dict("records"),
        "models": models,
        "rolling_windows": [20, 60, 120],
    }


def _daily_metrics(frame: DataFrame) -> DataFrame:
    records: list[dict[str, object]] = []
    group_columns = ["model_id", "model_role", "horizon", "signal_date"]
    for key, group in frame.groupby(group_columns, sort=True):
        model_id = str(key[0])
        model_role = str(key[1])
        horizon = int(str(key[2]))
        signal_date = str(key[3])
        score = pd.to_numeric(group["prediction_score"], errors="coerce")
        target = pd.to_numeric(group["future_excess_ret"], errors="coerce")
        valid = score.notna() & target.notna() & np.isfinite(score) & np.isfinite(target)
        scored = group.loc[valid].copy()
        score = score.loc[valid]
        target = target.loc[valid]
        pearson = score.corr(target, method="pearson") if len(scored) >= 2 else np.nan
        rank_ic = score.corr(target, method="spearman") if len(scored) >= 2 else np.nan
        record: dict[str, object] = {
            "model_id": model_id,
            "model_role": model_role,
            "horizon": horizon,
            "signal_date": signal_date,
            "observations": len(scored),
            "pearson_ic": pearson,
            "rank_ic": rank_ic,
            "hit_rate": float((target > 0).mean()) if len(target) else np.nan,
            "decile_monotonicity": _decile_monotonicity(scored),
        }
        for top_n in (10, 20, 50):
            selected = scored.sort_values(["rank", "ts_code"], kind="mergesort").head(top_n)
            record[f"top{top_n}_excess_return"] = (
                float(selected["future_excess_ret"].mean()) if not selected.empty else np.nan
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _aggregate_metrics(frame: DataFrame, daily: DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    keys = ["model_id", "model_role", "horizon"]
    for key, group in daily.groupby(keys, sort=True):
        model_id = str(key[0])
        model_role = str(key[1])
        horizon = int(str(key[2]))
        rank_ic = pd.to_numeric(group["rank_ic"], errors="coerce").dropna()
        pearson = pd.to_numeric(group["pearson_ic"], errors="coerce").dropna()
        rank_mean = float(rank_ic.mean()) if not rank_ic.empty else None
        rank_std = float(rank_ic.std(ddof=1)) if len(rank_ic) > 1 else None
        record: dict[str, Any] = {
            "model_id": model_id,
            "model_role": model_role,
            "horizon": horizon,
            "rows": int(
                len(frame.loc[frame["model_id"].eq(model_id) & frame["horizon"].eq(horizon)])
            ),
            "sessions": len(group),
            "pearson_ic": float(pearson.mean()) if not pearson.empty else None,
            "rank_ic": rank_mean,
            "icir": _safe_ratio(rank_mean, rank_std),
            "positive_ic_ratio": (float((rank_ic > 0).mean()) if not rank_ic.empty else None),
            "hit_rate": _finite_mean(group["hit_rate"]),
            "decile_monotonicity": _finite_mean(group["decile_monotonicity"]),
            "top10_excess_return": _finite_mean(group["top10_excess_return"]),
            "top20_excess_return": _finite_mean(group["top20_excess_return"]),
            "top50_excess_return": _finite_mean(group["top50_excess_return"]),
            "rolling": {},
        }
        ordered = group.sort_values("signal_date", kind="mergesort")
        for window in (20, 60, 120):
            tail = ordered.tail(window)
            values = pd.to_numeric(tail["rank_ic"], errors="coerce").dropna()
            mean = float(values.mean()) if not values.empty else None
            std = float(values.std(ddof=1)) if len(values) > 1 else None
            record["rolling"][str(window)] = {
                "sessions": len(tail),
                "rank_ic": mean,
                "icir": _safe_ratio(mean, std),
                "positive_ic_ratio": (float((values > 0).mean()) if not values.empty else None),
                "top10_excess_return": _finite_mean(tail["top10_excess_return"]),
                "top20_excess_return": _finite_mean(tail["top20_excess_return"]),
                "top50_excess_return": _finite_mean(tail["top50_excess_return"]),
            }
        records.append(record)
    return records


def _decile_monotonicity(frame: DataFrame) -> float | None:
    if len(frame) < 2:
        return None
    percentile = pd.to_numeric(frame["score_percentile"], errors="coerce")
    target = pd.to_numeric(frame["future_excess_ret"], errors="coerce")
    valid = percentile.notna() & target.notna()
    if int(valid.sum()) < 2:
        return None
    decile = np.ceil(percentile.loc[valid].clip(0.0, 1.0) * 10).clip(1, 10)
    means = target.loc[valid].groupby(decile).mean()
    if len(means) < 2:
        return None
    buckets = pd.Series(means.index, dtype=float)
    return float(buckets.corr(means.reset_index(drop=True), method="spearman"))


def _finite_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if not numeric.empty else None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or not math.isfinite(denominator):
        return None
    if denominator == 0:
        return 0.0
    return numerator / denominator
