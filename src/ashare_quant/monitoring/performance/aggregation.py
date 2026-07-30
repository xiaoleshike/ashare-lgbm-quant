"""Deterministic model-performance aggregation from mature observations."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from ashare_quant.monitoring.performance.decay import safe_decay_ratio
from ashare_quant.monitoring.performance.schemas import PERFORMANCE_METRIC_COLUMNS

type DataFrame = pd.DataFrame


def aggregate_performance(
    observations: DataFrame,
    model_lineage: dict[str, dict[str, Any]],
) -> tuple[DataFrame, dict[str, Any], list[str]]:
    """Calculate daily, rolling, Top-N, bucket, and decay statistics."""

    available = observations.loc[
        observations["label_status"].astype(str).eq("available")
        & observations["future_excess_ret"].notna()
    ].copy()
    daily = _daily_metrics(available)
    metric_rows: list[dict[str, Any]] = []
    model_details: list[dict[str, Any]] = []
    warnings: list[str] = []
    if observations.empty:
        warnings.append("insufficient observations: no mature performance observations")
    group_keys = ["model_id", "model_role", "horizon"]
    for key, all_group in observations.groupby(group_keys, sort=True):
        model_id, model_role, horizon = str(key[0]), str(key[1]), int(str(key[2]))
        group = available.loc[
            available["model_id"].astype(str).eq(model_id)
            & pd.to_numeric(available["horizon"], errors="coerce").eq(horizon)
        ]
        group_daily = daily.loc[
            daily["model_id"].astype(str).eq(model_id)
            & pd.to_numeric(daily["horizon"], errors="coerce").eq(horizon)
        ].sort_values("signal_date", kind="mergesort")
        row, rolling = _aggregate_group(
            model_id=model_id,
            model_role=model_role,
            horizon=horizon,
            all_rows=all_group,
            available_rows=group,
            daily=group_daily,
        )
        metric_rows.append(row)
        deciles = _decile_returns(group)
        lineage = model_lineage[model_id]
        model_details.append(
            {
                **row,
                "rolling": rolling,
                "daily_ic": _json_records(group_daily),
                "decile_returns": deciles,
                "source_models": lineage["source_models"],
                "fusion_method": lineage["fusion_method"],
            }
        )
        for window in (20, 60, 120):
            if len(group_daily) < window:
                warnings.append(
                    f"insufficient history: model={model_id} horizon={horizon} "
                    f"window={window} sessions={len(group_daily)}"
                )
        if len(group) < 50:
            warnings.append(
                f"sparse observation: model={model_id} horizon={horizon} rows={len(group)}"
            )
    metrics = pd.DataFrame.from_records(metric_rows, columns=list(PERFORMANCE_METRIC_COLUMNS))
    metrics = metrics.sort_values(["model_id", "horizon"], kind="mergesort").reset_index(drop=True)
    details = {
        "models": sorted(model_details, key=lambda item: (item["model_id"], item["horizon"])),
        "daily_ic_rows": len(daily),
    }
    return metrics, details, sorted(set(warnings))


def _daily_metrics(frame: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["model_id", "model_role", "horizon", "signal_date"]
    for key, group in frame.groupby(keys, sort=True):
        score = pd.to_numeric(group["prediction_score"], errors="coerce")
        target = pd.to_numeric(group["future_excess_ret"], errors="coerce")
        valid = score.notna() & target.notna() & np.isfinite(score) & np.isfinite(target)
        scored = group.loc[valid].sort_values(["rank", "ts_code"], kind="mergesort")
        score = score.loc[valid]
        target = target.loc[valid]
        record: dict[str, Any] = {
            "model_id": str(key[0]),
            "model_role": str(key[1]),
            "horizon": int(str(key[2])),
            "signal_date": str(key[3]),
            "rows": len(scored),
            "pearson_ic": _correlation(score, target, "pearson"),
            "rank_ic": _correlation(score, target, "spearman"),
        }
        for top_n in (10, 20, 50):
            ranks = pd.to_numeric(scored["rank"], errors="coerce")
            selected = scored.loc[ranks.le(top_n)]
            returns = pd.to_numeric(selected["future_excess_ret"], errors="coerce")
            record[f"top{top_n}_return"] = _finite_mean(returns)
            record[f"top{top_n}_hit_rate"] = (
                float((returns > 0).mean()) if not returns.empty else None
            )
        records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "model_id",
            "model_role",
            "horizon",
            "signal_date",
            "rows",
            "pearson_ic",
            "rank_ic",
            "top10_return",
            "top10_hit_rate",
            "top20_return",
            "top20_hit_rate",
            "top50_return",
            "top50_hit_rate",
        ],
    )


def _aggregate_group(
    *,
    model_id: str,
    model_role: str,
    horizon: int,
    all_rows: DataFrame,
    available_rows: DataFrame,
    daily: DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pearson = _numeric_column(daily, "pearson_ic")
    rank_ic = _numeric_column(daily, "rank_ic")
    historical_rank = _finite_mean(rank_ic)
    historical_top10 = _finite_mean(daily.get("top10_return"))
    rolling: dict[str, Any] = {}
    row: dict[str, Any] = {
        "model_id": model_id,
        "model_role": model_role,
        "horizon": horizon,
        "feature_hash": str(all_rows["feature_hash"].iloc[0]),
        "universe_hash": str(all_rows["universe_hash"].iloc[0]),
        "observation_rows": len(all_rows),
        "available_rows": len(available_rows),
        "sessions": len(daily),
        "pearson_ic": _finite_mean(pearson),
        "rank_ic": historical_rank,
        "icir": _icir(rank_ic),
        "positive_ic_ratio": float((rank_ic > 0).mean()) if not rank_ic.empty else None,
        "top10_average_excess_ret": historical_top10,
        "top10_hit_rate": _finite_mean(daily.get("top10_hit_rate")),
        "top20_average_excess_ret": _finite_mean(daily.get("top20_return")),
        "top20_hit_rate": _finite_mean(daily.get("top20_hit_rate")),
        "top50_average_excess_ret": _finite_mean(daily.get("top50_return")),
        "top50_hit_rate": _finite_mean(daily.get("top50_hit_rate")),
        "decile_monotonicity": _decile_monotonicity(available_rows),
    }
    for window in (20, 60, 120):
        tail = daily.tail(window)
        values = _numeric_column(tail, "rank_ic")
        prefix = f"rolling_{window}"
        rolling[str(window)] = {
            "sessions": len(tail),
            "ic_mean": _finite_mean(values),
            "ic_std": _sample_std(values),
            "icir": _icir(values),
            "positive_ic_ratio": float((values > 0).mean()) if not values.empty else None,
            "top10_average_excess_ret": _finite_mean(tail.get("top10_return")),
        }
        row[f"{prefix}_ic_mean"] = rolling[str(window)]["ic_mean"]
        row[f"{prefix}_ic_std"] = rolling[str(window)]["ic_std"]
        row[f"{prefix}_icir"] = rolling[str(window)]["icir"]
        row[f"{prefix}_positive_ic_ratio"] = rolling[str(window)]["positive_ic_ratio"]
    row["alpha_decay_ratio"] = safe_decay_ratio(row["rolling_20_ic_mean"], historical_rank)
    row["top10_decay_ratio"] = safe_decay_ratio(
        rolling["20"]["top10_average_excess_ret"], historical_top10
    )
    return row, rolling


def _decile_returns(frame: DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    percentile = pd.to_numeric(frame["score_percentile"], errors="coerce")
    target = pd.to_numeric(frame["future_excess_ret"], errors="coerce")
    valid = percentile.notna() & target.notna()
    if not valid.any():
        return []
    buckets = np.ceil(percentile.loc[valid].clip(0.0, 1.0) * 10).clip(1, 10).astype(int)
    working = pd.DataFrame({"decile": buckets, "future_excess_ret": target.loc[valid]})
    return [
        {
            "decile": int(str(decile)),
            "rows": len(group),
            "average_excess_ret": float(group["future_excess_ret"].mean()),
        }
        for decile, group in working.groupby("decile", sort=True)
    ]


def _decile_monotonicity(frame: DataFrame) -> float | None:
    deciles = _decile_returns(frame)
    if len(deciles) < 2:
        return None
    x = pd.Series([item["decile"] for item in deciles], dtype=float)
    y = pd.Series([item["average_excess_ret"] for item in deciles], dtype=float)
    value = x.corr(y, method="spearman")
    return float(value) if pd.notna(value) else None


def _correlation(
    left: pd.Series,
    right: pd.Series,
    method: Literal["pearson", "spearman"],
) -> float | None:
    if len(left) < 2 or left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
        return None
    value = left.corr(right, method=method)
    return float(value) if pd.notna(value) else None


def _sample_std(values: pd.Series) -> float | None:
    return float(values.std(ddof=1)) if len(values) > 1 else None


def _icir(values: pd.Series) -> float | None:
    mean = _finite_mean(values)
    std = _sample_std(values)
    if mean is None or std is None or not math.isfinite(std) or std == 0:
        return None
    return mean / std


def _finite_mean(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if not numeric.empty else None


def _json_records(frame: DataFrame) -> list[dict[str, Any]]:
    normalized = frame.astype(object).where(frame.notna(), None)
    return cast(list[dict[str, Any]], normalized.to_dict("records"))


def _numeric_column(frame: DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()
