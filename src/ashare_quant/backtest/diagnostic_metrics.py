"""Post-hoc score, IC, and stability metrics for frozen predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

type DataFrame = pd.DataFrame


def assign_score_layers(
    frame: DataFrame,
    fractions: Sequence[float],
    bottom_fraction: float,
) -> DataFrame:
    """Assign overlapping daily score layers before considering label availability."""

    ranked = frame.copy()
    if "cross_section_size" not in ranked:
        ranked["cross_section_size"] = ranked.groupby("trade_date")["ts_code"].transform("size")
    records: list[DataFrame] = []
    for fraction in fractions:
        cutoff = np.ceil(ranked["cross_section_size"] * fraction)
        selected = ranked.loc[ranked["rank"] <= cutoff].copy()
        selected["layer"] = _layer_name(fraction)
        records.append(selected)
    bottom_start = np.floor(ranked["cross_section_size"] * (1.0 - bottom_fraction))
    bottom = ranked.loc[ranked["rank"] > bottom_start].copy()
    bottom["layer"] = "bottom"
    records.append(bottom)
    return pd.concat(records, ignore_index=True)


def daily_layer_returns(layered: DataFrame) -> DataFrame:
    """Calculate equal-weight forward excess return for each score cohort."""

    valid = layered.loc[layered["future_excess_ret"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=["trade_date", "layer", "future_excess_ret", "stocks", "coverage"]
        )
    available = (
        valid.groupby(["trade_date", "layer"], sort=True)
        .agg(future_excess_ret=("future_excess_ret", "mean"), stocks=("ts_code", "size"))
        .reset_index()
    )
    totals = (
        layered.groupby(["trade_date", "layer"], sort=True)["ts_code"]
        .size()
        .rename("selected_stocks")
        .reset_index()
    )
    result = available.merge(totals, on=["trade_date", "layer"], validate="one_to_one")
    result["coverage"] = result["stocks"] / result["selected_stocks"]
    return result.drop(columns="selected_stocks")


def summarize_score_layers(
    daily: DataFrame,
    *,
    horizon: int,
    annualization_days: int,
) -> list[dict[str, Any]]:
    """Summarize overlapping H-day cohorts using H non-overlapping vintage paths."""

    records: list[dict[str, Any]] = []
    for layer, group in daily.groupby("layer", sort=True):
        ordered = group.sort_values("trade_date").reset_index(drop=True)
        values = ordered["future_excess_ret"].to_numpy(dtype=float)
        vintage_metrics = [
            _path_metrics(values[offset::horizon], annualization_days / horizon)
            for offset in range(horizon)
            if len(values[offset::horizon])
        ]
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        records.append(
            {
                "layer": str(layer),
                "observations": len(values),
                "mean_forward_excess_return": float(np.mean(values)),
                "cumulative_return": _median_metric(vintage_metrics, "cumulative_return"),
                "annual_return": _median_metric(vintage_metrics, "annual_return"),
                "sharpe": (
                    None
                    if not np.isfinite(standard_deviation) or standard_deviation == 0
                    else float(
                        np.mean(values) / standard_deviation * np.sqrt(annualization_days / horizon)
                    )
                ),
                "max_drawdown": _minimum_metric(vintage_metrics, "max_drawdown"),
                "win_rate": float(np.mean(values > 0)),
                "mean_label_coverage": float(ordered["coverage"].mean()),
                "vintage_count": len(vintage_metrics),
            }
        )
    return records


def daily_prediction_ic(frame: DataFrame, minimum_cross_section: int) -> DataFrame:
    """Compute daily score/return Pearson IC and Spearman Rank IC."""

    records: list[dict[str, object]] = []
    for trade_date, daily in frame.groupby("trade_date", sort=True):
        score = pd.to_numeric(daily["prediction_score"], errors="coerce")
        target = pd.to_numeric(daily["future_excess_ret"], errors="coerce")
        valid = score.notna() & target.notna() & np.isfinite(score) & np.isfinite(target)
        count = int(valid.sum())
        if count < minimum_cross_section:
            records.append(
                {
                    "date": str(trade_date),
                    "rank_ic": math.nan,
                    "spearman_ic": math.nan,
                    "pearson_ic": math.nan,
                    "observations": count,
                }
            )
            continue
        pearson = score[valid].corr(target[valid], method="pearson")
        spearman = score[valid].corr(target[valid], method="spearman")
        records.append(
            {
                "date": str(trade_date),
                "rank_ic": spearman,
                "spearman_ic": spearman,
                "pearson_ic": pearson,
                "observations": count,
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_ic(frame: DataFrame) -> dict[str, float | int | None]:
    """Aggregate finite daily Rank IC observations."""

    values = pd.to_numeric(frame["rank_ic"], errors="coerce").dropna()
    if values.empty:
        return {
            "days": 0,
            "mean_ic": None,
            "ic_std": None,
            "icir": None,
            "positive_ic_ratio": None,
        }
    standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else math.nan
    return {
        "days": len(values),
        "mean_ic": float(values.mean()),
        "ic_std": None if not np.isfinite(standard_deviation) else standard_deviation,
        "icir": (
            None
            if not np.isfinite(standard_deviation) or standard_deviation == 0
            else float(values.mean() / standard_deviation)
        ),
        "positive_ic_ratio": float((values > 0).mean()),
    }


def monthly_stability(daily_returns: DataFrame, daily_ic: DataFrame) -> DataFrame:
    """Report monthly return and win ratio for every layer alongside Rank IC."""

    layer_returns = daily_returns.copy()
    layer_returns["month"] = layer_returns["trade_date"].astype(str).str[:6]
    returns = (
        layer_returns.groupby(["month", "layer"], sort=True)["future_excess_ret"]
        .agg(return_="mean", win_rate=lambda values: float((values > 0).mean()))
        .reset_index()
        .rename(columns={"return_": "return"})
    )
    ic = daily_ic.copy()
    ic["month"] = ic["date"].astype(str).str[:6]
    monthly_ic = ic.groupby("month", sort=True)["rank_ic"].mean().rename("ic").reset_index()
    return returns.merge(monthly_ic, on="month", how="left").sort_values(["month", "layer"])


def _layer_name(fraction: float) -> str:
    percentage = fraction * 100
    label = str(int(percentage)) if percentage.is_integer() else f"{percentage:g}"
    return f"top_{label}pct"


def _path_metrics(values: np.ndarray, periods_per_year: float) -> dict[str, float]:
    equity = np.cumprod(1.0 + values)
    cumulative = float(equity[-1] - 1.0)
    years = len(values) / periods_per_year
    annual = (
        float((equity[-1] ** (1.0 / years)) - 1.0) if years > 0 and equity[-1] > 0 else math.nan
    )
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return {
        "cumulative_return": cumulative,
        "annual_return": annual,
        "max_drawdown": float(np.min(drawdown)),
    }


def _median_metric(records: list[dict[str, float]], key: str) -> float | None:
    values = [record[key] for record in records if np.isfinite(record[key])]
    return None if not values else float(np.median(values))


def _minimum_metric(records: list[dict[str, float]], key: str) -> float | None:
    values = [record[key] for record in records if np.isfinite(record[key])]
    return None if not values else float(min(values))
