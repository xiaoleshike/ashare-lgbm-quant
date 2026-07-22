"""Pure ranking, stability, portfolio-proxy, and promotion-gate calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.backtest.diagnostic_metrics import (
    assign_score_layers,
    daily_layer_returns,
    daily_prediction_ic,
    summarize_ic,
)
from ashare_quant.config.settings import ChallengerEvaluationSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import RegisteredModel

type DataFrame = pd.DataFrame


def evaluate_comparison(
    comparison: DataFrame,
    champion: RegisteredModel,
    challenger: RegisteredModel,
    settings: ChallengerEvaluationSettings,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return overall and stability records for two scores on identical rows."""

    rows: list[dict[str, Any]] = []
    overall: dict[str, dict[str, Any]] = {}
    benchmark = (
        comparison.groupby("trade_date", sort=True)["benchmark_forward_ret"]
        .median()
        .rename("benchmark_forward_ret")
        .reset_index()
    )
    benchmark["regime"] = np.select(
        [
            benchmark["benchmark_forward_ret"] > settings.regime_return_threshold,
            benchmark["benchmark_forward_ret"] < -settings.regime_return_threshold,
        ],
        ["bull", "bear"],
        default="neutral",
    )
    periods: list[tuple[str, str, set[str]]] = [
        ("overall", "all", set(comparison["trade_date"].astype(str)))
    ]
    dates = sorted(comparison["trade_date"].astype(str).unique())
    periods.extend(
        ("year", year, {date for date in dates if date.startswith(year)})
        for year in sorted({date[:4] for date in dates})
    )
    periods.extend(
        ("month", month, {date for date in dates if date.startswith(month)})
        for month in sorted({date[:6] for date in dates})
    )
    periods.extend(
        (
            "regime",
            regime,
            set(benchmark.loc[benchmark["regime"] == regime, "trade_date"].astype(str)),
        )
        for regime in ("bull", "bear", "neutral")
    )
    for role, record, score_column in (
        ("champion", champion, "champion_score"),
        ("challenger", challenger, "challenger_score"),
    ):
        evaluation = comparison[
            ["trade_date", "ts_code", "future_excess_ret", "benchmark_forward_ret"]
        ].copy()
        evaluation["prediction_score"] = comparison[score_column]
        evaluation = evaluation.sort_values(
            ["trade_date", "prediction_score", "ts_code"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        evaluation["rank"] = evaluation.groupby("trade_date", sort=False).cumcount() + 1
        evaluation["cross_section_size"] = evaluation.groupby("trade_date", sort=False)[
            "ts_code"
        ].transform("size")
        daily_ic = daily_prediction_ic(evaluation, settings.minimum_cross_section)
        layers = daily_layer_returns(
            assign_score_layers(evaluation, settings.score_layers, bottom_fraction=0.20)
        )
        for period_type, period, period_dates in periods:
            if not period_dates:
                continue
            ic_subset = daily_ic.loc[daily_ic["date"].astype(str).isin(period_dates)]
            layer_subset = layers.loc[layers["trade_date"].astype(str).isin(period_dates)]
            row = _metric_row(
                record.model_id,
                role,
                period_type,
                period,
                ic_subset,
                layer_subset,
                settings.score_layers,
            )
            rows.append(row)
            if period_type == "overall":
                overall[record.model_id] = row
    return rows, overall


def build_promotion_gate(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    settings: ChallengerEvaluationSettings,
) -> dict[str, Any]:
    """Report transparent manual-review criteria without changing model status."""

    champion_ic = _finite_metric(champion, "rank_ic")
    challenger_ic = _finite_metric(challenger, "rank_ic")
    challenger_positive = _finite_metric(challenger, "positive_ic_ratio")
    champion_top10 = _finite_metric(champion, "top_10pct_mean_excess_return")
    challenger_top10 = _finite_metric(challenger, "top_10pct_mean_excess_return")
    criteria = [
        {
            "name": "minimum_labelled_days",
            "value": int(challenger["days"]),
            "threshold": settings.minimum_labelled_days,
            "passed": int(challenger["days"]) >= settings.minimum_labelled_days,
        },
        {
            "name": "minimum_rank_ic",
            "value": challenger_ic,
            "threshold": settings.minimum_rank_ic,
            "passed": challenger_ic >= settings.minimum_rank_ic,
        },
        {
            "name": "minimum_rank_ic_delta",
            "value": challenger_ic - champion_ic,
            "threshold": settings.minimum_rank_ic_delta,
            "passed": challenger_ic - champion_ic >= settings.minimum_rank_ic_delta,
        },
        {
            "name": "minimum_positive_ic_ratio",
            "value": challenger_positive,
            "threshold": settings.minimum_positive_ic_ratio,
            "passed": challenger_positive >= settings.minimum_positive_ic_ratio,
        },
        {
            "name": "minimum_top10_return_delta",
            "value": challenger_top10 - champion_top10,
            "threshold": settings.minimum_top10_return_delta,
            "passed": challenger_top10 - champion_top10 >= settings.minimum_top10_return_delta,
        },
    ]
    return {
        "policy": "manual_review_only",
        "eligible_for_manual_review": all(bool(item["passed"]) for item in criteria),
        "criteria": criteria,
        "automatic_promotion": False,
        "registry_modified": False,
    }


def _metric_row(
    model_id: str,
    role: str,
    period_type: str,
    period: str,
    daily_ic: DataFrame,
    layer_returns: DataFrame,
    score_layers: tuple[float, ...],
) -> dict[str, Any]:
    ic = summarize_ic(daily_ic)
    row: dict[str, Any] = {
        "model_id": model_id,
        "model_role": role,
        "period_type": period_type,
        "period": period,
        "days": ic["days"],
        "rank_ic": ic["mean_ic"],
        "icir": ic["icir"],
        "positive_ic_ratio": ic["positive_ic_ratio"],
    }
    for fraction in score_layers:
        name = _layer_name(fraction)
        values = layer_returns.loc[layer_returns["layer"] == name, "future_excess_ret"]
        row[f"{name}_mean_excess_return"] = None if values.empty else float(values.mean())
        row[f"{name}_positive_ratio"] = None if values.empty else float((values > 0).mean())
    return row


def _finite_metric(metrics: dict[str, Any], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise DataValidationError(f"promotion gate metric is unavailable: {name}")
    return float(value)


def _layer_name(fraction: float) -> str:
    percentage = fraction * 100
    label = str(int(percentage)) if percentage.is_integer() else f"{percentage:g}"
    return f"top_{label}pct"
