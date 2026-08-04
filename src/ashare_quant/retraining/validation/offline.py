"""Selection-fold offline validation for retrained Challengers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.backtest.diagnostic_metrics import (
    assign_score_layers,
    daily_layer_returns,
    daily_prediction_ic,
    summarize_ic,
)
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger_evaluation import _load_mature_labels
from ashare_quant.models.inference import (
    PredictionModel,
    score_registered_model_range,
)
from ashare_quant.retraining.validation.schemas import (
    CandidateValidationContext,
    OfflineValidationEvidence,
    OfflineValidationRun,
)

type DataFrame = pd.DataFrame


class RetrainingOfflineValidator:
    """Score only the frozen selection-fold evaluation range and summarize labels post hoc."""

    def __init__(
        self,
        *,
        processed_root: Path,
        settings: AppSettings,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.processed_root = processed_root
        self.settings = settings
        self.model_loader = model_loader

    def evaluate(self, context: CandidateValidationContext) -> OfflineValidationRun:
        batch = score_registered_model_range(
            context.model,
            processed_root=self.processed_root,
            start_date=context.evaluation_start,
            end_date=context.evaluation_end,
            allowed_ranges=((context.evaluation_start, context.evaluation_end),),
            model_loader=self.model_loader,
        )
        predictions = batch.predictions
        dates = tuple(sorted(predictions["trade_date"].astype(str).unique()))
        labels = _load_mature_labels(
            self.processed_root,
            context.artifact.horizon,
            dates=dates,
            maximum_mature_date=context.maximum_mature_evaluation_date,
        )
        evaluation = predictions.merge(
            labels,
            on=["trade_date", "ts_code"],
            how="left",
            validate="one_to_one",
        )
        if not evaluation["future_excess_ret"].notna().any():
            raise DataValidationError("VALIDATION_FAILED: offline validation has no mature labels")
        evaluation["cross_section_size"] = evaluation.groupby("trade_date", sort=False)[
            "ts_code"
        ].transform("size")
        settings = self.settings.models.challenger_evaluation
        daily_ic = daily_prediction_ic(evaluation, settings.minimum_cross_section)
        layers = daily_layer_returns(
            assign_score_layers(evaluation, settings.score_layers, bottom_fraction=0.20)
        )
        periods = _periods(evaluation, settings.regime_return_threshold)
        rows = tuple(
            _metrics_for_period(
                context.model.model_id,
                period_type,
                period,
                dates_in_period,
                daily_ic,
                layers,
                settings.score_layers,
            )
            for period_type, period, dates_in_period in periods
            if dates_in_period
        )
        overall = next(row for row in rows if row["period_type"] == "overall")
        evidence = OfflineValidationEvidence(
            model_id=context.model.model_id,
            horizon=context.artifact.horizon,
            evaluation_start=context.evaluation_start,
            evaluation_end=context.evaluation_end,
            prediction_rows=len(predictions),
            labelled_rows=int(evaluation["future_excess_ret"].notna().sum()),
            evaluation_sessions=len(dates),
            overall_metrics=overall,
            stability_metrics=rows,
        )
        return OfflineValidationRun(evidence, predictions)


def _periods(evaluation: DataFrame, regime_threshold: float) -> list[tuple[str, str, set[str]]]:
    dates = sorted(evaluation["trade_date"].astype(str).unique())
    periods: list[tuple[str, str, set[str]]] = [("overall", "all", set(dates))]
    periods.extend(
        ("year", year, {date for date in dates if date.startswith(year)})
        for year in sorted({date[:4] for date in dates})
    )
    benchmark = (
        evaluation.groupby("trade_date", sort=True)["benchmark_forward_ret"].median().reset_index()
    )
    benchmark["regime"] = np.select(
        [
            benchmark["benchmark_forward_ret"] > regime_threshold,
            benchmark["benchmark_forward_ret"] < -regime_threshold,
        ],
        ["bull", "bear"],
        default="neutral",
    )
    periods.extend(
        (
            "regime",
            regime,
            set(benchmark.loc[benchmark["regime"] == regime, "trade_date"].astype(str)),
        )
        for regime in ("bull", "bear", "neutral")
    )
    return periods


def _metrics_for_period(
    model_id: str,
    period_type: str,
    period: str,
    dates: set[str],
    daily_ic: DataFrame,
    layers: DataFrame,
    fractions: tuple[float, ...],
) -> dict[str, Any]:
    ic = summarize_ic(daily_ic.loc[daily_ic["date"].astype(str).isin(dates)])
    layer_subset = layers.loc[layers["trade_date"].astype(str).isin(dates)]
    row: dict[str, Any] = {
        "model_id": model_id,
        "model_role": "challenger",
        "period_type": period_type,
        "period": period,
        "days": ic["days"],
        "rank_ic": ic["mean_ic"],
        "ic_std": ic["ic_std"],
        "icir": ic["icir"],
        "positive_ic_ratio": ic["positive_ic_ratio"],
    }
    for fraction in fractions:
        percentage = fraction * 100
        label = str(int(percentage)) if percentage.is_integer() else f"{percentage:g}"
        layer = f"top_{label}pct"
        values = layer_subset.loc[layer_subset["layer"] == layer, "future_excess_ret"]
        row[f"{layer}_mean_excess_return"] = None if values.empty else float(values.mean())
        row[f"{layer}_positive_ratio"] = None if values.empty else float((values > 0).mean())
    return row
