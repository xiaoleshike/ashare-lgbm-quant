"""Champion-reference and candidate-only shadow scoring."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import PredictionModel, score_registered_model_range
from ashare_quant.models.registry import RegisteredModel

type DataFrame = pd.DataFrame


def load_champion_reference(
    *,
    predictions_path: Path,
    ranking_path: Path,
    as_of: str,
    model_id: str,
) -> DataFrame:
    """Copy existing Champion scores and ranks without loading its model artifact."""

    predictions = pd.read_parquet(predictions_path)
    ranking = pd.read_csv(ranking_path, dtype={"ts_code": str})
    required_predictions = {"trade_date", "ts_code", "prediction_score", "model_id"}
    required_ranking = {"rank", "ts_code", "prediction_score"}
    if missing := sorted(required_predictions - set(predictions.columns)):
        raise DataValidationError(f"Champion predictions lack columns: {missing}")
    if missing := sorted(required_ranking - set(ranking.columns)):
        raise DataValidationError(f"Champion ranking lacks columns: {missing}")
    if predictions.empty or set(predictions["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("Champion predictions do not match shadow as_of")
    if set(predictions["model_id"].astype(str)) != {model_id}:
        raise DataValidationError("Champion prediction model_id mismatch")
    if (
        predictions.duplicated(["trade_date", "ts_code"]).any()
        or ranking["ts_code"].duplicated().any()
    ):
        raise DataValidationError("Champion production artifacts contain duplicate stocks")
    merged = predictions.merge(
        ranking,
        on="ts_code",
        how="inner",
        suffixes=("_prediction", "_ranking"),
        validate="one_to_one",
    )
    if len(merged) != len(predictions):
        raise DataValidationError("Champion predictions and ranking stock sets differ")
    left = pd.to_numeric(merged["prediction_score_prediction"], errors="coerce")
    right = pd.to_numeric(merged["prediction_score_ranking"], errors="coerce")
    if not np.isfinite(left).all() or not np.allclose(left, right, rtol=0.0, atol=1e-12):
        raise DataValidationError("Champion prediction scores differ from published ranking")
    result = merged.loc[:, ["trade_date", "ts_code"]].copy()
    result["prediction_score"] = left
    result["rank"] = pd.to_numeric(merged["rank"], errors="raise").astype(int)
    return result.sort_values("rank", kind="mergesort").reset_index(drop=True)


def score_challenger(
    model: RegisteredModel,
    *,
    processed_root: Path,
    as_of: str,
    model_loader: Callable[[Path], PredictionModel] | None = None,
) -> DataFrame:
    """Score one candidate through the shared production inference data path."""

    batch = score_registered_model_range(
        model,
        processed_root=processed_root,
        start_date=as_of,
        end_date=as_of,
        model_loader=model_loader,
    )
    return batch.predictions.loc[:, ["trade_date", "ts_code", "prediction_score", "rank"]].copy()


def add_score_percentile(frame: DataFrame) -> DataFrame:
    """Add deterministic within-date score percentiles without changing rank."""

    result = frame.copy()
    result["score_percentile"] = result.groupby("trade_date", sort=False)["prediction_score"].rank(
        method="average", pct=True
    )
    return result
