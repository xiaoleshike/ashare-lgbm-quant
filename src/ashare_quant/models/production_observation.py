"""Deterministic read-only snapshots of published predictions and candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json


@dataclass(frozen=True, slots=True)
class ProductionObservationResult:
    """One published production observation snapshot."""

    prediction_date: str
    model_id: str
    candidate_count: int
    output_path: Path


class ProductionObservationRecorder:
    """Record existing model and candidate outputs without making trading decisions."""

    def __init__(self, reports_root: Path) -> None:
        self.reports_root = reports_root

    def record(self, as_of: str) -> ProductionObservationResult:
        """Write one deterministic JSON observation for an existing report date."""

        if len(as_of) != 8 or not as_of.isdigit():
            raise DataValidationError(f"as_of must be YYYYMMDD: {as_of}")
        report_dir = self.reports_root / as_of
        prediction_path = report_dir / "predictions.parquet"
        candidate_path = report_dir / "candidates.csv"
        predictions = _load_predictions(prediction_path, as_of)
        candidates = _load_candidates(candidate_path, as_of)
        model_ids = set(predictions["model_id"].astype(str)) | set(
            candidates["model_id"].astype(str)
        )
        if len(model_ids) != 1:
            raise DataValidationError(
                f"prediction and candidate model_id values do not match: {sorted(model_ids)}"
            )
        model_id = next(iter(model_ids))
        ranked = candidates.sort_values(
            ["rank", "prediction_score", "ts_code"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_name": "production_observation",
            "model_id": model_id,
            "prediction_date": as_of,
            "candidate_count": len(ranked),
            "top10_rank": _ranking_records(ranked, 10),
            "top20_rank": _ranking_records(ranked, 20),
            "top50_rank": _ranking_records(ranked, 50),
            "future_returns": {
                "status": "pending",
                "5d": None,
                "10d": None,
                "20d": None,
                "60d": None,
            },
            "source_fingerprints": {
                "predictions": _file_hash(prediction_path),
                "candidates": _file_hash(candidate_path),
            },
            "constraints": {
                "orders_generated": False,
                "trading_signal_generated": False,
                "future_returns_calculated": False,
            },
        }
        output_path = self.reports_root / "production_observation" / f"{as_of}.json"
        atomic_write_json(output_path, payload)
        return ProductionObservationResult(
            prediction_date=as_of,
            model_id=model_id,
            candidate_count=len(ranked),
            output_path=output_path,
        )


def _load_predictions(path: Path, as_of: str) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"predictions do not exist: {path}")
    frame = pd.read_parquet(path)
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"predictions are missing columns: {missing}")
    if frame.empty or not frame["trade_date"].astype(str).eq(as_of).all():
        raise DataValidationError(f"predictions do not exclusively represent {as_of}")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("predictions contain duplicate stock rows")
    scores = pd.to_numeric(frame["prediction_score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise DataValidationError("predictions contain non-finite scores")
    return frame


def _load_candidates(path: Path, as_of: str) -> pd.DataFrame:
    if not path.is_file():
        raise DataValidationError(f"candidates do not exist: {path}")
    frame = pd.read_csv(path, dtype={"trade_date": str, "ts_code": str, "model_id": str})
    required = {"rank", "ts_code", "prediction_score", "trade_date", "model_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"candidates are missing columns: {missing}")
    if not frame.empty and not frame["trade_date"].astype(str).eq(as_of).all():
        raise DataValidationError(f"candidates contain dates other than {as_of}")
    if frame["ts_code"].duplicated().any():
        raise DataValidationError("candidates contain duplicate stocks")
    frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["prediction_score"] = pd.to_numeric(frame["prediction_score"], errors="coerce")
    if frame[["rank", "prediction_score"]].isna().any().any():
        raise DataValidationError("candidates contain invalid rank or prediction_score")
    return frame


def _ranking_records(frame: pd.DataFrame, size: int) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(cast(Any, row.rank)),
            "ts_code": str(row.ts_code),
            "prediction_score": float(cast(Any, row.prediction_score)),
        }
        for row in frame.head(size).itertuples(index=False)
    ]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
