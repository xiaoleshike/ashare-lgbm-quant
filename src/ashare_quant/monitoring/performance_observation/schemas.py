"""Typed contracts for prospective performance observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

OBSERVATION_COLUMNS: tuple[str, ...] = (
    "observation_id",
    "signal_date",
    "observation_as_of",
    "model_id",
    "model_role",
    "model_origin",
    "horizon",
    "ts_code",
    "prediction_score",
    "rank",
    "score_percentile",
    "future_excess_ret",
    "entry_date",
    "exit_date",
    "label_status",
    "feature_hash",
    "universe_hash",
    "prediction_hash",
    "production_run_id",
    "shadow_run_id",
    "parent_model_id",
    "training_request_id",
    "training_run_id",
    "validation_run_id",
)

OBSERVATION_KEY: tuple[str, ...] = ("model_id", "signal_date", "ts_code", "horizon")
SUPPORTED_HORIZONS: tuple[int, ...] = (5, 10, 20, 60)


@dataclass(frozen=True, slots=True)
class PerformanceObservationResult:
    """One immutable incremental observation publication."""

    observation_as_of: str
    observation_rows: int
    available_rows: int
    output_dir: Path
    manifest_path: Path
    idempotent: bool = False
