"""Typed contracts for prospective shadow predictions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ashare_quant.models.registry import RegisteredModel

type DataFrame = pd.DataFrame
type ModelRole = Literal[
    "champion",
    "challenger_h5",
    "challenger_h10",
    "challenger_h20",
    "challenger_h60",
    "multi_horizon_ensemble",
]

MODEL_ROLES: frozenset[str] = frozenset(
    {
        "champion",
        "challenger_h5",
        "challenger_h10",
        "challenger_h20",
        "challenger_h60",
        "multi_horizon_ensemble",
    }
)

MODEL_ORIGINS: frozenset[str] = frozenset(
    {"champion", "research_challenger", "retrained_challenger"}
)


@dataclass(frozen=True, slots=True)
class ReadinessResult:
    """Structured pre-scoring readiness outcome."""

    ready: bool
    hard_failures: tuple[str, ...]
    checks: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShadowContext:
    """Validated immutable inputs for one shadow run."""

    as_of: str
    production_run_id: str
    champion_model_id: str
    champion_feature_hash: str
    champion_prediction_hash: str
    champion_prediction_file_hash: str
    feature_hash: str
    universe_hash: str
    generated_at: str
    champion_predictions: DataFrame
    challenger_models: dict[int, RegisteredModel]
    challenger_manifest_hashes: dict[int, str]
    readiness: ReadinessResult


@dataclass(frozen=True, slots=True)
class ShadowPredictionResult:
    """Published immutable shadow bundle."""

    as_of: str
    production_run_id: str
    shadow_run_id: str
    prediction_rows: int
    model_count: int
    output_dir: Path
    idempotent: bool = False
