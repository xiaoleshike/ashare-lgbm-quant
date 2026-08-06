"""Immutable lineage contracts for retrained-Challenger shadow scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ashare_quant.models.registry import RegisteredModel


class RetrainedModelLineage(BaseModel):
    """Validated lineage linking training, validation, and prospective scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_origin: Literal["retrained_challenger"] = "retrained_challenger"
    model_id: str
    parent_model_id: str
    training_request_id: str
    training_run_id: str
    validation_run_id: str


@dataclass(frozen=True, slots=True)
class RetrainedShadowContext:
    """Fully validated inputs for one retrained shadow publication."""

    as_of: str
    production_run_id: str
    production_shadow_run_id: str
    current_universe_hash: str
    generated_at: str
    model: RegisteredModel
    horizon: int
    artifact_hash: str
    feature_hash: str
    training_universe_hash: str
    validation_manifest_hash: str
    lineage: RetrainedModelLineage


@dataclass(frozen=True, slots=True)
class RetrainedShadowResult:
    """Result of one immutable retrained-Challenger shadow publication."""

    model_id: str
    as_of: str
    shadow_run_id: str
    prediction_rows: int
    output_dir: Path
    idempotent: bool = False
