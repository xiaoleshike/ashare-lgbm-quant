"""Authoritative model/evaluation boundary resolution for evidence-grade backtests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ashare_quant.data.exceptions import DataValidationError

ArtifactRole = Literal[
    "live_inference",
    "challenger",
    "research_candidate",
    "qualification_only",
    "legacy_unknown",
]


@dataclass(frozen=True, slots=True)
class ModelEvaluationBoundary:
    """Frozen model provenance and the strictest known selection boundary."""

    model_id: str
    artifact_type: str
    artifact_role: ArtifactRole
    training_start: str
    training_end: str
    validation_start: str | None
    validation_end: str | None
    selection_end: str
    final_test_start: str | None
    manifest_hash: str
    boundary_source: str
    evidence_eligible: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_model_evaluation_boundary(model_dir: Path) -> ModelEvaluationBoundary:
    """Resolve boundaries only from the immutable model manifest, never its path name."""

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DataValidationError(
            f"BACKTEST_MODEL_PROVENANCE_REQUIRED: model manifest is missing: {manifest_path}"
        )
    try:
        manifest_bytes = manifest_path.read_bytes()
        payload = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(
            f"BACKTEST_MODEL_PROVENANCE_REQUIRED: invalid model manifest: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise DataValidationError("BACKTEST_MODEL_PROVENANCE_REQUIRED: manifest is not an object")
    artifact_type = str(payload.get("artifact_name", ""))
    role = _artifact_role(artifact_type, payload)
    training_start = _date(payload, "train_start", "training_start", nested="train_dates")
    training_end = _date(payload, "train_end", "training_end", nested="train_dates", end=True)
    validation_start = _optional_date(
        payload, "validation_start", nested="validation_dates", end=False
    )
    validation_end = _optional_date(payload, "validation_end", nested="validation_dates", end=True)
    selection_end = max(training_end, validation_end or training_end)
    final_test_start = _optional_date(payload, "test_start", nested=None, end=False)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    model_id = str(
        payload.get("model_id") or payload.get("experiment_id") or f"manifest:{manifest_hash[:16]}"
    )
    if role == "legacy_unknown":
        raise DataValidationError(
            "BACKTEST_MODEL_PROVENANCE_REQUIRED: unsupported or unidentified model artifact "
            f"type={artifact_type or '<missing>'}"
        )
    return ModelEvaluationBoundary(
        model_id=model_id,
        artifact_type=artifact_type,
        artifact_role=role,
        training_start=training_start,
        training_end=training_end,
        validation_start=validation_start,
        validation_end=validation_end,
        selection_end=selection_end,
        final_test_start=final_test_start,
        manifest_hash=manifest_hash,
        boundary_source="model_manifest.json",
        evidence_eligible=role != "qualification_only",
    )


def require_oos_evaluation(
    boundary: ModelEvaluationBoundary,
    *,
    model_dir: Path,
    evaluation_start: str,
    evaluation_end: str,
) -> None:
    """Reject unproven or overlapping historical performance evidence."""

    _validate_date(evaluation_start, "evaluation_start")
    _validate_date(evaluation_end, "evaluation_end")
    if evaluation_start > evaluation_end:
        raise DataValidationError("BACKTEST_IN_SAMPLE_OVERLAP: evaluation dates are reversed")
    if not boundary.evidence_eligible:
        raise DataValidationError(
            "BACKTEST_MODEL_PROVENANCE_REQUIRED: qualification-only artifact cannot produce "
            f"general performance evidence: model_id={boundary.model_id}"
        )
    if evaluation_start <= boundary.selection_end:
        overlap_end = min(evaluation_end, boundary.selection_end)
        raise DataValidationError(
            "BACKTEST_IN_SAMPLE_OVERLAP: "
            f"model_id={boundary.model_id} model_artifact={model_dir} "
            f"training_end={boundary.training_end} selection_end={boundary.selection_end} "
            f"requested_start={evaluation_start} requested_end={evaluation_end} "
            f"overlap_start={evaluation_start} overlap_end={overlap_end}"
        )


def _artifact_role(artifact_type: str, payload: dict[str, Any]) -> ArtifactRole:
    if payload.get("qualification_only") is True:
        return "qualification_only"
    if artifact_type == "production_lightgbm_ranker":
        return "live_inference"
    if artifact_type in {"lightgbm_ranker_challenger", "governed_retraining_challenger"}:
        return "challenger"
    if artifact_type == "lightgbm_ranker_baseline":
        return "research_candidate"
    return "legacy_unknown"


def _date(
    payload: dict[str, Any],
    primary: str,
    secondary: str,
    *,
    nested: str,
    end: bool = False,
) -> str:
    value = payload.get(primary, payload.get(secondary))
    if value is None and isinstance(payload.get(nested), dict):
        value = payload[nested].get("end" if end else "start")
    if not isinstance(value, str):
        raise DataValidationError(
            f"BACKTEST_MODEL_PROVENANCE_REQUIRED: manifest lacks {primary}/{secondary}"
        )
    _validate_date(value, primary)
    return value


def _optional_date(
    payload: dict[str, Any], primary: str, *, nested: str | None, end: bool
) -> str | None:
    value = payload.get(primary)
    if value is None and nested is not None and isinstance(payload.get(nested), dict):
        value = payload[nested].get("end" if end else "start")
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataValidationError(
            f"BACKTEST_MODEL_PROVENANCE_REQUIRED: invalid manifest boundary {primary}"
        )
    _validate_date(value, primary)
    return value


def _validate_date(value: str, label: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"BACKTEST_MODEL_PROVENANCE_REQUIRED: {label} must be YYYYMMDD")
