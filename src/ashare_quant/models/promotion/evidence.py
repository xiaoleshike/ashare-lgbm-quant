"""Collection and freezing of immutable model-promotion evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.schemas import (
    EvidenceReference,
    EvidenceSnapshot,
    EvidenceType,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256

_EVIDENCE_SPECS = {
    "challenger_evaluation": ("challenger_evaluation_manifest",),
    "executable_validation": ("executable_oos_portfolio_validation_manifest",),
    "shadow_prediction": ("shadow_prediction_bundle",),
    "performance_observation": ("performance_observation",),
    "monitoring_summary": ("production_monitor_summary",),
    "alerts": ("alert_engine",),
}


@dataclass(frozen=True, slots=True)
class PromotionEvidencePaths:
    """Explicit allowlist of the six evidence sources required by governance."""

    challenger_evaluation: Path
    executable_validation: Path
    shadow_prediction: Path
    performance_observation: Path
    monitoring_summary: Path
    alerts: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        """Return evidence paths in canonical order."""

        return tuple((name, getattr(self, name)) for name in sorted(_EVIDENCE_SPECS))


def build_evidence_snapshot(
    *,
    paths: PromotionEvidencePaths,
    reports_root: Path,
    candidate_model_id: str,
    champion_model_id: str,
    cutoff_date: str,
) -> EvidenceSnapshot:
    """Validate and hash all required evidence without executing research services."""

    _validate_date(cutoff_date, "evidence_cutoff_date")
    references = tuple(
        _reference(
            evidence_type=name,
            path=path,
            reports_root=reports_root,
            candidate_model_id=candidate_model_id,
            champion_model_id=champion_model_id,
            cutoff_date=cutoff_date,
        )
        for name, path in paths.items()
    )
    core = {
        "schema_version": 1,
        "artifact_name": "promotion_evidence_snapshot",
        "candidate_model_id": candidate_model_id,
        "evidence_cutoff_date": cutoff_date,
        "sources": [item.model_dump(mode="json") for item in references],
    }
    return EvidenceSnapshot(
        candidate_model_id=candidate_model_id,
        evidence_cutoff_date=cutoff_date,
        sources=references,
        evidence_snapshot_hash=canonical_payload_hash(core),
    )


def verify_evidence_snapshot(snapshot: EvidenceSnapshot, reports_root: Path) -> None:
    """Recheck source bytes, identities, cutoff isolation, and snapshot hash."""

    core = snapshot.model_dump(mode="json", exclude={"evidence_snapshot_hash"})
    if canonical_payload_hash(core) != snapshot.evidence_snapshot_hash:
        raise DataValidationError("promotion evidence snapshot hash is invalid")
    for source in snapshot.sources:
        path = _resolve_recorded_path(source.source_path, reports_root)
        if file_sha256(path) != source.sha256:
            raise DataValidationError(f"promotion evidence source changed: {source.source_path}")
        payload = _load_json(path)
        if _manifest_identity(payload) != source.manifest_identity:
            raise DataValidationError(f"promotion evidence identity changed: {source.source_path}")
        if source.evidence_date > snapshot.evidence_cutoff_date:
            raise DataValidationError(
                f"promotion evidence is newer than cutoff: {source.source_path}"
            )


def _reference(
    *,
    evidence_type: str,
    path: Path,
    reports_root: Path,
    candidate_model_id: str,
    champion_model_id: str,
    cutoff_date: str,
) -> EvidenceReference:
    resolved = path.resolve()
    root = reports_root.resolve()
    if not resolved.is_relative_to(root):
        raise DataValidationError(f"promotion evidence must be under reports root: {resolved}")
    payload = _load_json(resolved)
    expected_names = _EVIDENCE_SPECS[evidence_type]
    if payload.get("artifact_name") not in expected_names:
        raise DataValidationError(
            f"invalid {evidence_type} artifact identity: {payload.get('artifact_name')}"
        )
    evidence_date = _evidence_date(evidence_type, payload, resolved)
    _validate_date(evidence_date, f"{evidence_type} evidence date")
    if evidence_date > cutoff_date:
        raise DataValidationError(
            f"{evidence_type} evidence date {evidence_date} exceeds cutoff {cutoff_date}"
        )
    _validate_model_lineage(evidence_type, payload, candidate_model_id, champion_model_id)
    return EvidenceReference(
        evidence_type=cast(EvidenceType, evidence_type),
        source_path=str(resolved.relative_to(root)),
        sha256=file_sha256(resolved),
        manifest_identity=_manifest_identity(payload),
        evidence_date=evidence_date,
        cutoff_date=cutoff_date,
    )


def _validate_model_lineage(
    evidence_type: str,
    payload: dict[str, Any],
    candidate_model_id: str,
    champion_model_id: str,
) -> None:
    if evidence_type in {"challenger_evaluation", "executable_validation"}:
        if payload.get("challenger_model_id") != candidate_model_id:
            raise DataValidationError(f"{evidence_type} does not identify candidate model")
        if payload.get("champion_model_id") != champion_model_id:
            raise DataValidationError(f"{evidence_type} does not identify current champion")
    if evidence_type == "shadow_prediction":
        models = payload.get("models")
        if not isinstance(models, list) or not any(
            isinstance(item, dict)
            and item.get("model_id") == candidate_model_id
            and item.get("access_policy") == "prospective_production"
            for item in models
        ):
            raise DataValidationError(
                "shadow prediction does not contain prospective candidate evidence"
            )
    if payload.get("access_policy") == "frozen_oos_evaluation":
        raise DataValidationError(f"{evidence_type} uses forbidden frozen OOS evidence")


def _evidence_date(evidence_type: str, payload: dict[str, Any], path: Path) -> str:
    key_candidates = {
        "challenger_evaluation": ("maximum_prediction_date", "evaluation_end", "created_date"),
        "executable_validation": ("maximum_signal_date", "end_date"),
        "shadow_prediction": ("as_of",),
        "performance_observation": ("observation_as_of", "as_of"),
        "monitoring_summary": ("as_of",),
        "alerts": ("as_of",),
    }[evidence_type]
    for key in key_candidates:
        value = payload.get(key)
        if isinstance(value, str) and len(value) == 8 and value.isdigit():
            return value
    for parent in path.parents:
        if len(parent.name) == 8 and parent.name.isdigit():
            return parent.name
    nested = payload.get("input_manifests")
    if isinstance(nested, dict):
        challenger = nested.get("challenger_predictions")
        if isinstance(challenger, dict):
            value = challenger.get("maximum_prediction_date")
            if isinstance(value, str):
                return value
    raise DataValidationError(f"cannot determine {evidence_type} evidence date: {path}")


def _manifest_identity(payload: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "artifact_name",
        "run_id",
        "model_id",
        "challenger_model_id",
        "champion_model_id",
        "shadow_run_id",
        "prediction_hash",
        "observation_hash",
        "observation_as_of",
        "as_of",
    )
    return {key: payload[key] for key in keys if key in payload}


def _resolve_recorded_path(value: str, reports_root: Path) -> Path:
    path = (reports_root / value).resolve()
    if not path.is_relative_to(reports_root.resolve()):
        raise DataValidationError(f"promotion evidence path escapes reports root: {value}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"promotion evidence source does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid promotion evidence JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"promotion evidence must contain an object: {path}")
    return payload


def _validate_date(value: str, field: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise DataValidationError(f"invalid {field}: {value}") from error
