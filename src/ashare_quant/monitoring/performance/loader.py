"""Load only immutable Phase 2.7.3C observation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.performance.schemas import ObservationSources
from ashare_quant.monitoring.performance.validation import (
    validate_model_lineage,
    validate_observation_frame,
    validate_source_manifest,
)
from ashare_quant.monitoring.performance_observation.storage import (
    normalize_observation_lineage,
    read_observation_artifact,
)


def load_observation_sources(reports_root: Path, as_of: str) -> ObservationSources:
    """Load complete observation batches through as-of and validate all lineage."""

    root = reports_root / "performance_observation"
    if not root.is_dir():
        raise DataValidationError(f"performance observation root is missing: {root}")
    frames: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    model_lineage: dict[str, dict[str, Any]] = {}
    directories = sorted(path for path in root.iterdir() if path.is_dir() and path.name <= as_of)
    if not directories:
        raise DataValidationError(f"no performance observations exist through {as_of}")
    for directory in directories:
        artifact = read_observation_artifact(directory)
        if artifact is None:
            raise DataValidationError(f"incomplete performance observation artifact: {directory}")
        frame, manifest = artifact
        validate_source_manifest(manifest, directory.name)
        metrics_path = directory / "metrics.json"
        if file_sha256(metrics_path) != manifest.get("metrics_file_sha256"):
            raise DataValidationError("performance observation metrics hash mismatch")
        _merge_model_lineage(model_lineage, manifest)
        frames.append(frame)
        manifests.append(_source_identity(directory.name, manifest))
        source_hashes[directory.name] = canonical_payload_hash(
            {
                "manifest": json.loads((directory / "manifest.json").read_text(encoding="utf-8")),
                "manifest_file_sha256": file_sha256(directory / "manifest.json"),
            }
        )
    combined = normalize_observation_lineage(pd.concat(frames, ignore_index=True))
    validate_observation_frame(combined, as_of)
    validate_model_lineage(combined, model_lineage)
    ordered = combined.sort_values(
        ["model_id", "horizon", "signal_date", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    return ObservationSources(
        observations=ordered,
        source_manifests=tuple(manifests),
        source_hashes=source_hashes,
        model_lineage=model_lineage,
    )


def _merge_model_lineage(
    target: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    records = manifest.get("model_lineage")
    if not isinstance(records, list):
        raise DataValidationError("performance observation manifest lacks model_lineage")
    if not records:
        if int(manifest.get("row_count", -1)) == 0:
            return
        raise DataValidationError("non-empty performance observation lacks model_lineage")
    for raw in records:
        if not isinstance(raw, dict):
            raise DataValidationError("invalid model_lineage record")
        model_id = str(raw.get("model_id", ""))
        normalized = {
            "model_id": model_id,
            "model_role": str(raw.get("model_role", "")),
            "model_origin": str(
                raw.get("model_origin")
                or ("champion" if raw.get("model_role") == "champion" else "research_challenger")
            ),
            "feature_hash": str(raw.get("feature_hash", "")),
            "universe_hash": str(raw.get("universe_hash", "")),
            "source_models": sorted(str(value) for value in raw.get("source_models", [])),
            "fusion_method": raw.get("fusion_method"),
            "parent_model_id": str(raw.get("parent_model_id", "")),
            "training_request_id": str(raw.get("training_request_id", "")),
            "training_run_id": str(raw.get("training_run_id", "")),
            "validation_run_id": str(raw.get("validation_run_id", "")),
        }
        if not model_id or not normalized["model_role"]:
            raise DataValidationError("model_lineage record lacks identity")
        previous = target.get(model_id)
        if previous is not None and previous != normalized:
            raise DataValidationError(f"model lineage changed across observations: {model_id}")
        target[model_id] = normalized


def _source_identity(date: str, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation_as_of": date,
        "observation_hash": manifest.get("observation_hash"),
        "source_identity_hash": manifest.get("source_identity_hash"),
        "row_count": manifest.get("row_count"),
        "available_rows": manifest.get("available_rows"),
    }
