"""Immutable completion manifest for optional production closed-loop stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json


@dataclass(frozen=True, slots=True)
class ClosedLoopStage:
    """One component outcome recorded independently from hard pipeline stages."""

    name: str
    status: str
    artifact_paths: tuple[str, ...]
    artifact_hashes: dict[str, str]
    duration_seconds: float
    warnings: tuple[str, ...]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "artifact_paths": list(self.artifact_paths),
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "duration_seconds": self.duration_seconds,
            "warnings": list(self.warnings),
            "metrics": self.metrics,
        }


def artifact_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    """Hash files and directory commit manifests without scanning complete artifact trees."""

    hashes: dict[str, str] = {}
    for value in sorted(set(paths)):
        path = Path(value)
        source = path if path.is_file() else path / "manifest.json"
        if source.is_file():
            hashes[str(source)] = file_sha256(source)
    return hashes


def closed_loop_stages_from_run(
    run_manifest_path: Path,
    *,
    overrides: tuple[ClosedLoopStage, ...] = (),
) -> tuple[ClosedLoopStage, ...]:
    """Normalize all completed production stages into the closed-loop contract."""

    payload = _load_json(run_manifest_path)
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list):
        raise ValueError("production run manifest has no stage list")
    overridden = {stage.name: stage for stage in overrides}
    normalized: list[ClosedLoopStage] = []
    for raw in raw_stages:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError("production run manifest contains an invalid stage")
        name = str(raw["name"])
        if name == "publish_closed_loop_manifest":
            continue
        if name in overridden:
            normalized.append(overridden[name])
            continue
        result = raw.get("result")
        result_payload = result if isinstance(result, dict) else {}
        raw_paths = result_payload.get("artifact_paths")
        paths = tuple(str(path) for path in raw_paths) if isinstance(raw_paths, list) else ()
        raw_warnings = result_payload.get("warnings")
        warnings = (
            tuple(str(warning) for warning in raw_warnings)
            if isinstance(raw_warnings, list)
            else ()
        )
        raw_metrics = result_payload.get("metrics")
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        normalized.append(
            ClosedLoopStage(
                name=name,
                status=str(raw.get("status") or "unknown"),
                artifact_paths=paths,
                artifact_hashes=artifact_hashes(paths),
                duration_seconds=float(raw.get("elapsed_seconds") or 0.0),
                warnings=warnings,
                metrics=metrics,
            )
        )
    return tuple(normalized)


def publish_closed_loop_manifest(
    *,
    reports_root: Path,
    as_of: str,
    production_run_id: str,
    component_ids: dict[str, str | None],
    stages: tuple[ClosedLoopStage, ...],
    created_at: str,
) -> tuple[Path, str]:
    """Publish one immutable run manifest and atomically update its latest projection."""

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_name": "production_closed_loop_manifest",
        "as_of": as_of,
        "production_run_id": production_run_id,
        **component_ids,
        "stages": [stage.to_dict() for stage in stages],
        "warnings": [warning for stage in stages for warning in stage.warnings],
        "created_at": created_at,
        "manifest_written_last": True,
    }
    identity = canonical_payload_hash(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    payload["closed_loop_id"] = f"closed_loop_{identity[:24]}"
    root = reports_root / as_of / "closed_loop" / production_run_id
    path = root / "manifest.json"
    if path.exists():
        existing = _load_json(path)
        existing_identity = canonical_payload_hash(
            {
                key: value
                for key, value in existing.items()
                if key not in {"created_at", "closed_loop_id"}
            }
        )
        if existing_identity != identity:
            raise ValueError("immutable closed-loop manifest identity differs")
    else:
        atomic_write_json(path, payload)
    atomic_write_json(reports_root / as_of / "closed_loop_manifest.json", _load_json(path))
    return path, str(payload["closed_loop_id"])


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"closed-loop manifest must contain an object: {path}")
    return payload
