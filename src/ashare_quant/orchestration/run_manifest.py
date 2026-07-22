"""Atomic run tracking for future production pipeline commands."""

from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

RUN_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_RUNS_ROOT = Path("runs")

type RunStatus = Literal["running", "success", "failed"]
type StageStatus = Literal["pending", "running", "success", "failed"]


@dataclass(frozen=True, slots=True)
class ProductionRun:
    """Filesystem identity for one production run record."""

    run_id: str
    run_dir: Path

    @property
    def manifest_path(self) -> Path:
        """Return this run's JSON manifest path."""

        return self.run_dir / "manifest.json"


def create_run(
    command: str,
    *,
    config_path: str | Path | None = None,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    stages: tuple[str, ...] = (),
    upstream_manifests: dict[str, dict[str, Any]] | None = None,
    model_id: str | None = None,
    feature_hash: str | None = None,
    data_fingerprint: dict[str, Any] | None = None,
    pipeline_type: str | None = None,
    as_of: str | None = None,
    run_id: str | None = None,
) -> ProductionRun:
    """Create an atomic manifest in the running state and return its handle."""

    if not command.strip():
        raise ValueError("run command must not be empty")
    if len(set(stages)) != len(stages) or any(not stage.strip() for stage in stages):
        raise ValueError("run stages must be unique non-empty names")

    started = datetime.now(UTC)
    resolved_run_id = run_id or _generate_run_id(started)
    _validate_run_id(resolved_run_id)
    run_dir = Path(runs_root) / started.strftime("%Y%m%d") / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run = ProductionRun(run_id=resolved_run_id, run_dir=run_dir)
    git_info = current_git_info()
    manifest: dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "command": command,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "start_time": _format_time(started),
        "end_time": None,
        "elapsed_seconds": None,
        "status": "running",
        "pipeline_type": pipeline_type,
        "as_of": as_of,
        "model_id": model_id,
        "artifact_paths": [],
        "warnings": [],
        "current_stage": None,
        "error_message": None,
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "config_path": str(config_path) if config_path is not None else None,
        "config_hash": config_hash(config_path),
        "stages": [_new_stage(name) for name in stages],
        "source_provenance": {
            "upstream_manifests": upstream_manifests or {},
            "input_manifests": upstream_manifests or {},
            "resulting_manifests": {},
            "model_id": model_id,
            "feature_hash": feature_hash,
            "data_fingerprint": data_fingerprint or {},
        },
    }
    try:
        atomic_write_json(run.manifest_path, manifest)
    except Exception:
        run_dir.rmdir()
        raise
    return run


def update_run_context(
    run: ProductionRun,
    *,
    model_id: str | None = None,
    artifact_paths: tuple[str, ...] = (),
    warnings: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Atomically add production outputs and warnings to a running manifest."""

    manifest = _read_running_manifest(run)
    if model_id is not None:
        manifest["model_id"] = model_id
        provenance = manifest.get("source_provenance")
        if isinstance(provenance, dict):
            provenance["model_id"] = model_id
    current_paths = manifest.get("artifact_paths", [])
    current_warnings = manifest.get("warnings", [])
    if not isinstance(current_paths, list) or not isinstance(current_warnings, list):
        raise ValueError("run manifest has invalid output context")
    manifest["artifact_paths"] = list(dict.fromkeys([*current_paths, *artifact_paths]))
    manifest["warnings"] = list(dict.fromkeys([*current_warnings, *warnings]))
    atomic_write_json(run.manifest_path, manifest)
    return manifest


def update_source_provenance(
    run: ProductionRun,
    artifact_name: str,
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish a newly produced artifact as current run provenance."""

    if not artifact_name.strip() or not artifact_manifest:
        raise ValueError("artifact provenance requires a name and non-empty manifest")
    manifest = _read_running_manifest(run)
    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("run manifest has invalid source_provenance")
    upstream = provenance.setdefault("upstream_manifests", {})
    resulting = provenance.setdefault("resulting_manifests", {})
    if not isinstance(upstream, dict) or not isinstance(resulting, dict):
        raise ValueError("run manifest has invalid artifact provenance mappings")
    upstream[artifact_name] = artifact_manifest
    resulting[artifact_name] = artifact_manifest
    atomic_write_json(run.manifest_path, manifest)
    return manifest


def update_run_status(
    run: ProductionRun,
    status: RunStatus,
    *,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Apply a valid run status transition and atomically persist it."""

    manifest = _read_run_manifest(run)
    current_status = _run_status(manifest)
    if current_status in {"success", "failed"}:
        if current_status == status:
            return manifest
        raise ValueError(f"cannot change terminal run status {current_status} to {status}")
    if status == "running":
        return manifest
    if status == "failed":
        return _fail_manifest(run, manifest, error_message or "production run failed")

    incomplete = [
        str(stage.get("name"))
        for stage in _stage_list(manifest)
        if stage.get("status") != "success"
    ]
    if incomplete:
        raise ValueError(f"cannot mark run successful; incomplete stages={incomplete}")
    manifest["status"] = "success"
    manifest["current_stage"] = None
    manifest["end_time"] = _utc_now()
    manifest["elapsed_seconds"] = _elapsed_seconds(
        _optional_string(manifest.get("start_time")),
        _optional_string(manifest.get("end_time")),
    )
    manifest["error_message"] = None
    atomic_write_json(run.manifest_path, manifest)
    return manifest


def record_stage_start(run: ProductionRun, stage_name: str) -> dict[str, Any]:
    """Start one stage, appending it when it was not predeclared."""

    if not stage_name.strip():
        raise ValueError("stage_name must not be empty")
    manifest = _read_running_manifest(run)
    current_stage = manifest.get("current_stage")
    if current_stage not in {None, stage_name}:
        raise ValueError(f"stage {current_stage} is already running")
    stage = _find_stage(manifest, stage_name)
    if stage is None:
        stage = _new_stage(stage_name)
        _stage_list(manifest).append(stage)
    stage_status = _stage_status(stage)
    if stage_status == "running":
        return manifest
    if stage_status != "pending":
        raise ValueError(f"cannot start stage {stage_name} from status {stage_status}")
    stage["status"] = "running"
    stage["start_time"] = _utc_now()
    stage["end_time"] = None
    stage["elapsed_seconds"] = None
    stage["error_message"] = None
    stage["result"] = None
    manifest["current_stage"] = stage_name
    atomic_write_json(run.manifest_path, manifest)
    return manifest


def record_stage_end(
    run: ProductionRun,
    stage_name: str,
    *,
    status: Literal["success", "failed"] = "success",
    error_message: str | None = None,
    result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finish a running stage and fail the enclosing run when requested."""

    manifest = _read_running_manifest(run)
    stage = _find_stage(manifest, stage_name)
    if stage is None:
        raise ValueError(f"unknown run stage: {stage_name}")
    stage_status = _stage_status(stage)
    if stage_status == status:
        return manifest
    if stage_status != "running":
        raise ValueError(f"cannot end stage {stage_name} from status {stage_status}")
    if manifest.get("current_stage") != stage_name:
        raise ValueError(f"stage {stage_name} is not the current stage")
    if status == "failed":
        stage["result"] = result
        return _fail_manifest(run, manifest, error_message or f"stage {stage_name} failed")

    stage["status"] = "success"
    stage["end_time"] = _utc_now()
    stage["elapsed_seconds"] = _elapsed_seconds(
        _optional_string(stage.get("start_time")),
        _optional_string(stage.get("end_time")),
    )
    stage["error_message"] = None
    stage["result"] = result
    manifest["current_stage"] = None
    atomic_write_json(run.manifest_path, manifest)
    return manifest


def record_failure(
    run: ProductionRun,
    error: BaseException | str,
    *,
    stage_name: str | None = None,
) -> dict[str, Any]:
    """Record a terminal run failure and fail its active or named stage."""

    manifest = _read_run_manifest(run)
    if _run_status(manifest) == "failed":
        return manifest
    if _run_status(manifest) == "success":
        raise ValueError("cannot record failure after run success")
    message = _error_message(error)
    selected_stage = stage_name or _optional_string(manifest.get("current_stage"))
    if selected_stage is not None:
        stage = _find_stage(manifest, selected_stage)
        if stage is None:
            stage = _new_stage(selected_stage)
            _stage_list(manifest).append(stage)
        if stage.get("start_time") is None:
            stage["start_time"] = _utc_now()
        stage["status"] = "failed"
        stage["end_time"] = _utc_now()
        stage["elapsed_seconds"] = _elapsed_seconds(
            _optional_string(stage.get("start_time")),
            _optional_string(stage.get("end_time")),
        )
        stage["error_message"] = message
        manifest["current_stage"] = selected_stage
    return _fail_manifest(run, manifest, message)


def _fail_manifest(
    run: ProductionRun,
    manifest: dict[str, Any],
    error_message: str,
) -> dict[str, Any]:
    current_stage = _optional_string(manifest.get("current_stage"))
    if current_stage is not None:
        stage = _find_stage(manifest, current_stage)
        if stage is not None and stage.get("status") == "running":
            stage["status"] = "failed"
            stage["end_time"] = _utc_now()
            stage["elapsed_seconds"] = _elapsed_seconds(
                _optional_string(stage.get("start_time")),
                _optional_string(stage.get("end_time")),
            )
            stage["error_message"] = error_message
    manifest["status"] = "failed"
    manifest["end_time"] = _utc_now()
    manifest["elapsed_seconds"] = _elapsed_seconds(
        _optional_string(manifest.get("start_time")),
        _optional_string(manifest.get("end_time")),
    )
    manifest["error_message"] = error_message
    atomic_write_json(run.manifest_path, manifest)
    return manifest


def _read_running_manifest(run: ProductionRun) -> dict[str, Any]:
    manifest = _read_run_manifest(run)
    status = _run_status(manifest)
    if status != "running":
        raise ValueError(f"run is already terminal with status {status}")
    return manifest


def _read_run_manifest(run: ProductionRun) -> dict[str, Any]:
    try:
        loaded = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"run manifest does not exist: {run.manifest_path}") from error
    if not isinstance(loaded, dict) or loaded.get("run_id") != run.run_id:
        raise ValueError(f"invalid run manifest: {run.manifest_path}")
    return loaded


def _find_stage(manifest: dict[str, Any], stage_name: str) -> dict[str, Any] | None:
    return next(
        (stage for stage in _stage_list(manifest) if stage.get("name") == stage_name),
        None,
    )


def _stage_list(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    stages = manifest.get("stages")
    if not isinstance(stages, list) or any(not isinstance(stage, dict) for stage in stages):
        raise ValueError("run manifest has an invalid stage list")
    return stages


def _new_stage(stage_name: str) -> dict[str, Any]:
    return {
        "name": stage_name,
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "elapsed_seconds": None,
        "error_message": None,
        "result": None,
    }


def _run_status(manifest: dict[str, Any]) -> RunStatus:
    status = manifest.get("status")
    if status not in {"running", "success", "failed"}:
        raise ValueError(f"invalid run status: {status}")
    return cast(RunStatus, status)


def _stage_status(stage: dict[str, Any]) -> StageStatus:
    status = stage.get("status")
    if status not in {"pending", "running", "success", "failed"}:
        raise ValueError(f"invalid stage status: {status}")
    return cast(StageStatus, status)


def _generate_run_id(started: datetime) -> str:
    return f"{started.strftime('%Y%m%dT%H%M%S%fZ')}_{uuid4().hex[:8]}"


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty simple directory name")


def _error_message(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return error


def _utc_now() -> str:
    return _format_time(datetime.now(UTC))


def _format_time(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _elapsed_seconds(start_time: str | None, end_time: str | None) -> float | None:
    if start_time is None or end_time is None:
        return None
    elapsed = datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)
    return max(elapsed.total_seconds(), 0.0)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
