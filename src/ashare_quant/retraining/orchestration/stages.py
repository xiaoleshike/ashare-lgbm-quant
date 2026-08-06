"""Read-only observation and evidence tracking for lifecycle orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.monitoring.performance.validation import validate_source_manifest
from ashare_quant.monitoring.performance_observation.storage import read_observation_artifact
from ashare_quant.retraining.orchestration.schemas import ObservationProgress, ObservationStatus


def track_prospective_observations(
    *,
    reports_root: Path,
    model_id: str,
    horizon: int,
    training_run_id: str,
    validation_run_id: str,
    required_sessions: int,
) -> ObservationProgress:
    """Count only mature available retrained observations from Phase 2.7.3C."""

    root = reports_root / "performance_observation"
    frames: list[pd.DataFrame] = []
    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    if root.is_dir():
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            artifact = read_observation_artifact(directory)
            if artifact is None:
                raise DataValidationError(f"incomplete observation artifact: {directory}")
            frame, manifest = artifact
            validate_source_manifest(manifest, directory.name)
            selected = frame.loc[
                frame["model_id"].astype(str).eq(model_id)
                & frame["model_origin"].astype(str).eq("retrained_challenger")
                & pd.to_numeric(frame["horizon"], errors="coerce").eq(horizon)
                & frame["training_run_id"].astype(str).eq(training_run_id)
                & frame["validation_run_id"].astype(str).eq(validation_run_id)
            ].copy()
            if selected.empty:
                continue
            frames.append(selected)
            key = f"performance_observation:{directory.name}"
            path = directory / "manifest.json"
            artifacts[key] = str(path)
            hashes[key] = file_sha256(path)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    available = (
        combined.loc[
            combined["label_status"].astype(str).eq("available")
            & combined["future_excess_ret"].notna()
        ]
        if not combined.empty
        else combined
    )
    sessions = int(available["signal_date"].astype(str).nunique()) if not available.empty else 0
    status = (
        "OBSERVATION_PENDING"
        if sessions == 0
        else "OBSERVATION_SUFFICIENT"
        if sessions >= required_sessions
        else "OBSERVATION_ACCUMULATING"
    )
    shadow_ids = (
        tuple(sorted(available["shadow_run_id"].astype(str).unique()))
        if not available.empty
        else ()
    )
    return ObservationProgress(
        cast(ObservationStatus, status),
        sessions,
        required_sessions,
        artifacts,
        hashes,
        shadow_ids,
    )


def resolve_promotion_evidence_references(
    *,
    reports_root: Path,
    model_id: str,
    execution_path: Path,
    validation_path: Path,
    shadow_path: Path,
    observation: ObservationProgress,
    policy: PromotionGatePolicy,
) -> tuple[bool, dict[str, str], dict[str, str], tuple[str, ...]]:
    """Check whether immutable evidence preparation inputs exist; never create a request."""

    artifacts = {
        "retraining_execution": str(execution_path),
        "retraining_validation": str(validation_path),
        "retrained_shadow": str(shadow_path),
        **observation.source_artifacts,
    }
    hashes = {
        name: file_sha256(Path(path)) for name, path in artifacts.items()
    } | observation.source_hashes
    warnings: list[str] = []
    performance = _latest_matching(
        reports_root / "model_monitor", "performance/manifest.json", model_id
    )
    alerts = _latest_matching(reports_root / "model_monitor", "alerts/manifest.json", model_id)
    for name, path in (("performance_monitor", performance), ("alerts", alerts)):
        if path is None:
            warnings.append(f"missing {name} evidence")
        else:
            artifacts[name] = str(path)
            hashes[name] = file_sha256(path)
    if policy.require.paper_trading:
        paper = _latest_any(reports_root / "paper_trading_daily", "manifest.json")
        if paper is None:
            warnings.append("missing policy-required paper_trading evidence")
        else:
            artifacts["paper_trading"] = str(paper)
            hashes["paper_trading"] = file_sha256(paper)
    ready = not warnings and bool(observation.source_artifacts)
    return ready, artifacts, hashes, tuple(warnings)


def _latest_matching(root: Path, suffix: str, model_id: str) -> Path | None:
    matches: list[Path] = []
    if not root.is_dir():
        return None
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        candidate = directory / suffix
        if not candidate.is_file():
            continue
        payload = _json(candidate)
        models = payload.get("models")
        if isinstance(models, list) and any(
            isinstance(item, dict) and item.get("model_id") == model_id for item in models
        ):
            matches.append(candidate)
    return matches[-1] if matches else None


def _latest_any(root: Path, filename: str) -> Path | None:
    if not root.is_dir():
        return None
    matches = sorted(path / filename for path in root.iterdir() if (path / filename).is_file())
    return matches[-1] if matches else None


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid lifecycle evidence JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"lifecycle evidence must contain an object: {path}")
    return value
