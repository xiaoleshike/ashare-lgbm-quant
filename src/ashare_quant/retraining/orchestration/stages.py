"""Read-only exact-lineage observation and promotion-evidence tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.performance.validation import validate_source_manifest
from ashare_quant.monitoring.performance_observation.storage import read_observation_artifact
from ashare_quant.retraining.orchestration.schemas import (
    EvidenceReference,
    ObservationProgress,
    ObservationStatus,
)


def track_prospective_observations(
    *,
    reports_root: Path,
    model_id: str,
    horizon: int,
    training_run_id: str,
    validation_run_id: str,
    required_sessions: int,
    accepted_shadow_run_ids: tuple[str, ...] = (),
) -> ObservationProgress:
    """Count only mature available observations with exact retrained lineage."""

    root = reports_root / "performance_observation"
    frames: list[pd.DataFrame] = []
    artifacts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    accepted = set(accepted_shadow_run_ids)
    if root.is_dir():
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            artifact = read_observation_artifact(directory)
            if artifact is None:
                raise DataValidationError(f"incomplete observation artifact: {directory}")
            frame, manifest = artifact
            validate_source_manifest(manifest, directory.name)
            _require_observation_columns(frame)
            selected = frame.loc[
                frame["model_id"].astype(str).eq(model_id)
                & frame["model_origin"].astype(str).eq("retrained_challenger")
                & pd.to_numeric(frame["horizon"], errors="coerce").eq(horizon)
                & frame["training_run_id"].astype(str).eq(training_run_id)
                & frame["validation_run_id"].astype(str).eq(validation_run_id)
            ].copy()
            if selected.empty:
                continue
            if not accepted:
                raise DataValidationError(
                    "prospective observation cannot be accepted without successful shadow lineage"
                )
            unknown = sorted(set(selected["shadow_run_id"].astype(str)) - accepted)
            if unknown:
                raise DataValidationError(f"observation contains unknown shadow_run_id: {unknown}")
            frames.append(selected)
            key = f"performance_observation:{directory.name}"
            path = directory / "manifest.json"
            artifacts[key] = str(path)
            hashes[key] = file_sha256(path)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if (
        not combined.empty
        and combined.duplicated(["model_id", "signal_date", "ts_code", "horizon"]).any()
    ):
        raise DataValidationError("prospective observations contain duplicate identities")
    available = (
        combined.loc[
            combined["label_status"].astype(str).eq("available")
            & combined["future_excess_ret"].notna()
        ].copy()
        if not combined.empty
        else combined
    )
    signal_dates = (
        tuple(sorted(available["signal_date"].astype(str).unique())) if not available.empty else ()
    )
    sessions = len(signal_dates)
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
    cutoff = max((key.rsplit(":", 1)[-1] for key in artifacts), default=None)
    aggregate_hash = canonical_payload_hash(
        {
            "model_id": model_id,
            "horizon": horizon,
            "training_run_id": training_run_id,
            "validation_run_id": validation_run_id,
            "accepted_shadow_run_ids": list(shadow_ids),
            "signal_dates": list(signal_dates),
            "source_hashes": hashes,
        }
    )
    return ObservationProgress(
        cast(ObservationStatus, status),
        sessions,
        required_sessions,
        artifacts,
        hashes,
        shadow_ids,
        signal_dates[0] if signal_dates else None,
        signal_dates[-1] if signal_dates else None,
        cutoff,
        aggregate_hash,
    )


def resolve_promotion_evidence_references(
    *,
    reports_root: Path,
    lifecycle_run_id: str,
    request_id: str,
    model_id: str,
    parent_model_id: str,
    horizon: int,
    training_run_id: str,
    validation_run_id: str,
    execution_path: Path,
    validation_path: Path,
    shadow_path: Path,
    observation: ObservationProgress,
    policy: PromotionGatePolicy,
) -> tuple[bool, dict[str, str], dict[str, str], tuple[str, ...], tuple[EvidenceReference, ...]]:
    """Resolve only exact-lineage immutable evidence; never create a request."""

    warnings: list[str] = []
    shadow = _json(shadow_path)
    _require_shadow_lineage(
        shadow,
        model_id=model_id,
        horizon=horizon,
        training_request_id=request_id,
        training_run_id=training_run_id,
        validation_run_id=validation_run_id,
        accepted_shadow_run_ids=observation.shadow_run_ids,
    )
    artifacts = {
        "retraining_execution": str(execution_path),
        "retraining_validation": str(validation_path),
        "retrained_shadow": str(shadow_path),
        **observation.source_artifacts,
    }
    hashes = {name: file_sha256(Path(path)) for name, path in artifacts.items()}
    for name, digest in observation.source_hashes.items():
        if hashes.get(name, digest) != digest:
            raise DataValidationError(f"observation source hash changed: {name}")
        hashes[name] = digest

    cutoff = observation.observation_cutoff
    performance = _matching_performance(
        reports_root,
        model_id=model_id,
        horizon=horizon,
        training_run_id=training_run_id,
        validation_run_id=validation_run_id,
        cutoff=cutoff,
    )
    if performance is None:
        warnings.append("missing exact-lineage performance monitor evidence")
    else:
        artifacts["performance_monitor"] = str(performance)
        hashes["performance_monitor"] = file_sha256(performance)
    alerts = _matching_alerts(reports_root, performance, cutoff)
    if alerts is None:
        warnings.append("missing exact-lineage alert evidence")
    else:
        artifacts["alerts"] = str(alerts)
        hashes["alerts"] = file_sha256(alerts)
    if policy.require.paper_trading:
        paper = _matching_paper(
            reports_root,
            model_id=model_id,
            horizon=horizon,
            training_run_id=training_run_id,
            cutoff=cutoff,
        )
        if paper is None:
            warnings.append("missing policy-required retrained Challenger paper-trading evidence")
        else:
            artifacts["paper_trading"] = str(paper)
            hashes["paper_trading"] = file_sha256(paper)

    references = tuple(
        EvidenceReference(
            evidence_type=name,
            lifecycle_run_id=lifecycle_run_id,
            request_id=request_id,
            model_id=model_id,
            parent_model_id=parent_model_id,
            horizon=cast(Any, horizon),
            training_request_id=request_id,
            training_run_id=training_run_id,
            validation_run_id=validation_run_id,
            shadow_run_ids=observation.shadow_run_ids,
            production_run_id=str(shadow.get("production_run_id") or "") or None,
            observation_cutoff=cutoff,
            monitoring_cutoff=cutoff if name in {"performance_monitor", "alerts"} else None,
            promotion_policy_hash=policy.policy_hash,
            artifact_path=path,
            artifact_sha256=hashes[name],
        )
        for name, path in sorted(artifacts.items())
    )
    ready = not warnings and bool(observation.source_artifacts)
    return ready, artifacts, hashes, tuple(warnings), references


def latest_successful_shadow_path(stage_results: dict[str, Any]) -> Path:
    """Select the latest successful verified enrollment/refresh, including legacy stages."""

    candidates = []
    for name in ("shadow_enrollment", "shadow_refresh", "shadow"):
        stage = stage_results.get(name)
        if stage is None or stage.status != "success":
            continue
        for raw in stage.artifact_paths:
            path = Path(raw)
            if path.is_file():
                candidates.append(path)
    if not candidates:
        raise DataValidationError("lifecycle lacks successful immutable Shadow evidence")
    validated = [(str(_json(path).get("generated_at", "")), path) for path in candidates]
    return max(validated, key=lambda item: (item[0], str(item[1])))[1]


def _matching_performance(
    reports_root: Path,
    *,
    model_id: str,
    horizon: int,
    training_run_id: str,
    validation_run_id: str,
    cutoff: str | None,
) -> Path | None:
    if cutoff is None:
        return None
    path = reports_root / "model_monitor" / cutoff / "performance" / "manifest.json"
    if not path.is_file():
        return None
    payload = _json(path)
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    matches = [
        item
        for item in models
        if isinstance(item, dict)
        and item.get("model_id") == model_id
        and item.get("model_origin") == "retrained_challenger"
        and item.get("horizon") == horizon
        and item.get("training_run_id") == training_run_id
        and item.get("validation_run_id") == validation_run_id
    ]
    return path if len(matches) == 1 and payload.get("as_of") == cutoff else None


def _matching_alerts(
    reports_root: Path, performance: Path | None, cutoff: str | None
) -> Path | None:
    if performance is None or cutoff is None:
        return None
    path = reports_root / "model_monitor" / cutoff / "alerts" / "manifest.json"
    if not path.is_file():
        return None
    payload = _json(path)
    if payload.get("as_of") != cutoff:
        return None
    monitor_path = reports_root / "model_monitor" / cutoff / "manifest.json"
    if not monitor_path.is_file():
        return None
    monitor = _json(monitor_path)
    metric_hashes = monitor.get("monitor_metric_file_hashes")
    if (
        monitor.get("artifact_name") != "production_monitor_manifest"
        or monitor.get("as_of") != cutoff
        or not isinstance(metric_hashes, dict)
        or metric_hashes.get("performance_manifest") != file_sha256(performance)
    ):
        return None
    return path


def _matching_paper(
    reports_root: Path,
    *,
    model_id: str,
    horizon: int,
    training_run_id: str,
    cutoff: str | None,
) -> Path | None:
    root = reports_root / "paper_trading_daily"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/manifest.json"), reverse=True):
        payload = _json(path)
        if (
            payload.get("model_id") == model_id
            and payload.get("model_origin") == "retrained_challenger"
            and payload.get("horizon") == horizon
            and payload.get("training_run_id") == training_run_id
            and isinstance(payload.get("as_of"), str)
            and cutoff is not None
            and str(payload["as_of"]) <= cutoff
        ):
            return path
    return None


def _require_shadow_lineage(
    payload: dict[str, Any],
    *,
    model_id: str,
    horizon: int,
    training_request_id: str,
    training_run_id: str,
    validation_run_id: str,
    accepted_shadow_run_ids: tuple[str, ...],
) -> None:
    expected = {
        "model_id": model_id,
        "model_origin": "retrained_challenger",
        "training_request_id": training_request_id,
        "training_run_id": training_run_id,
        "validation_run_id": validation_run_id,
        "access_policy": "prospective_production",
    }
    mismatches = [name for name, value in expected.items() if payload.get(name) != value]
    models = payload.get("models")
    model = (
        next(
            (
                item
                for item in models
                if isinstance(item, dict) and item.get("model_id") == model_id
            ),
            None,
        )
        if isinstance(models, list)
        else None
    )
    if model is not None and model.get("native_horizon") != horizon:
        mismatches.append("horizon")
    if payload.get("shadow_run_id") not in set(accepted_shadow_run_ids):
        mismatches.append("shadow_run_id")
    if mismatches:
        raise DataValidationError(f"retrained Shadow lineage mismatch: {sorted(set(mismatches))}")


def _require_observation_columns(frame: pd.DataFrame) -> None:
    required = {
        "model_id",
        "model_origin",
        "horizon",
        "signal_date",
        "ts_code",
        "training_run_id",
        "validation_run_id",
        "shadow_run_id",
        "label_status",
        "future_excess_ret",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"observation artifact lacks exact lineage: {missing}")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid lifecycle evidence JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"lifecycle evidence must contain an object: {path}")
    return value
