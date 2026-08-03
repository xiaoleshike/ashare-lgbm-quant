"""Validation for immutable monitoring evidence consumed by retraining."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.retraining.schemas import EvidenceReference, RetrainingEvidence


def validate_as_of(value: str) -> None:
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as error:
        raise DataValidationError(f"retraining as_of must use YYYYMMDD: {value}") from error
    if parsed.strftime("%Y%m%d") != value:
        raise DataValidationError(f"retraining as_of must use YYYYMMDD: {value}")


def validate_monitoring_bundle(
    *,
    reports_root: Path,
    as_of: str,
    monitor_manifest: dict[str, Any],
    health: dict[str, Any],
    performance_manifest: dict[str, Any],
    performance_metrics: pd.DataFrame,
    alerts_manifest: dict[str, Any],
    alerts: dict[str, Any],
    paths: dict[str, Path],
) -> RetrainingEvidence:
    """Validate identities and hashes without reading labels, features, or model artifacts."""

    _identity(monitor_manifest, "production_monitor_manifest", as_of, "monitor manifest")
    expected = monitor_manifest.get("monitor_metric_file_hashes")
    if not isinstance(expected, dict):
        raise DataValidationError("monitor manifest lacks metric file hashes")
    for name in ("health", "performance_metrics", "performance_manifest"):
        if file_sha256(paths[name]) != expected.get(name):
            raise DataValidationError(f"monitor metric source hash mismatch: {name}")
    _identity(performance_manifest, "performance_monitor", as_of, "performance manifest")
    if performance_manifest.get("status") != "success":
        raise DataValidationError("performance monitor is not successful")
    if file_sha256(paths["performance_metrics"]) != performance_manifest.get("metrics_file_sha256"):
        raise DataValidationError("performance metrics hash mismatch")
    _validate_performance_rows(performance_metrics, performance_manifest)
    _validate_observation_lineage(reports_root, as_of, performance_manifest)

    _identity(alerts_manifest, "alert_engine", as_of, "alerts manifest")
    if alerts_manifest.get("status") != "success":
        raise DataValidationError("alert artifact is not successful")
    if file_sha256(paths["alerts"]) != alerts_manifest.get("alerts_file_sha256"):
        raise DataValidationError("alerts JSON hash mismatch")
    if alerts.get("artifact_name") != "monitoring_alerts" or str(alerts.get("as_of")) != as_of:
        raise DataValidationError("invalid alerts JSON identity")
    if not isinstance(alerts.get("alerts"), list):
        raise DataValidationError("alerts JSON lacks alert rows")
    if str(health.get("as_of")) != as_of or not isinstance(health.get("model_id"), str):
        raise DataValidationError("health metrics identity differs from monitor date")
    _validate_drift_reference(reports_root, health)

    root = reports_root.resolve()
    return RetrainingEvidence(
        monitor_snapshot=_reference(paths["monitor_manifest"], root, monitor_manifest, as_of),
        performance_observation=_reference(
            paths["performance_manifest"], root, performance_manifest, as_of
        ),
        alerts=_reference(paths["alerts_manifest"], root, alerts_manifest, as_of),
    )


def validate_recorded_evidence(reports_root: Path, evidence: RetrainingEvidence) -> None:
    """Recheck request-bound evidence bytes and manifest identities."""

    root = reports_root.resolve()
    for name, reference in (
        ("monitor_snapshot", evidence.monitor_snapshot),
        ("performance_observation", evidence.performance_observation),
        ("alerts", evidence.alerts),
    ):
        path = (root / reference.path).resolve()
        if not path.is_relative_to(root):
            raise DataValidationError(f"retraining evidence path escapes reports root: {name}")
        if file_sha256(path) != reference.sha256:
            raise DataValidationError(f"retraining evidence hash mismatch: {name}")
        payload = _load_json(path, name)
        if payload.get("artifact_name") != reference.artifact_name:
            raise DataValidationError(f"retraining evidence identity changed: {name}")
        if str(payload.get("as_of") or payload.get("observation_as_of")) != reference.as_of:
            raise DataValidationError(f"retraining evidence date changed: {name}")


def evidence_hash(evidence: RetrainingEvidence) -> str:
    return canonical_payload_hash(evidence.model_dump(mode="json"))


def _validate_performance_rows(frame: pd.DataFrame, manifest: dict[str, Any]) -> None:
    required = {
        "model_id",
        "model_role",
        "horizon",
        "sessions",
        "alpha_decay_ratio",
        "rolling_20_ic_mean",
        "rolling_60_ic_mean",
        "rolling_120_ic_mean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"performance metrics lack retraining columns: {missing}")
    if frame.duplicated(["model_id", "horizon"]).any():
        raise DataValidationError("performance metrics duplicate model/horizon identity")
    expected_rows = manifest.get("row_counts")
    if not isinstance(expected_rows, dict) or int(
        expected_rows.get("model_horizon_metrics", -1)
    ) != len(frame):
        raise DataValidationError("performance metrics row count differs from manifest")
    models = manifest.get("models")
    if not isinstance(models, list):
        raise DataValidationError("performance manifest lacks model lineage")
    declared: set[tuple[str, int]] = set()
    for item in models:
        if not isinstance(item, dict):
            continue
        horizon = item.get("horizon")
        if isinstance(horizon, int) and not isinstance(horizon, bool):
            declared.add((str(item.get("model_id")), horizon))
    observed = {
        (str(row.model_id), int(cast(Any, row.horizon)))
        for row in frame.loc[:, ["model_id", "horizon"]].itertuples(index=False)
    }
    if declared != observed:
        raise DataValidationError("performance metric identities differ from manifest")


def _validate_observation_lineage(
    reports_root: Path,
    as_of: str,
    manifest: dict[str, Any],
) -> None:
    hashes = manifest.get("source_observation_hashes")
    if not isinstance(hashes, dict):
        raise DataValidationError("performance manifest lacks observation source hashes")
    for date, expected in sorted(hashes.items()):
        if not isinstance(date, str) or date > as_of:
            raise DataValidationError("performance manifest contains future observation source")
        path = reports_root / "performance_observation" / date / "manifest.json"
        payload = _load_json(path, f"performance observation {date}")
        if payload.get("artifact_name") != "performance_observation":
            raise DataValidationError("invalid performance observation identity")
        if payload.get("access_policy") != "prospective_production":
            raise DataValidationError("performance observation is not prospective production")
        contracts = payload.get("contracts")
        if (
            not isinstance(contracts, dict)
            or contracts.get("labels_used_only_after_maturity") is not True
        ):
            raise DataValidationError("performance observation lacks maturity contract")
        actual = canonical_payload_hash(
            {
                "manifest": payload,
                "manifest_file_sha256": file_sha256(path),
            }
        )
        if actual != expected:
            raise DataValidationError("performance observation source hash mismatch")


def _validate_drift_reference(reports_root: Path, health: dict[str, Any]) -> None:
    drift = health.get("drift_reference")
    if drift is None:
        return
    if not isinstance(drift, dict):
        raise DataValidationError("health drift_reference is invalid")
    manifest_path = drift.get("manifest_path")
    manifest_hash = drift.get("manifest_hash")
    if not isinstance(manifest_path, str) or not isinstance(manifest_hash, str):
        raise DataValidationError("health drift_reference lacks immutable manifest identity")
    path = Path(manifest_path).resolve()
    if not path.is_relative_to(reports_root.resolve()) or file_sha256(path) != manifest_hash:
        raise DataValidationError("health drift manifest hash mismatch")


def _reference(
    path: Path,
    reports_root: Path,
    payload: dict[str, Any],
    as_of: str,
) -> EvidenceReference:
    resolved = path.resolve()
    if not resolved.is_relative_to(reports_root):
        raise DataValidationError(f"retraining evidence escapes reports root: {path}")
    identity = payload.get("identity_hash") or payload.get("source_artifact_hash")
    return EvidenceReference(
        path=resolved.relative_to(reports_root).as_posix(),
        sha256=file_sha256(resolved),
        artifact_name=str(payload["artifact_name"]),
        as_of=as_of,
        identity_hash=str(identity) if identity is not None else None,
    )


def _identity(payload: dict[str, Any], name: str, as_of: str, description: str) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_name") != name
        or str(payload.get("as_of")) != as_of
    ):
        raise DataValidationError(f"invalid {description} identity")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required {description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid {description}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must contain an object")
    return payload
