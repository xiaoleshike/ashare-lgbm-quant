"""Read-only validation of registry and interrupted-transaction recovery inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.config.settings import AppSettings
from ashare_quant.governance.schemas import GovernanceCheck
from ashare_quant.governance.status import SourceCatalog
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.promotion.registry_versions import load_registry_records
from ashare_quant.models.promotion.schemas import DeploymentContract
from ashare_quant.models.shadow.storage import file_sha256


def validate_recovery_state(
    *, settings: AppSettings, sources: SourceCatalog
) -> tuple[dict[str, Any], list[GovernanceCheck]]:
    """Prove recovery inputs are readable without changing current state."""

    models_root = settings.paths.models
    checks: list[GovernanceCheck] = []
    recoverable_versions = _registry_recovery(models_root, sources, checks)
    _artifact_recovery(models_root, sources, checks)
    interrupted = _interrupted_transactions(models_root, checks)
    incomplete = _incomplete_publications(models_root, checks)
    return {
        "recoverable_registry_versions": recoverable_versions,
        "interrupted_transactions": interrupted,
        "incomplete_publications": incomplete,
    }, checks


def _registry_recovery(
    models_root: Path, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> list[str]:
    registry = models_root / "registry.json"
    if not registry.is_file():
        checks.append(
            GovernanceCheck(
                name="recovery.current_registry",
                status="FAIL",
                message="registry.json is missing",
                source_path=str(registry),
            )
        )
    else:
        try:
            load_registry_records(registry)
            sources.track(registry)
            checks.append(
                GovernanceCheck(
                    name="recovery.current_registry",
                    status="PASS",
                    message="current registry is valid",
                    source_path=str(registry),
                )
            )
        except Exception as error:
            checks.append(
                GovernanceCheck(
                    name="recovery.current_registry",
                    status="FAIL",
                    message=f"current registry is corrupted: {error}",
                    source_path=str(registry),
                )
            )
    valid: list[str] = []
    invalid: list[str] = []
    root = models_root / "registry_versions"
    if root.exists():
        for path in sorted(root.glob("*.json")):
            try:
                load_registry_records(path)
                sources.track(path)
                valid.append(path.name)
            except Exception:
                invalid.append(path.name)
    if invalid:
        checks.append(
            GovernanceCheck(
                name="recovery.registry_versions",
                status="FAIL",
                message=f"corrupted registry versions={invalid}",
                details={"invalid": invalid},
            )
        )
    elif valid:
        checks.append(
            GovernanceCheck(
                name="recovery.registry_versions",
                status="PASS",
                message=f"recoverable registry versions={len(valid)}",
            )
        )
    else:
        checks.append(
            GovernanceCheck(
                name="recovery.registry_versions",
                status="WARNING",
                message=(
                    "no immutable registry version exists; current registry is the only "
                    "recovery source"
                ),
            )
        )
    history_root = models_root / "champion_history"
    invalid_history: list[str] = []
    history = sorted(history_root.glob("*.json")) if history_root.exists() else []
    for path in history:
        try:
            payload = _load_json(path)
            required = {"champion_assignment_id", "model_id", "registry_version_id", "activated_at"}
            if not required.issubset(payload):
                raise ValueError("missing assignment fields")
            version = root / f"{payload['registry_version_id']}.json"
            if not version.is_file():
                raise ValueError("referenced registry version is missing")
            sources.track(path)
        except Exception:
            invalid_history.append(path.name)
    checks.append(
        GovernanceCheck(
            name="recovery.champion_history",
            status="FAIL" if invalid_history else ("PASS" if history else "WARNING"),
            message=f"invalid champion history={invalid_history}"
            if invalid_history
            else (
                f"Champion assignments={len(history)}" if history else "no Champion history exists"
            ),
        )
    )
    return valid


def _artifact_recovery(
    models_root: Path, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> None:
    registry = models_root / "registry.json"
    if not registry.is_file():
        return
    try:
        records = load_registry_records(registry)
    except Exception:
        return
    invalid: list[str] = []
    for record in records:
        artifact = Path(record.artifact_path)
        files = ("model.txt", "feature_list.json", "manifest.json", "metrics.json")
        if any(not (artifact / name).is_file() for name in files):
            invalid.append(record.model_id)
            continue
        try:
            payload = _load_json(artifact / "feature_list.json")
            names = payload.get("features")
            if (
                not isinstance(names, list)
                or feature_list_hash(tuple(map(str, names))) != record.feature_hash
            ):
                raise ValueError("feature hash mismatch")
            for name in files:
                sources.track(artifact / name)
        except Exception:
            invalid.append(record.model_id)
    # Promotion contracts freeze exact model bytes; verify every completed request.
    request_root = models_root / "promotion_requests"
    if request_root.exists():
        for path in sorted(request_root.glob("*/deployment_contract.json")):
            try:
                contract = DeploymentContract.model_validate(_load_json(path))
                record = next(item for item in records if item.model_id == contract.model_id)
                artifact = Path(record.artifact_path)
                if any(
                    file_sha256(artifact / name) != digest
                    for name, digest in contract.artifact_hashes.items()
                ):
                    invalid.append(contract.model_id)
                sources.track(path)
            except (ValidationError, StopIteration, OSError, ValueError):
                invalid.append(path.parent.name)
    invalid = sorted(set(invalid))
    checks.append(
        GovernanceCheck(
            name="recovery.model_artifacts",
            status="FAIL" if invalid else "PASS",
            message=f"invalid model artifacts={invalid}"
            if invalid
            else f"validated model artifacts={len(records)}",
            details={"invalid_model_ids": invalid},
        )
    )


def _interrupted_transactions(models_root: Path, checks: list[GovernanceCheck]) -> list[str]:
    pending: list[str] = []
    promotion_root = models_root / "promotion_requests"
    if promotion_root.exists():
        for path in promotion_root.glob("*/apply/*/apply_pending.json"):
            if not (path.parent / "manifest.json").is_file():
                pending.append(str(path))
    rollback_root = models_root / "rollback_requests"
    if rollback_root.exists():
        for path in rollback_root.glob("*/rollback_apply_pending.json"):
            if not (path.parent / "rollback_apply_manifest.json").is_file():
                pending.append(str(path))
    checks.append(
        GovernanceCheck(
            name="recovery.interrupted_transactions",
            status="WARNING" if pending else "PASS",
            message=f"interrupted journals={pending}"
            if pending
            else "no interrupted apply journal exists",
            details={"paths": pending},
        )
    )
    return pending


def _incomplete_publications(models_root: Path, checks: list[GovernanceCheck]) -> list[str]:
    incomplete: list[str] = []
    for root_name in (
        "promotion_requests",
        "rollback_requests",
        "registry_versions",
        "champion_history",
    ):
        root = models_root / root_name
        if not root.exists():
            continue
        incomplete.extend(str(path) for path in root.iterdir() if path.name.startswith("."))
    for root_name in ("promotion_requests", "rollback_requests"):
        root = models_root / root_name
        if not root.exists():
            continue
        for directory in (
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        ):
            if not (directory / "manifest.json").is_file():
                incomplete.append(str(directory))
    checks.append(
        GovernanceCheck(
            name="recovery.incomplete_publications",
            status="WARNING" if incomplete else "PASS",
            message=f"incomplete staging/publications={incomplete}"
            if incomplete
            else "no incomplete staging directory exists",
            details={"paths": incomplete},
        )
    )
    return sorted(incomplete)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload
