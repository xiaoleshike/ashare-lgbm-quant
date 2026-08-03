"""Read-only validation of registry and interrupted-transaction recovery inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
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


@dataclass(frozen=True, slots=True)
class RegistryRecoveryPreview:
    latest_valid_registry: Path
    registry_hash: str
    champion_model_id: str
    champion_assignment_id: str | None
    registry_version_id: str | None
    transition_manifest: Path | None
    corrupted_versions: tuple[str, ...]


def registry_recovery_preview(models_root: Path) -> RegistryRecoveryPreview:
    """Select the latest lineage-valid registry version without restoring it."""

    versions_root = models_root / "registry_versions"
    valid: list[tuple[str, Path, dict[str, Any], dict[str, Any] | None, Path | None]] = []
    corrupted: list[str] = []
    for path in sorted(versions_root.glob("*.json")) if versions_root.exists() else []:
        try:
            records = load_registry_records(path)
            payload = _load_json(path)
            champions = [item for item in records if item.status == "champion"]
            if len(champions) != 1:
                raise ValueError("registry version does not have exactly one Champion")
            assignment, transition = _registry_transition_lineage(models_root, path, payload)
            valid.append(
                (str(payload.get("updated_at") or ""), path, payload, assignment, transition)
            )
        except Exception:
            corrupted.append(path.name)
    current = models_root / "registry.json"
    if current.is_file():
        try:
            records = load_registry_records(current)
            payload = _load_json(current)
            champions = [item for item in records if item.status == "champion"]
            if len(champions) == 1:
                assignment, transition = _registry_transition_lineage(models_root, current, payload)
                valid.append(
                    (str(payload.get("updated_at") or ""), current, payload, assignment, transition)
                )
        except Exception:  # noqa: S110 - a corrupted current file is reported by validation.
            pass
    if not valid:
        raise DataValidationError("no valid registry version is available for manual recovery")
    _, path, payload, assignment, transition = max(valid, key=lambda item: (item[0], str(item[1])))
    records = load_registry_records(path)
    champion = next(item for item in records if item.status == "champion")
    version_id = payload.get("registry_version_id")
    assignment_id = (
        str(assignment.get("champion_assignment_id")) if assignment is not None else None
    )
    return RegistryRecoveryPreview(
        latest_valid_registry=path,
        registry_hash=file_sha256(path),
        champion_model_id=champion.model_id,
        champion_assignment_id=assignment_id,
        registry_version_id=None if version_id is None else str(version_id),
        transition_manifest=transition,
        corrupted_versions=tuple(corrupted),
    )


def _registry_transition_lineage(
    models_root: Path,
    registry_path: Path,
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, Path | None]:
    """Validate history and apply commit marker for a governed Registry version."""

    version_id = payload.get("registry_version_id")
    if not isinstance(version_id, str):
        return None, None
    history_root = models_root / "champion_history"
    assignments: list[dict[str, Any]] = []
    if history_root.exists():
        for path in sorted(history_root.glob("*.json")):
            assignment = _load_json(path)
            if assignment.get("registry_version_id") == version_id:
                assignments.append(assignment)
    if len(assignments) != 1:
        raise ValueError("governed Registry version lacks one Champion assignment")
    assignment = assignments[0]
    request_id = payload.get("promotion_request_id") or payload.get("rollback_request_id")
    if not isinstance(request_id, str):
        raise ValueError("governed Registry version lacks transition request identity")
    if payload.get("promotion_request_id") is not None:
        manifests = sorted(
            (models_root / "promotion_requests" / request_id / "apply").glob("*/manifest.json")
        )
    else:
        rollback = models_root / "rollback_requests" / request_id / "rollback_apply_manifest.json"
        manifests = [rollback] if rollback.is_file() else []
    matching = []
    for manifest_path in manifests:
        manifest = _load_json(manifest_path)
        if manifest.get("registry_version_id") == version_id:
            matching.append((manifest_path, manifest))
    if len(matching) != 1:
        raise ValueError("governed Registry version lacks one apply commit manifest")
    transition_path, transition = matching[0]
    if transition.get("registry_file_hash") != file_sha256(registry_path):
        raise ValueError("apply manifest Registry hash differs")
    assignment_path = history_root / f"{assignment.get('champion_assignment_id')}.json"
    if transition.get("champion_history_hash") != file_sha256(assignment_path):
        raise ValueError("apply manifest Champion history hash differs")
    return assignment, transition_path
