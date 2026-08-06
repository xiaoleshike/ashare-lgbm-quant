"""Non-destructive lifecycle recovery inspection."""

from __future__ import annotations

import json

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.orchestration.schemas import RecoveryInspection
from ashare_quant.retraining.orchestration.storage import LifecycleStorage


def inspect_lifecycle_recovery(
    storage: LifecycleStorage, lifecycle_run_id: str
) -> RecoveryInspection:
    """Inspect complete snapshots and staging remnants without repairing them."""

    remnants = [path for path in storage.staging_root.glob("*") if path.exists()]
    dry_run_root = storage.root.parent / "lifecycle_dry_runs"
    if dry_run_root.is_dir():
        remnants.extend(
            path
            for path in dry_run_root.iterdir()
            if path.name.startswith(".dry-run-")
            or (path.is_dir() and not (path / "manifest.json").is_file())
        )
    staging = tuple(sorted(str(path) for path in remnants))
    output = storage.output_dir(lifecycle_run_id)
    try:
        snapshot = storage.read(lifecycle_run_id)
    except DataValidationError as error:
        return RecoveryInspection(
            lifecycle_run_id,
            "MANUAL_RECOVERY_REQUIRED",
            None,
            False,
            staging,
            str(error),
        )
    if snapshot is None:
        return RecoveryInspection(
            lifecycle_run_id, "MISSING", None, False, staging, "lifecycle run does not exist"
        )
    warnings = _snapshot_warnings(snapshot)
    status = (
        "MANUAL_RECOVERY_REQUIRED"
        if warnings
        else "CLEAN"
        if not staging
        else "STAGING_REVIEW_REQUIRED"
    )
    return RecoveryInspection(
        lifecycle_run_id,
        status,
        snapshot.summary.current_state,
        output.is_dir(),
        staging,
        (
            "; ".join(warnings)
            if warnings
            else "snapshot is complete"
            if not staging
            else "unpublished staging or backup paths exist"
        ),
        tuple(warnings),
    )


def _snapshot_warnings(snapshot: object) -> list[str]:
    from ashare_quant.retraining.orchestration.schemas import LifecycleSnapshot

    if not isinstance(snapshot, LifecycleSnapshot):
        return ["invalid lifecycle snapshot type"]
    warnings: list[str] = []
    summary = snapshot.summary
    enrolled_states = {
        "SHADOW_ENROLLED",
        "OBSERVATION_PENDING",
        "OBSERVATION_ACCUMULATING",
        "OBSERVATION_SUFFICIENT",
        "POLICY_REVIEW_REQUIRED",
        "EVIDENCE_READY",
    }
    successful_shadow = [
        stage
        for name, stage in snapshot.stage_results.items()
        if name in {"shadow", "shadow_enrollment", "shadow_refresh"}
        and stage.status == "success"
        and stage.artifact_paths
    ]
    if summary.current_state in enrolled_states and not successful_shadow:
        warnings.append("enrolled lifecycle lacks successful verifiable Shadow evidence")
    if summary.current_state in enrolled_states and not summary.successful_shadow_run_ids:
        legacy = snapshot.stage_results.get("shadow")
        if legacy is None or not _legacy_shadow_is_verifiable(snapshot, legacy.artifact_paths):
            warnings.append("legacy enrolled lifecycle has ambiguous Shadow lineage")
    observation = snapshot.stage_results.get("observation")
    if observation is not None:
        progress_events = [
            event for event in snapshot.events if "observation progress" in event.message
        ]
        if summary.mature_sessions and not progress_events:
            warnings.append("observation summary changed without append-only progress event")
        aggregate = observation.metrics.get("aggregate_hash")
        if summary.observation_evidence_hash and aggregate != summary.observation_evidence_hash:
            warnings.append("observation aggregate hash differs from lifecycle summary")
    if summary.evidence_stale:
        warnings.append("promotion evidence evaluation is stale under current policy")
    for name, stage in snapshot.stage_results.items():
        if stage.status != "success":
            continue
        expected = set(stage.artifact_hashes.values())
        for raw in stage.artifact_paths:
            from pathlib import Path

            path = Path(raw)
            if not path.is_file():
                warnings.append(f"missing immutable source artifact: {name}:{path}")
            elif file_sha256(path) not in expected:
                warnings.append(f"changed immutable source artifact: {name}:{path}")
    return warnings


def _legacy_shadow_is_verifiable(snapshot: object, paths: tuple[str, ...]) -> bool:
    from pathlib import Path

    from ashare_quant.retraining.orchestration.schemas import LifecycleSnapshot

    if not isinstance(snapshot, LifecycleSnapshot) or not paths:
        return False
    for raw in paths:
        try:
            payload = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        expected = {
            "model_id": snapshot.summary.model_id,
            "model_origin": "retrained_challenger",
            "training_run_id": snapshot.summary.training_run_id,
            "validation_run_id": snapshot.summary.validation_run_id,
            "access_policy": "prospective_production",
        }
        if (
            any(payload.get(key) != value for key, value in expected.items())
            or not payload.get("training_request_id")
            or not payload.get("shadow_run_id")
        ):
            return False
    return True
