"""Read-only recovery inspection for qualification snapshots."""

from __future__ import annotations

from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.qualification.invariants import compare_protected_state
from ashare_quant.retraining.qualification.schemas import QualificationRecovery
from ashare_quant.retraining.qualification.storage import QualificationStorage


def inspect_qualification_recovery(
    storage: QualificationStorage, run_id: str
) -> QualificationRecovery:
    issues: list[str] = []
    actions: list[str] = []
    output = storage.output_dir(run_id)
    if not output.exists():
        return QualificationRecovery(run_id, "MISSING", ("qualification does not exist",), ())
    try:
        snapshot = storage.read(run_id)
    except (DataValidationError, OSError, ValueError) as error:
        issues.append(str(error))
        actions.append("inspect the incomplete snapshot and preserve it for forensic review")
        snapshot = None
    for stale_path in (
        sorted(storage.staging_root.glob("*")) if storage.staging_root.is_dir() else ()
    ):
        issues.append(f"stale qualification staging path: {stale_path}")
    backup = storage.staging_root / f".{run_id}.backup"
    if backup.exists():
        issues.append(f"stale qualification backup: {backup}")
    if snapshot is not None:
        for name, source in snapshot.source_inventory.items():
            source_path = source.get("path")
            digest = source.get("sha256")
            if not isinstance(source_path, str) or not isinstance(digest, str):
                issues.append(f"invalid source inventory entry: {name}")
                continue
            try:
                if file_sha256(Path(source_path)) != digest:
                    issues.append(f"qualification source changed: {name}")
            except DataValidationError:
                issues.append(f"qualification source missing: {name}")
        baseline = snapshot.invariant_results.get("baseline")
        current = snapshot.invariant_results.get("current")
        if isinstance(baseline, dict) and isinstance(current, dict):
            issues.extend(
                f"protected invariant changed: {name}"
                for name in compare_protected_state(baseline, current)
            )
        if snapshot.summary.current_state in {"TRAINING", "VALIDATING", "SHADOW_ENROLLING"}:
            issues.append(f"ambiguous interrupted state: {snapshot.summary.current_state}")
    if issues:
        actions.append("run qualification-status and inspect referenced immutable stage manifests")
    return QualificationRecovery(
        run_id,
        "ACTION_REQUIRED" if issues else "CLEAN",
        tuple(dict.fromkeys(issues)),
        tuple(dict.fromkeys(actions)),
    )
