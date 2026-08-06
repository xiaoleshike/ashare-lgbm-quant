"""Non-destructive lifecycle recovery inspection."""

from __future__ import annotations

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.orchestration.schemas import RecoveryInspection
from ashare_quant.retraining.orchestration.storage import LifecycleStorage


def inspect_lifecycle_recovery(
    storage: LifecycleStorage, lifecycle_run_id: str
) -> RecoveryInspection:
    """Inspect complete snapshots and staging remnants without repairing them."""

    staging = tuple(sorted(str(path) for path in storage.staging_root.glob("*") if path.exists()))
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
    status = "CLEAN" if not staging else "STAGING_REVIEW_REQUIRED"
    return RecoveryInspection(
        lifecycle_run_id,
        status,
        snapshot.summary.current_state,
        output.is_dir(),
        staging,
        "snapshot is complete" if not staging else "unpublished staging paths exist",
    )
