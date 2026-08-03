"""Interrupted retraining detection without automatic model continuation."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ashare_quant.retraining.execution.lifecycle import LifecycleJournal


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    training_run_id: str
    status: str
    staging_paths: tuple[str, ...]
    retry_allowed: bool


def recover_interrupted(
    *, reports_root: Path, training_run_id: str, cleanup: bool = True
) -> RecoveryResult:
    """Mark a known incomplete run interrupted and remove only its unpublished staging."""

    journal = LifecycleJournal(reports_root / "retraining" / "execution_journals", training_run_id)
    events = journal.events()
    latest = events[-1].status if events else "MISSING"
    staging_root = reports_root / "retraining" / ".tmp"
    staging = tuple(sorted(staging_root.glob(f"execution_{training_run_id}_*")))
    if latest in {"CREATED", "DATA_READY", "TRAINING", "ARTIFACT_VALIDATING"}:
        journal.append("INTERRUPTED", "operator recovery check marked incomplete execution")
        latest = "INTERRUPTED"
    if cleanup:
        for path in staging:
            if path.is_dir():
                shutil.rmtree(path)
    return RecoveryResult(
        training_run_id,
        latest,
        tuple(str(path) for path in staging),
        latest in {"FAILED", "INTERRUPTED"},
    )
