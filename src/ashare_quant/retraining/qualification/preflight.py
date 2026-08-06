"""Read-only preflight for controlled operational qualification."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.orchestration.lock import detect_production_lock_owner
from ashare_quant.retraining.orchestration.schemas import LifecycleInput
from ashare_quant.retraining.orchestration.storage import LifecycleStorage
from ashare_quant.retraining.qualification.schemas import QualificationCheck


def run_preflight(
    *,
    frozen: LifecycleInput,
    as_of: str,
    project_root: Path,
    reports_root: Path,
    models_root: Path,
    processed_root: Path,
    config_path: Path,
    retraining_policy_path: Path,
    promotion_policy_path: Path,
    qualification_enabled: bool,
    require_clean_worktree: bool,
    git_dirty: bool,
    minimum_free_disk_bytes: int | None,
    minimum_available_memory_bytes: int | None,
    production_lock_path: Path,
    lifecycle_lock_path: Path,
) -> tuple[tuple[QualificationCheck, ...], dict[str, dict[str, object]]]:
    checks: list[QualificationCheck] = []
    inventory: dict[str, dict[str, object]] = {}

    def record(name: str, path: Path) -> None:
        if not path.is_file():
            raise DataValidationError(f"required immutable source is missing: {path}")
        inventory[name] = {"path": str(path), "sha256": file_sha256(path)}

    try:
        if not qualification_enabled:
            raise DataValidationError("operational qualification is disabled")
        if frozen.request.as_of != as_of:
            raise DataValidationError("qualification as_of differs from Training Request")
        if len(frozen.request.target_models) != 1:
            raise DataValidationError("qualification requires exactly one target model")
        target = frozen.request.target_models[0]
        if target.horizon not in {5, 10, 20, 60}:
            raise DataValidationError("qualification horizon is unsupported")
        parent = next(
            (
                item
                for item in ModelRegistry(models_root).list_models()
                if item.model_id == target.model_id
            ),
            None,
        )
        if parent is None:
            raise DataValidationError("qualification parent model is absent from Registry")
        for name, path in (
            ("config", config_path),
            ("retraining_policy", retraining_policy_path),
            ("promotion_policy", promotion_policy_path),
            (
                "training_request",
                reports_root
                / "retraining"
                / "requests"
                / frozen.request.request_id
                / "training_request.json",
            ),
            (
                "training_request_manifest",
                reports_root
                / "retraining"
                / "requests"
                / frozen.request.request_id
                / "manifest.json",
            ),
            ("feature_manifest", processed_root / "features_daily" / "_manifest.json"),
            ("universe_manifest", processed_root / "universe_daily" / "_manifest.json"),
            ("label_manifest", processed_root / "labels_forward" / "_manifest.json"),
        ):
            record(name, path)
        for name, reference in (
            ("monitor_evidence", frozen.request.evidence.monitor_snapshot),
            ("observation_evidence", frozen.request.evidence.performance_observation),
            ("alert_evidence", frozen.request.evidence.alerts),
        ):
            source = reports_root / reference.path
            record(name, source)
            if inventory[name]["sha256"] != reference.sha256:
                raise DataValidationError(f"training request evidence hash changed: {name}")
        checks.append(
            QualificationCheck(name="request_lineage", status="PASS", message="validated")
        )
    except (DataValidationError, OSError, ValueError) as error:
        checks.append(QualificationCheck(name="request_lineage", status="FAIL", message=str(error)))

    for name, path in (
        ("production_lock", production_lock_path),
        ("lifecycle_lock", lifecycle_lock_path),
    ):
        owner = detect_production_lock_owner(path)
        checks.append(
            QualificationCheck(
                name=name,
                status="PASS" if owner is None else "FAIL",
                message="available" if owner is None else owner.describe(),
            )
        )

    previous = LifecycleStorage(reports_root).find_by_request(frozen.request.request_id)
    conflict = previous is not None and any(event.state == "TRAINING" for event in previous.events)
    checks.append(
        QualificationCheck(
            name="conflicting_lifecycle",
            status="FAIL" if conflict else "PASS",
            message=(
                "existing lifecycle already entered training"
                if conflict
                else "no training conflict"
            ),
        )
    )
    free_disk = shutil.disk_usage(project_root).free
    disk_status: Literal["PASS", "FAIL", "WARN"] = (
        "FAIL"
        if minimum_free_disk_bytes is not None and free_disk < minimum_free_disk_bytes
        else ("PASS" if minimum_free_disk_bytes is not None else "WARN")
    )
    checks.append(
        QualificationCheck(
            name="disk_capacity",
            status=disk_status,
            message=(
                "configured threshold checked"
                if minimum_free_disk_bytes is not None
                else "no hard threshold configured"
            ),
            details={"free_bytes": free_disk, "minimum_bytes": minimum_free_disk_bytes},
        )
    )
    available_memory = _available_memory()
    memory_status: Literal["PASS", "FAIL", "WARN"] = (
        "FAIL"
        if minimum_available_memory_bytes is not None
        and (available_memory is None or available_memory < minimum_available_memory_bytes)
        else ("PASS" if minimum_available_memory_bytes is not None else "WARN")
    )
    checks.append(
        QualificationCheck(
            name="memory_capacity",
            status=memory_status,
            message=(
                "configured threshold checked"
                if minimum_available_memory_bytes is not None
                else "no hard threshold configured"
            ),
            details={
                "available_bytes": available_memory,
                "minimum_bytes": minimum_available_memory_bytes,
            },
        )
    )
    checks.append(
        QualificationCheck(
            name="git_worktree",
            status="FAIL"
            if require_clean_worktree and git_dirty
            else ("WARN" if git_dirty else "PASS"),
            message="dirty" if git_dirty else "clean",
        )
    )
    return tuple(checks), inventory


def _available_memory() -> int | None:
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, KeyError):
        return None
