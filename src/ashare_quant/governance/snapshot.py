"""Atomic dated governance snapshots for the production closed loop."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ashare_quant.config.settings import AppSettings
from ashare_quant.governance.recovery import validate_recovery_state
from ashare_quant.governance.schemas import GovernanceCheck, GovernanceReport, overall_status
from ashare_quant.governance.status import SourceCatalog, collect_governance_status
from ashare_quant.governance.validation import validate_production_state
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json


@dataclass(frozen=True, slots=True)
class GovernanceSnapshotResult:
    snapshot_id: str
    artifact_paths: tuple[Path, ...]
    warnings: tuple[str, ...]


class DailyGovernanceSnapshotService:
    """Publish a read-only four-report governance snapshot for one production date."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        project_root: Path,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.project_root = project_root

    def publish_daily(self, as_of: str, *, production_run_id: str) -> GovernanceSnapshotResult:
        generated_at = datetime.now(UTC).isoformat()
        status_sources = SourceCatalog()
        status_summary, status_checks = collect_governance_status(
            settings=self.settings,
            project_root=self.project_root,
            sources=status_sources,
        )
        production = status_summary.get("production")
        if isinstance(production, dict) and production.get("latest_run_id") == production_run_id:
            production["pipeline_status"] = "success_pending_commit"
            status_checks = [
                item.model_copy(
                    update={
                        "status": "PASS",
                        "message": "current production run is ready for terminal commit",
                    }
                )
                if item.name == "production.latest"
                else item
                for item in status_checks
            ]
        validation_sources = SourceCatalog()
        validation_summary, validation_checks = validate_production_state(
            settings=self.settings,
            config_path=self.config_path,
            project_root=self.project_root,
            sources=validation_sources,
            expected_production_run_id=production_run_id,
        )
        recovery_sources = SourceCatalog()
        recovery_summary, recovery_checks = validate_recovery_state(
            settings=self.settings,
            sources=recovery_sources,
        )
        reports = {
            "status.json": _report(
                "status", status_summary, status_checks, status_sources, generated_at
            ),
            "validation.json": _report(
                "validation",
                validation_summary,
                validation_checks,
                validation_sources,
                generated_at,
            ),
            "recovery.json": _report(
                "recovery",
                recovery_summary,
                recovery_checks,
                recovery_sources,
                generated_at,
            ),
            "promotion_status.json": {
                "schema_version": 1,
                "artifact_name": "governance_promotion_status",
                "as_of": as_of,
                "production_run_id": production_run_id,
                "promotion": status_summary.get("promotion", {}),
                "rollback": status_summary.get("rollback", {}),
                "generated_at": generated_at,
                "read_only": True,
            },
        }
        identity = canonical_payload_hash(
            {
                "as_of": as_of,
                "production_run_id": production_run_id,
                "reports": {
                    name: {key: value for key, value in payload.items() if key != "generated_at"}
                    for name, payload in reports.items()
                },
            }
        )
        snapshot_id = f"governance_{as_of}_{identity[:16]}"
        paths = self._publish(as_of, snapshot_id, reports, generated_at)
        warnings = tuple(
            item.message
            for item in (*status_checks, *validation_checks, *recovery_checks)
            if item.status == "WARNING"
        )
        return GovernanceSnapshotResult(snapshot_id, paths, warnings)

    def _publish(
        self,
        as_of: str,
        snapshot_id: str,
        reports: dict[str, dict[str, Any]],
        generated_at: str,
    ) -> tuple[Path, ...]:
        date_root = self.settings.paths.reports / "governance" / as_of
        history = date_root / "history" / snapshot_id
        if not history.exists():
            history.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(dir=history.parent, prefix=f".{snapshot_id}."))
            try:
                for name, payload in reports.items():
                    atomic_write_json(staging / name, payload)
                hashes = {name: file_sha256(staging / name) for name in sorted(reports)}
                atomic_write_json(
                    staging / "manifest.json",
                    {
                        "schema_version": 1,
                        "artifact_name": "daily_governance_snapshot",
                        "snapshot_id": snapshot_id,
                        "as_of": as_of,
                        "artifact_hashes": hashes,
                        "created_at": generated_at,
                        "manifest_written_last": True,
                    },
                )
                os.replace(staging, history)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        manifest = _load_json(history / "manifest.json")
        raw_hashes = manifest.get("artifact_hashes")
        hashes = (
            {str(key): str(value) for key, value in raw_hashes.items()}
            if isinstance(raw_hashes, dict)
            else {}
        )
        if not hashes or any(
            file_sha256(history / name) != digest for name, digest in hashes.items()
        ):
            raise ValueError("immutable daily governance snapshot hash mismatch")
        for name in reports:
            atomic_write_json(date_root / name, _load_json(history / name))
        atomic_write_json(date_root / "manifest.json", manifest)
        return tuple(date_root / name for name in (*reports, "manifest.json"))


def _report(
    report_type: Literal["status", "validation", "recovery"],
    summary: dict[str, Any],
    checks: list[GovernanceCheck],
    sources: SourceCatalog,
    generated_at: str,
) -> dict[str, Any]:
    report = GovernanceReport(
        artifact_name=f"governance_{report_type}_report",
        report_type=report_type,
        status=overall_status(checks),
        generated_at=generated_at,
        summary=summary,
        checks=tuple(checks),
        source_hashes=dict(sorted(sources.hashes.items())),
    )
    return report.model_dump(mode="json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"governance snapshot JSON must contain an object: {path}")
    return payload
