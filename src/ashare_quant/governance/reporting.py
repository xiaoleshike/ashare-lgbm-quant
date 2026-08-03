"""Atomic immutable storage for governance reports."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.governance.schemas import GovernanceManifest, GovernanceReport
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json


class GovernanceReportPublisher:
    """Publish immutable snapshots plus a validated latest projection."""

    def __init__(self, reports_root: Path) -> None:
        self.root = reports_root / "governance"

    def publish(self, report: GovernanceReport) -> tuple[Path, Path]:
        """Publish one report atomically, with its completion manifest last."""

        logical = report.model_dump(mode="json", exclude={"generated_at"})
        snapshot_id = f"{report.report_type}_{canonical_payload_hash(logical)[:24]}"
        snapshot = self.root / "history" / report.report_type / snapshot_id
        report_name = f"{report.report_type}.json"
        if snapshot.exists():
            self._validate_snapshot(snapshot, report_name, snapshot_id)
        else:
            self._publish_snapshot(snapshot, report_name, snapshot_id, report)

        snapshot_report = snapshot / report_name
        snapshot_manifest = snapshot / "manifest.json"
        latest_report = self.root / report_name
        latest_manifest = self.root / f"{report.report_type}.manifest.json"
        atomic_write_json(latest_report, _load_json(snapshot_report))
        atomic_write_json(latest_manifest, _load_json(snapshot_manifest))
        if file_sha256(latest_report) != str(_load_json(latest_manifest)["report_sha256"]):
            raise DataValidationError("governance latest report publication hash mismatch")
        return latest_report, latest_manifest

    def _publish_snapshot(
        self,
        snapshot: Path,
        report_name: str,
        snapshot_id: str,
        report: GovernanceReport,
    ) -> None:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=snapshot.parent, prefix=f".{snapshot_id}."))
        try:
            report_path = staging / report_name
            atomic_write_json(report_path, report.model_dump(mode="json"))
            manifest = GovernanceManifest(
                report_type=report.report_type,
                snapshot_id=snapshot_id,
                report_sha256=file_sha256(report_path),
                source_hashes=report.source_hashes,
                created_at=report.generated_at,
            )
            # Completion marker is intentionally the final file written in staging.
            atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.replace(staging, snapshot)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @staticmethod
    def _validate_snapshot(snapshot: Path, report_name: str, snapshot_id: str) -> None:
        manifest_path = snapshot / "manifest.json"
        report_path = snapshot / report_name
        if not manifest_path.is_file() or not report_path.is_file():
            raise DataValidationError(f"incomplete immutable governance snapshot: {snapshot}")
        manifest = GovernanceManifest.model_validate(_load_json(manifest_path))
        if manifest.snapshot_id != snapshot_id:
            raise DataValidationError("governance snapshot identity mismatch")
        if file_sha256(report_path) != manifest.report_sha256:
            raise DataValidationError("immutable governance report hash mismatch")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid governance JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"governance JSON must contain an object: {path}")
    return payload
