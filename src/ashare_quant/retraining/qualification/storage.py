"""Atomic materialized qualification snapshots with append-only events."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.qualification.reporting import render_qualification_report
from ashare_quant.retraining.qualification.schemas import (
    QualificationCheckpoint,
    QualificationEvent,
    QualificationManifest,
    QualificationSnapshot,
    QualificationSummary,
)
from ashare_quant.utils.manifest import atomic_write_json


class QualificationStorage:
    """Publish immutable-identity qualification snapshots without mutating stage artifacts."""

    def __init__(self, reports_root: Path) -> None:
        self.root = reports_root / "retraining" / "qualification"
        self.staging_root = self.root / ".tmp"

    def output_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def read(self, run_id: str) -> QualificationSnapshot | None:
        output = self.output_dir(run_id)
        if not output.exists():
            return None
        required = (
            "qualification_summary.json",
            "qualification_events.parquet",
            "checkpoint_results.json",
            "source_inventory.json",
            "invariant_results.json",
            "report.md",
            "manifest.json",
        )
        if any(not (output / name).is_file() for name in required):
            raise DataValidationError(f"incomplete qualification requires recovery: {output}")
        try:
            summary = QualificationSummary.model_validate(_json(output / required[0]))
            manifest = QualificationManifest.model_validate(_json(output / "manifest.json"))
            checkpoints = {
                name: QualificationCheckpoint.model_validate(value)
                for name, value in _json(output / "checkpoint_results.json").items()
                if isinstance(value, dict)
            }
            inventory = _json(output / "source_inventory.json")
            invariants = _json(output / "invariant_results.json")
            frame = pd.read_parquet(output / "qualification_events.parquet")
            events = tuple(
                _event_from_record(cast(dict[str, Any], row)) for row in frame.to_dict("records")
            )
        except (OSError, ValueError, TypeError) as error:
            raise DataValidationError(f"invalid qualification snapshot: {error}") from error
        expected = {
            "summary_sha256": file_sha256(output / "qualification_summary.json"),
            "events_sha256": file_sha256(output / "qualification_events.parquet"),
            "checkpoints_sha256": file_sha256(output / "checkpoint_results.json"),
            "inventory_sha256": file_sha256(output / "source_inventory.json"),
            "invariants_sha256": file_sha256(output / "invariant_results.json"),
            "report_sha256": file_sha256(output / "report.md"),
        }
        if any(getattr(manifest, key) != digest for key, digest in expected.items()):
            raise DataValidationError("qualification snapshot hash mismatch")
        if not events or events[-1].state != summary.current_state:
            raise DataValidationError("qualification event tail differs from summary")
        if [event.sequence for event in events] != list(range(1, len(events) + 1)):
            raise DataValidationError("qualification event sequence is not contiguous")
        return QualificationSnapshot(summary, events, checkpoints, inventory, invariants, manifest)

    def publish(
        self, snapshot: QualificationSnapshot, manifest: QualificationManifest
    ) -> QualificationSnapshot:
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(exist_ok=True)
        output = self.output_dir(snapshot.summary.qualification_run_id)
        previous = self.read(snapshot.summary.qualification_run_id) if output.exists() else None
        if previous is not None:
            if (
                previous.manifest is None
                or previous.manifest.qualification_identity_hash
                != manifest.qualification_identity_hash
            ):
                raise DataValidationError("qualification identity cannot overwrite existing run")
            if snapshot.events[: len(previous.events)] != previous.events:
                raise DataValidationError("qualification event history is not append-only")
            _preserve_successful_checkpoints(previous, snapshot)
        staging = Path(tempfile.mkdtemp(dir=self.staging_root, prefix="qualification_"))
        backup = self.staging_root / f".{snapshot.summary.qualification_run_id}.backup"
        try:
            atomic_write_json(
                staging / "qualification_summary.json",
                snapshot.summary.model_dump(mode="json"),
            )
            frame = pd.DataFrame.from_records(
                [
                    {
                        "schema_version": event.schema_version,
                        "sequence": event.sequence,
                        "state": event.state,
                        "created_at": event.created_at,
                        "message": event.message,
                        "details_json": json.dumps(
                            event.details,
                            ensure_ascii=True,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                    for event in snapshot.events
                ]
            )
            frame.to_parquet(staging / "qualification_events.parquet", index=False)
            atomic_write_json(
                staging / "checkpoint_results.json",
                {
                    name: value.model_dump(mode="json")
                    for name, value in sorted(snapshot.checkpoints.items())
                },
            )
            atomic_write_json(staging / "source_inventory.json", snapshot.source_inventory)
            atomic_write_json(staging / "invariant_results.json", snapshot.invariant_results)
            (staging / "report.md").write_text(
                render_qualification_report(snapshot), encoding="utf-8"
            )
            completed = manifest.model_copy(
                update={
                    "summary_sha256": file_sha256(staging / "qualification_summary.json"),
                    "events_sha256": file_sha256(staging / "qualification_events.parquet"),
                    "checkpoints_sha256": file_sha256(staging / "checkpoint_results.json"),
                    "inventory_sha256": file_sha256(staging / "source_inventory.json"),
                    "invariants_sha256": file_sha256(staging / "invariant_results.json"),
                    "report_sha256": file_sha256(staging / "report.md"),
                }
            )
            atomic_write_json(staging / "manifest.json", completed.model_dump(mode="json"))
            if backup.exists():
                raise DataValidationError(f"stale qualification backup requires recovery: {backup}")
            if output.exists():
                os.replace(output, backup)
            try:
                os.replace(staging, output)
            except Exception:
                if backup.exists() and not output.exists():
                    os.replace(backup, output)
                raise
            published = self.read(snapshot.summary.qualification_run_id)
            if backup.exists():
                shutil.rmtree(backup)
            if published is None:
                raise DataValidationError("qualification publication disappeared")
            return published
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _preserve_successful_checkpoints(
    previous: QualificationSnapshot, current: QualificationSnapshot
) -> None:
    for name, old in previous.checkpoints.items():
        if old.status != "success":
            continue
        new = current.checkpoints.get(name)
        if new is None or new.status != "success":
            raise DataValidationError(f"successful qualification checkpoint disappeared: {name}")
        if not set(old.artifact_paths).issubset(new.artifact_paths):
            raise DataValidationError(f"qualification checkpoint artifact removed: {name}")
        for key, digest in old.artifact_hashes.items():
            if new.artifact_hashes.get(key) != digest:
                raise DataValidationError(f"qualification checkpoint hash changed: {name}:{key}")


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid qualification JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"qualification JSON must contain an object: {path}")
    return payload


def _event_from_record(record: dict[str, Any]) -> QualificationEvent:
    details = record.pop("details_json", "{}")
    try:
        parsed = json.loads(str(details))
    except json.JSONDecodeError as error:
        raise DataValidationError(f"invalid qualification event details: {error}") from error
    return QualificationEvent.model_validate({**record, "details": parsed})
