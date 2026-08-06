"""Atomic lifecycle snapshots backed by append-only logical events."""

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
from ashare_quant.retraining.orchestration.reporting import render_lifecycle_report
from ashare_quant.retraining.orchestration.schemas import (
    LifecycleEvent,
    LifecycleManifest,
    LifecycleSnapshot,
    LifecycleSummary,
    StageResult,
)
from ashare_quant.utils.manifest import atomic_write_json


class LifecycleStorage:
    """Publish a recoverable current snapshot without mutating stage artifacts."""

    def __init__(self, reports_root: Path) -> None:
        self.root = reports_root / "retraining" / "lifecycle"
        self.staging_root = self.root / ".tmp"

    def output_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def find_by_request(self, request_id: str) -> LifecycleSnapshot | None:
        """Return the one immutable lifecycle bound to a request, failing on ambiguity."""

        if not self.root.is_dir():
            return None
        matches: list[LifecycleSnapshot] = []
        for directory in sorted(path for path in self.root.iterdir() if path.is_dir()):
            if directory.name == ".tmp":
                continue
            snapshot = self.read(directory.name)
            if snapshot is not None and snapshot.summary.request_id == request_id:
                matches.append(snapshot)
        if len(matches) > 1:
            raise DataValidationError(f"request has conflicting lifecycle identities: {request_id}")
        return matches[0] if matches else None

    def read(self, run_id: str) -> LifecycleSnapshot | None:
        output = self.output_dir(run_id)
        if not output.exists():
            return None
        required = (
            "lifecycle_summary.json",
            "lifecycle_events.parquet",
            "stage_results.json",
            "report.md",
            "manifest.json",
        )
        if any(not (output / name).is_file() for name in required):
            raise DataValidationError(f"incomplete lifecycle directory requires recovery: {output}")
        try:
            summary = LifecycleSummary.model_validate(_json(output / "lifecycle_summary.json"))
            manifest = LifecycleManifest.model_validate(_json(output / "manifest.json"))
            raw_stages = _json(output / "stage_results.json")
            frame = pd.read_parquet(output / "lifecycle_events.parquet")
            events = tuple(
                _event_from_record(cast(dict[str, Any], record))
                for record in frame.to_dict("records")
            )
            stages = {
                name: StageResult.model_validate(value)
                for name, value in raw_stages.items()
                if isinstance(value, dict)
            }
        except (OSError, ValueError, TypeError) as error:
            raise DataValidationError(f"invalid lifecycle snapshot: {error}") from error
        expected = {
            "summary_sha256": file_sha256(output / "lifecycle_summary.json"),
            "events_sha256": file_sha256(output / "lifecycle_events.parquet"),
            "stage_results_sha256": file_sha256(output / "stage_results.json"),
            "report_sha256": file_sha256(output / "report.md"),
        }
        if any(getattr(manifest, name) != digest for name, digest in expected.items()):
            raise DataValidationError("lifecycle snapshot hash mismatch")
        if not events or events[-1].state != summary.current_state:
            raise DataValidationError("lifecycle event tail differs from summary state")
        if [event.sequence for event in events] != list(range(1, len(events) + 1)):
            raise DataValidationError("lifecycle event sequence is not append-only")
        return LifecycleSnapshot(summary, events, stages, manifest)

    def publish(
        self,
        snapshot: LifecycleSnapshot,
        manifest: LifecycleManifest,
    ) -> LifecycleSnapshot:
        """Atomically replace the materialized snapshot while preserving event prefix."""

        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(exist_ok=True)
        output = self.output_dir(snapshot.summary.lifecycle_run_id)
        previous = self.read(snapshot.summary.lifecycle_run_id) if output.exists() else None
        if previous is not None:
            if previous.manifest is None or (
                previous.manifest.lifecycle_identity_hash != manifest.lifecycle_identity_hash
            ):
                raise DataValidationError("lifecycle identity cannot overwrite existing run")
            if snapshot.events[: len(previous.events)] != previous.events:
                raise DataValidationError("lifecycle event history is not append-only")
            _require_successful_stage_evidence_preserved(previous, snapshot)
        staging = Path(tempfile.mkdtemp(dir=self.staging_root, prefix="lifecycle_"))
        backup = self.staging_root / f".{snapshot.summary.lifecycle_run_id}.backup"
        try:
            atomic_write_json(
                staging / "lifecycle_summary.json",
                snapshot.summary.model_dump(mode="json"),
            )
            event_frame = pd.DataFrame.from_records(
                [
                    {
                        "schema_version": event.schema_version,
                        "sequence": event.sequence,
                        "state": event.state,
                        "created_at": event.created_at,
                        "message": event.message,
                        "details_json": json.dumps(
                            event.details, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                        ),
                    }
                    for event in snapshot.events
                ]
            )
            event_frame.to_parquet(staging / "lifecycle_events.parquet", index=False)
            atomic_write_json(
                staging / "stage_results.json",
                {
                    name: result.model_dump(mode="json")
                    for name, result in sorted(snapshot.stage_results.items())
                },
            )
            (staging / "report.md").write_text(render_lifecycle_report(snapshot), encoding="utf-8")
            completed = manifest.model_copy(
                update={
                    "summary_sha256": file_sha256(staging / "lifecycle_summary.json"),
                    "events_sha256": file_sha256(staging / "lifecycle_events.parquet"),
                    "stage_results_sha256": file_sha256(staging / "stage_results.json"),
                    "report_sha256": file_sha256(staging / "report.md"),
                }
            )
            atomic_write_json(staging / "manifest.json", completed.model_dump(mode="json"))
            _validate_staged_snapshot(staging, snapshot, completed)
            if backup.exists():
                raise DataValidationError(f"stale lifecycle backup requires recovery: {backup}")
            if output.exists():
                os.replace(output, backup)
            try:
                os.replace(staging, output)
            except Exception:
                if backup.exists() and not output.exists():
                    os.replace(backup, output)
                raise
            try:
                published = self.read(snapshot.summary.lifecycle_run_id)
            except Exception:
                if output.exists():
                    shutil.rmtree(output)
                if backup.exists():
                    os.replace(backup, output)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            return published or snapshot
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid lifecycle JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"lifecycle JSON must be an object: {path}")
    return payload


def _event_from_record(record: dict[str, Any]) -> LifecycleEvent:
    raw_details = record.pop("details_json", "{}")
    try:
        details = json.loads(str(raw_details))
    except json.JSONDecodeError as error:
        raise DataValidationError(f"invalid lifecycle event details: {error}") from error
    return LifecycleEvent.model_validate({**record, "details": details})


def _validate_staged_snapshot(
    directory: Path,
    snapshot: LifecycleSnapshot,
    manifest: LifecycleManifest,
) -> None:
    staged_summary = LifecycleSummary.model_validate(_json(directory / "lifecycle_summary.json"))
    staged_manifest = LifecycleManifest.model_validate(_json(directory / "manifest.json"))
    frame = pd.read_parquet(directory / "lifecycle_events.parquet")
    staged_events = tuple(
        _event_from_record(cast(dict[str, Any], record)) for record in frame.to_dict("records")
    )
    if (
        staged_summary != snapshot.summary
        or staged_events != snapshot.events
        or staged_manifest != manifest
    ):
        raise DataValidationError("staged lifecycle snapshot failed identity validation")
    expected = {
        "summary_sha256": file_sha256(directory / "lifecycle_summary.json"),
        "events_sha256": file_sha256(directory / "lifecycle_events.parquet"),
        "stage_results_sha256": file_sha256(directory / "stage_results.json"),
        "report_sha256": file_sha256(directory / "report.md"),
    }
    if any(getattr(manifest, key) != value for key, value in expected.items()):
        raise DataValidationError("staged lifecycle manifest hash mismatch")


def _require_successful_stage_evidence_preserved(
    previous: LifecycleSnapshot, current: LifecycleSnapshot
) -> None:
    """Fail closed if an immutable successful stage reference disappears or mutates."""

    for name, old_stage in previous.stage_results.items():
        if old_stage.status != "success":
            continue
        new_stage = current.stage_results.get(name)
        if new_stage is None:
            raise DataValidationError(f"successful lifecycle stage disappeared: {name}")
        missing_paths = sorted(set(old_stage.artifact_paths) - set(new_stage.artifact_paths))
        if missing_paths:
            raise DataValidationError(f"successful stage artifact references removed: {name}")
        for key, digest in old_stage.artifact_hashes.items():
            if new_stage.artifact_hashes.get(key) != digest:
                raise DataValidationError(f"successful stage artifact hash changed: {name}:{key}")
