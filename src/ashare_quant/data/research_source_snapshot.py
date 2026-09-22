"""Immutable content-defined raw-source snapshots for governed research rebuilds."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
from ashare_quant.orchestration.lock import production_lock
from ashare_quant.utils.manifest import atomic_write_json

type JsonObject = dict[str, Any]

SNAPSHOT_SCHEMA_VERSION = 2
SNAPSHOT_CONTRACT_VERSION = "research_source_snapshot_v2_coherent_generation"
DERIVED_SNAPSHOT_CONTRACT_VERSION = "research_source_snapshot_v3_repair_derived"
SNAPSHOT_ARTIFACT_NAME = "research_source_snapshot"
SNAPSHOT_MECHANISM = "PHYSICAL_COPY_ATOMIC"
DERIVED_SNAPSHOT_MECHANISM = "IMMUTABLE_PARENT_HARDLINK_COPY_ON_WRITE"

# Exact union consumed by UniverseBuilder, FeatureBuilder, and LabelBuilder in the
# current governed rebuild path. Forecast/express are not consumed by those builders.
RESEARCH_REBUILD_DATASETS = (
    "adj_factor",
    "balancesheet",
    "cashflow",
    "daily",
    "daily_basic",
    "fina_indicator",
    "income",
    "index_daily",
    "namechange",
    "stk_limit",
    "stock_basic",
    "suspend_d",
    "trade_cal",
)
DATE_COLUMNS = ("trade_date", "cal_date", "ann_date", "end_date", "list_date")


@dataclass(frozen=True, slots=True)
class ResearchSourceSnapshotResult:
    """One immutable physical source snapshot."""

    snapshot_id: str
    output_dir: Path
    idempotent: bool


def materialize_repair_derived_research_source_snapshot(
    *,
    parent_snapshot: Path,
    repair_source_artifact: Path,
    snapshots_root: Path,
    lifecycle_evidence_hash: str,
) -> ResearchSourceSnapshotResult:
    """Apply only frozen provider-proven suspend rows to an immutable child snapshot."""

    from ashare_quant.data.security_lifecycle_source_probe import (
        validate_lifecycle_source_probe_artifact,
    )

    parent_manifest = validate_research_source_snapshot(parent_snapshot)
    validate_lifecycle_source_probe_artifact(repair_source_artifact)
    comparison = pd.read_parquet(repair_source_artifact / "comparison.parquet")
    selected = comparison[
        comparison["source_completeness_status"]
        .astype(str)
        .isin({"LOCAL_SUSPEND_D_INCOMPLETE", "LOCAL_BOTH_INCOMPLETE"})
    ]
    repair_rows = _missing_suspend_rows(selected)
    if repair_rows.empty:
        raise DataValidationError("RESEARCH_SOURCE_REPAIR_ROWS_MISSING")
    provider = pd.read_parquet(repair_source_artifact / "suspend_d_provider.parquet")
    _validate_repair_rows_against_provider(repair_rows, provider)

    parent_logical = cast(JsonObject, parent_manifest["logical_identity"])
    datasets = tuple(
        str(item["dataset"]) for item in cast(list[JsonObject], parent_logical.get("datasets", []))
    )
    snapshots_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=snapshots_root, prefix=".repair-derived.staging-"))
    try:
        frozen_root = staging / "datasets"
        _hardlink_snapshot_files(parent_snapshot / "datasets", frozen_root)
        changes = _apply_suspend_repairs(
            frozen_root=frozen_root,
            parent_root=parent_snapshot / "datasets",
            repair_rows=repair_rows,
        )
        inventory = _source_inventory(frozen_root, datasets)
        repair_identity: JsonObject = {
            "parent_snapshot_id": parent_snapshot.name,
            "parent_snapshot_manifest_hash": file_sha256(parent_snapshot / "manifest.json"),
            "repair_source_artifact_id": repair_source_artifact.name,
            "repair_source_manifest_hash": file_sha256(repair_source_artifact / "manifest.json"),
            "dataset": "suspend_d",
            "repair_rows_hash": canonical_payload_hash(_json_records(repair_rows)),
            "partition_changes": changes,
            "repair_reason": "VERIFIED_LOCAL_SUSPEND_D_INCOMPLETE",
        }
        generation: JsonObject = {
            "mode": "REPAIR_DERIVED_SNAPSHOT",
            "generation_hash": canonical_payload_hash(repair_identity),
        }
        logical: JsonObject = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "contract_version": DERIVED_SNAPSHOT_CONTRACT_VERSION,
            "snapshot_mechanism": DERIVED_SNAPSHOT_MECHANISM,
            "datasets": inventory["datasets"],
            "security_identity_mapping_hash": parent_logical["security_identity_mapping_hash"],
            "lifecycle_evidence_hash": lifecycle_evidence_hash,
            "source_generation": generation,
            "parent_snapshot_id": parent_snapshot.name,
            "parent_snapshot_manifest_hash": repair_identity["parent_snapshot_manifest_hash"],
            "repair_source_artifact_id": repair_source_artifact.name,
            "repair_source_manifest_hash": repair_identity["repair_source_manifest_hash"],
            "repair_identity_hash": canonical_payload_hash(repair_identity),
        }
        snapshot_id = f"research_source_snapshot_{canonical_payload_hash(logical)[:24]}"
        output = snapshots_root / snapshot_id
        if output.exists():
            shutil.rmtree(staging)
            validate_research_source_snapshot(output)
            return ResearchSourceSnapshotResult(snapshot_id, output, True)

        evidence_target = staging / "repair_evidence" / repair_source_artifact.name
        shutil.copytree(repair_source_artifact, evidence_target)
        atomic_write_json(staging / "contract.json", _derived_snapshot_contract())
        atomic_write_json(staging / "source_inventory.json", inventory)
        atomic_write_json(staging / "repair_manifest.json", repair_identity)
        inventory_hash = canonical_payload_hash(inventory)
        atomic_write_json(
            staging / "capture_proof.json",
            {
                "capture_contract": "immutable_parent_repair_v1",
                "source_generation": generation,
                "inventory_hash_before": canonical_payload_hash(parent_logical.get("datasets", [])),
                "inventory_hash_after": inventory_hash,
                "parent_snapshot_manifest_hash": repair_identity["parent_snapshot_manifest_hash"],
            },
        )
        artifact_hashes = {
            name: file_sha256(staging / name)
            for name in (
                "capture_proof.json",
                "contract.json",
                "repair_manifest.json",
                "source_inventory.json",
            )
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "artifact_name": SNAPSHOT_ARTIFACT_NAME,
                "snapshot_id": snapshot_id,
                "logical_identity": logical,
                "artifact_hashes": artifact_hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_research_source_snapshot(output)
    return ResearchSourceSnapshotResult(snapshot_id, output, False)


def snapshot_contract() -> JsonObject:
    """Return the versioned D.4 source-freeze contract."""

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract_version": SNAPSHOT_CONTRACT_VERSION,
        "snapshot_mechanism": SNAPSHOT_MECHANISM,
        "capture_consistency": "shared production writer lock + before/after inventory",
        "dataset_dependencies": list(RESEARCH_REBUILD_DATASETS),
        "dependency_owners": {
            "universe": [
                "stock_basic",
                "trade_cal",
                "daily",
                "daily_basic",
                "suspend_d",
                "stk_limit",
                "namechange",
            ],
            "features": [
                "daily",
                "adj_factor",
                "daily_basic",
                "index_daily",
                "trade_cal",
                "fina_indicator",
                "income",
                "balancesheet",
                "cashflow",
            ],
            "labels": ["trade_cal", "daily", "adj_factor", "stk_limit", "index_daily"],
        },
        "identity_fields": [
            "dataset",
            "relative_path",
            "content_sha256",
            "dataset_fingerprint",
            "date_coverage",
            "security_identity_mapping_hash",
            "lifecycle_evidence_hash",
        ],
        "excluded_identity_fields": ["mtime", "hostname", "pid", "absolute_path"],
        "publication": "staging + atomic rename + manifest last",
    }


def materialize_research_source_snapshot(
    *,
    source_root: Path,
    snapshots_root: Path,
    security_identity_mapping_hash: str,
    lifecycle_evidence_hash: str,
    writer_lock_path: Path,
    source_generation_path: Path | None = None,
    datasets: tuple[str, ...] = RESEARCH_REBUILD_DATASETS,
) -> ResearchSourceSnapshotResult:
    """Freeze one coherent source generation under the production writer lock."""

    with production_lock(writer_lock_path, command="research source snapshot capture"):
        return _materialize_locked_snapshot(
            source_root=source_root,
            snapshots_root=snapshots_root,
            security_identity_mapping_hash=security_identity_mapping_hash,
            lifecycle_evidence_hash=lifecycle_evidence_hash,
            source_generation_path=source_generation_path,
            datasets=datasets,
        )


def _materialize_locked_snapshot(
    *,
    source_root: Path,
    snapshots_root: Path,
    security_identity_mapping_hash: str,
    lifecycle_evidence_hash: str,
    source_generation_path: Path | None,
    datasets: tuple[str, ...],
) -> ResearchSourceSnapshotResult:
    """Copy and publish while the caller holds the shared production writer lock."""

    selected = tuple(sorted(set(datasets)))
    if not selected or any(not item or Path(item).name != item for item in selected):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_DATASET_INVALID")
    inventory = _source_inventory(Path(source_root), selected)
    generation = _source_generation(source_generation_path, inventory)
    logical = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract_version": SNAPSHOT_CONTRACT_VERSION,
        "snapshot_mechanism": SNAPSHOT_MECHANISM,
        "datasets": inventory["datasets"],
        "security_identity_mapping_hash": security_identity_mapping_hash,
        "lifecycle_evidence_hash": lifecycle_evidence_hash,
        "source_generation": generation,
    }
    snapshot_id = f"research_source_snapshot_{canonical_payload_hash(logical)[:24]}"
    output = Path(snapshots_root) / snapshot_id
    if output.exists():
        validate_research_source_snapshot(output)
        return ResearchSourceSnapshotResult(snapshot_id, output, True)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{snapshot_id}.staging-"))
    try:
        frozen_root = staging / "datasets"
        for dataset in cast(list[JsonObject], inventory["datasets"]):
            for item in cast(list[JsonObject], dataset["files"]):
                relative = Path(str(item["relative_path"]))
                source = Path(source_root) / relative
                target = frozen_root / relative
                before = file_sha256(source)
                if before != item["content_sha256"]:
                    raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_SOURCE_CHANGED")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                if file_sha256(source) != before or file_sha256(target) != before:
                    raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_COPY_MISMATCH")
        inventory_after = _source_inventory(Path(source_root), selected)
        generation_after = _source_generation(source_generation_path, inventory_after)
        if inventory_after != inventory or generation_after != generation:
            raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_MIXED_GENERATION")
        atomic_write_json(staging / "contract.json", snapshot_contract())
        atomic_write_json(staging / "source_inventory.json", inventory)
        atomic_write_json(
            staging / "capture_proof.json",
            {
                "capture_contract": "shared_production_writer_lock_v1",
                "source_generation": generation,
                "inventory_hash_before": canonical_payload_hash(inventory),
                "inventory_hash_after": canonical_payload_hash(inventory_after),
            },
        )
        artifact_hashes = {
            "contract.json": file_sha256(staging / "contract.json"),
            "source_inventory.json": file_sha256(staging / "source_inventory.json"),
            "capture_proof.json": file_sha256(staging / "capture_proof.json"),
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "artifact_name": SNAPSHOT_ARTIFACT_NAME,
                "snapshot_id": snapshot_id,
                "logical_identity": logical,
                "artifact_hashes": artifact_hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_research_source_snapshot(output)
    return ResearchSourceSnapshotResult(snapshot_id, output, False)


def validate_research_source_snapshot(path: Path) -> JsonObject:
    """Validate manifests and every frozen source byte; references alone fail closed."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        inventory = json.loads((path / "source_inventory.json").read_text(encoding="utf-8"))
        capture = json.loads((path / "capture_proof.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or manifest.get("artifact_name") != SNAPSHOT_ARTIFACT_NAME
        or manifest.get("snapshot_id") != path.name
        or not isinstance(inventory, dict)
        or not isinstance(capture, dict)
    ):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_INVALID")
    hashes = manifest.get("artifact_hashes")
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    contract_version = logical.get("contract_version")
    expected_artifacts = {"contract.json", "source_inventory.json", "capture_proof.json"}
    if contract_version == DERIVED_SNAPSHOT_CONTRACT_VERSION:
        expected_artifacts.add("repair_manifest.json")
    if not isinstance(hashes, dict) or set(hashes) != expected_artifacts:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_ARTIFACT_SET_INVALID")
    for relative, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / relative) != expected:
            raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_ARTIFACT_HASH_MISMATCH")
    for dataset in cast(list[JsonObject], inventory.get("datasets", [])):
        files = cast(list[JsonObject], dataset.get("files", []))
        if not files:
            raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_DATASET_EMPTY")
        for item in files:
            relative = Path(str(item["relative_path"]))
            frozen = (path / "datasets" / relative).resolve()
            if (path / "datasets").resolve() not in frozen.parents or not frozen.is_file():
                raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_FROZEN_BYTES_MISSING")
            if file_sha256(frozen) != item["content_sha256"]:
                raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_FROZEN_HASH_MISMATCH")
    expected_files = {
        str(item["relative_path"])
        for dataset in cast(list[JsonObject], inventory.get("datasets", []))
        for item in cast(list[JsonObject], dataset.get("files", []))
    }
    actual_files = {
        item.relative_to(path / "datasets").as_posix()
        for item in (path / "datasets").glob("**/*")
        if item.is_file()
    }
    if actual_files != expected_files:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_FROZEN_SET_MISMATCH")
    if logical.get("datasets") != inventory.get("datasets"):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_IDENTITY_MISMATCH")
    inventory_hash = canonical_payload_hash(inventory)
    if contract_version == SNAPSHOT_CONTRACT_VERSION:
        if (
            capture.get("capture_contract") != "shared_production_writer_lock_v1"
            or capture.get("source_generation") != logical.get("source_generation")
            or capture.get("inventory_hash_before") != inventory_hash
            or capture.get("inventory_hash_after") != inventory_hash
        ):
            raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_CAPTURE_PROOF_INVALID")
    elif contract_version == DERIVED_SNAPSHOT_CONTRACT_VERSION:
        _validate_derived_snapshot(path, manifest, capture, inventory_hash)
    else:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_CAPTURE_PROOF_INVALID")
    datasets = cast(list[JsonObject], inventory.get("datasets", []))
    names = [str(item.get("dataset", "")) for item in datasets]
    if not names or len(names) != len(set(names)) or names != sorted(names):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_DATASET_SET_INVALID")
    expected_id = f"research_source_snapshot_{canonical_payload_hash(logical)[:24]}"
    if expected_id != path.name:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_IDENTITY_MISMATCH")
    return cast(JsonObject, manifest)


def _derived_snapshot_contract() -> JsonObject:
    contract = snapshot_contract()
    contract.update(
        {
            "contract_version": DERIVED_SNAPSHOT_CONTRACT_VERSION,
            "snapshot_mechanism": DERIVED_SNAPSHOT_MECHANISM,
            "capture_consistency": "validated immutable parent + copy-on-write repair",
            "publication": "staging + atomic rename + manifest last",
        }
    )
    return contract


def _validate_derived_snapshot(
    path: Path,
    manifest: JsonObject,
    capture: JsonObject,
    inventory_hash: str,
) -> None:
    from ashare_quant.data.security_lifecycle_source_probe import (
        validate_lifecycle_source_probe_artifact,
    )

    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    repair = _read_json_file(path / "repair_manifest.json")
    parent_id = str(logical.get("parent_snapshot_id", ""))
    parent = path.parent / parent_id
    parent_manifest = validate_research_source_snapshot(parent)
    parent_logical = cast(JsonObject, parent_manifest.get("logical_identity", {}))
    if (
        capture.get("capture_contract") != "immutable_parent_repair_v1"
        or capture.get("source_generation") != logical.get("source_generation")
        or capture.get("inventory_hash_after") != inventory_hash
        or file_sha256(parent / "manifest.json") != logical.get("parent_snapshot_manifest_hash")
        or parent_manifest.get("snapshot_id") != parent_id
        or capture.get("inventory_hash_before")
        != canonical_payload_hash(parent_logical.get("datasets", []))
        or canonical_payload_hash(repair) != logical.get("repair_identity_hash")
        or repair.get("parent_snapshot_id") != parent_id
        or repair.get("parent_snapshot_manifest_hash")
        != logical.get("parent_snapshot_manifest_hash")
        or repair.get("repair_source_artifact_id") != logical.get("repair_source_artifact_id")
        or repair.get("repair_source_manifest_hash") != logical.get("repair_source_manifest_hash")
        or repair.get("dataset") != "suspend_d"
    ):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_CAPTURE_PROOF_INVALID")
    evidence_id = str(logical.get("repair_source_artifact_id", ""))
    evidence = path / "repair_evidence" / evidence_id
    validate_lifecycle_source_probe_artifact(evidence)
    if file_sha256(evidence / "manifest.json") != logical.get("repair_source_manifest_hash"):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_REPAIR_EVIDENCE_INVALID")
    comparison = pd.read_parquet(evidence / "comparison.parquet")
    selected = comparison[
        comparison["source_completeness_status"]
        .astype(str)
        .isin({"LOCAL_SUSPEND_D_INCOMPLETE", "LOCAL_BOTH_INCOMPLETE"})
    ]
    rows = _missing_suspend_rows(selected)
    provider = pd.read_parquet(evidence / "suspend_d_provider.parquet")
    _validate_repair_rows_against_provider(rows, provider)
    changes = repair.get("partition_changes")
    if (
        repair.get("repair_rows_hash") != canonical_payload_hash(_json_records(rows))
        or not isinstance(changes, list)
        or sum(int(cast(JsonObject, item).get("rows_added", 0)) for item in changes) != len(rows)
    ):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_REPAIR_EVIDENCE_INVALID")
    _validate_partition_changes(path, parent, cast(list[JsonObject], changes))


def _validate_partition_changes(child: Path, parent: Path, changes: list[JsonObject]) -> None:
    seen: set[str] = set()
    for item in changes:
        relative = str(item.get("relative_path", ""))
        relative_path = Path(relative)
        if (
            not relative.startswith("suspend_d/")
            or relative in seen
            or int(item.get("rows_added", 0)) <= 0
            or file_sha256(parent / "datasets" / relative_path) != item.get("before_partition_hash")
            or file_sha256(child / "datasets" / relative_path) != item.get("after_partition_hash")
        ):
            raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_REPAIR_PARTITION_INVALID")
        seen.add(relative)


def _read_json_file(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_REPAIR_MANIFEST_INVALID") from error
    if not isinstance(value, dict):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_REPAIR_MANIFEST_INVALID")
    return cast(JsonObject, value)


def _missing_suspend_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[JsonObject] = []
    for value in comparison["missing_suspend_rows"].fillna("[]").astype(str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise DataValidationError("RESEARCH_SOURCE_REPAIR_ROWS_INVALID") from error
        if not isinstance(parsed, list):
            raise DataValidationError("RESEARCH_SOURCE_REPAIR_ROWS_INVALID")
        rows.extend(cast(list[JsonObject], parsed))
    required = [
        "canonical_ts_code",
        "source_ts_code",
        "trade_date",
        "suspend_timing",
        "suspend_type",
    ]
    frame = pd.DataFrame(rows, columns=required)
    if frame.empty or frame[["source_ts_code", "trade_date", "suspend_type"]].isna().any(axis=None):
        raise DataValidationError("RESEARCH_SOURCE_REPAIR_ROWS_INVALID")
    return frame.sort_values(required, na_position="first").drop_duplicates().reset_index(drop=True)


def _validate_repair_rows_against_provider(repair: pd.DataFrame, provider: pd.DataFrame) -> None:
    normalized = provider.copy()
    keys = ["source_ts_code", "canonical_ts_code", "trade_date", "suspend_type"]
    for frame in (repair, normalized):
        for column in keys:
            frame[column] = frame[column].astype(str)
        frame["suspend_timing"] = frame["suspend_timing"].fillna("").astype(str)
    expected = set(map(tuple, repair[[*keys, "suspend_timing"]].to_numpy()))
    available = set(map(tuple, normalized[[*keys, "suspend_timing"]].to_numpy()))
    if not expected.issubset(available):
        raise DataValidationError("RESEARCH_SOURCE_REPAIR_PROVIDER_MISMATCH")


def _hardlink_snapshot_files(source: Path, target: Path) -> None:
    for item in source.glob("**/*"):
        if not item.is_file():
            continue
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(item, destination)


def _apply_suspend_repairs(
    *, frozen_root: Path, parent_root: Path, repair_rows: pd.DataFrame
) -> list[JsonObject]:
    changes: list[JsonObject] = []
    working = repair_rows.copy()
    working["year"] = working["trade_date"].astype(str).str[:4]
    working["month"] = working["trade_date"].astype(str).str[4:6]
    for (year, month), group in working.groupby(["year", "month"], sort=True):
        relative = Path("suspend_d") / f"year={year}" / f"month={month}" / "data.parquet"
        parent_file = parent_root / relative
        child_file = frozen_root / relative
        if not parent_file.is_file() or not child_file.is_file():
            raise DataValidationError("RESEARCH_SOURCE_REPAIR_PARTITION_MISSING")
        before_hash = file_sha256(parent_file)
        existing = pd.read_parquet(parent_file)
        additions = pd.DataFrame(
            {
                "ts_code": group["source_ts_code"].astype(str),
                "trade_date": group["trade_date"].astype(str),
                "suspend_timing": group["suspend_timing"],
                "suspend_type": group["suspend_type"].astype(str),
                "month": str(month),
                "year": int(str(year)),
            }
        )
        columns = list(existing.columns)
        additions = additions.reindex(columns=columns)
        combined = pd.concat([existing, additions], ignore_index=True)
        normalized_timing = combined["suspend_timing"].fillna("").astype(str)
        combined = combined.assign(_timing=normalized_timing)
        combined = combined.drop_duplicates(
            ["ts_code", "trade_date", "suspend_type", "_timing"], keep="first"
        ).drop(columns="_timing")
        combined = combined.sort_values(
            ["trade_date", "ts_code", "suspend_type", "suspend_timing"],
            na_position="first",
        )
        temporary = child_file.with_suffix(".repair.parquet")
        combined.to_parquet(temporary, index=False)
        os.replace(temporary, child_file)
        added = len(combined) - len(existing)
        if added != len(additions):
            raise DataValidationError("RESEARCH_SOURCE_REPAIR_ROW_COUNT_MISMATCH")
        changes.append(
            {
                "relative_path": relative.as_posix(),
                "before_partition_hash": before_hash,
                "after_partition_hash": file_sha256(child_file),
                "rows_added": int(added),
                "repair_row_content_hash": canonical_payload_hash(
                    _json_records(group.drop(columns=["year", "month"]))
                ),
            }
        )
    return changes


def _json_records(frame: pd.DataFrame) -> list[JsonObject]:
    """Normalize pandas scalars/nulls before content-addressing repair rows."""

    normalized = frame.astype(object).where(pd.notna(frame), None)
    return cast(list[JsonObject], normalized.to_dict(orient="records"))


def research_rebuild_datasets(*, include_fundamentals: bool) -> tuple[str, ...]:
    """Derive mandatory raw dependencies from the enabled rebuild contract."""

    base = {
        "adj_factor",
        "daily",
        "daily_basic",
        "index_daily",
        "namechange",
        "stk_limit",
        "stock_basic",
        "suspend_d",
        "trade_cal",
    }
    if include_fundamentals:
        base.update({"balancesheet", "cashflow", "fina_indicator", "income"})
    return tuple(sorted(base))


def _source_generation(path: Path | None, inventory: JsonObject) -> JsonObject:
    if path is None:
        return {
            "mode": "LOCKED_CONTENT_INVENTORY",
            "generation_hash": canonical_payload_hash(inventory),
        }
    if not path.is_file():
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_GENERATION_MISSING")
    return {
        "mode": "COMMITTED_GENERATION_FILE",
        "generation_hash": file_sha256(path),
    }


def _source_inventory(source_root: Path, datasets: tuple[str, ...]) -> JsonObject:
    rows: list[JsonObject] = []
    for dataset in datasets:
        dataset_root = source_root / dataset
        files = sorted(dataset_root.glob("**/*.parquet"))
        if not files:
            raise DataValidationError(f"RESEARCH_SOURCE_SNAPSHOT_DATASET_MISSING: {dataset}")
        file_rows = [
            {
                "relative_path": file.relative_to(source_root).as_posix(),
                "content_sha256": file_sha256(file),
                "size_bytes": file.stat().st_size,
            }
            for file in files
        ]
        coverage = _date_coverage(files)
        rows.append(
            {
                "dataset": dataset,
                "files": file_rows,
                "partition_count": len(file_rows),
                "size_bytes": sum(file.stat().st_size for file in files),
                "date_coverage": coverage,
                "dataset_fingerprint": canonical_payload_hash(
                    {"dataset": dataset, "files": file_rows, "date_coverage": coverage}
                ),
            }
        )
    return {"datasets": rows}


def _date_coverage(files: list[Path]) -> JsonObject:
    minima: list[str] = []
    maxima: list[str] = []
    columns_seen: set[str] = set()
    for file in files:
        columns = set(pq.read_schema(file).names)  # type: ignore[no-untyped-call]
        chosen = next((column for column in DATE_COLUMNS if column in columns), None)
        if chosen is None:
            continue
        values = pd.read_parquet(file, columns=[chosen])[chosen].dropna().astype(str)
        if values.empty:
            continue
        columns_seen.add(chosen)
        minima.append(str(values.min()))
        maxima.append(str(values.max()))
    return {
        "columns": sorted(columns_seen),
        "minimum": min(minima) if minima else None,
        "maximum": max(maxima) if maxima else None,
    }
