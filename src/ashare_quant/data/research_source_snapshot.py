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
from ashare_quant.utils.manifest import atomic_write_json

type JsonObject = dict[str, Any]

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_CONTRACT_VERSION = "research_source_snapshot_v1_physical_copy"
SNAPSHOT_ARTIFACT_NAME = "research_source_snapshot"
SNAPSHOT_MECHANISM = "PHYSICAL_COPY_ATOMIC"

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


def snapshot_contract() -> JsonObject:
    """Return the versioned D.4 source-freeze contract."""

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract_version": SNAPSHOT_CONTRACT_VERSION,
        "snapshot_mechanism": SNAPSHOT_MECHANISM,
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
    datasets: tuple[str, ...] = RESEARCH_REBUILD_DATASETS,
) -> ResearchSourceSnapshotResult:
    """Physically freeze selected datasets without relying on mutable source paths."""

    selected = tuple(sorted(set(datasets)))
    if not selected or any(not item or Path(item).name != item for item in selected):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_DATASET_INVALID")
    inventory = _source_inventory(Path(source_root), selected)
    logical = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "contract_version": SNAPSHOT_CONTRACT_VERSION,
        "snapshot_mechanism": SNAPSHOT_MECHANISM,
        "datasets": inventory["datasets"],
        "security_identity_mapping_hash": security_identity_mapping_hash,
        "lifecycle_evidence_hash": lifecycle_evidence_hash,
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
        atomic_write_json(staging / "contract.json", snapshot_contract())
        atomic_write_json(staging / "source_inventory.json", inventory)
        artifact_hashes = {
            "contract.json": file_sha256(staging / "contract.json"),
            "source_inventory.json": file_sha256(staging / "source_inventory.json"),
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
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or manifest.get("artifact_name") != SNAPSHOT_ARTIFACT_NAME
        or manifest.get("snapshot_id") != path.name
        or not isinstance(inventory, dict)
    ):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_INVALID")
    hashes = manifest.get("artifact_hashes")
    expected_artifacts = {"contract.json", "source_inventory.json"}
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
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    if logical.get("datasets") != inventory.get("datasets"):
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_IDENTITY_MISMATCH")
    expected_id = f"research_source_snapshot_{canonical_payload_hash(logical)[:24]}"
    if expected_id != path.name:
        raise DataValidationError("RESEARCH_SOURCE_SNAPSHOT_IDENTITY_MISMATCH")
    return cast(JsonObject, manifest)


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
