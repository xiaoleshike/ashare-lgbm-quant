"""Immutable append-only storage for prospective observations."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import (
    canonical_payload_hash,
    file_sha256,
)
from ashare_quant.monitoring.performance_observation.schemas import (
    OBSERVATION_COLUMNS,
    OBSERVATION_KEY,
)
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame

LINEAGE_COLUMNS: tuple[str, ...] = (
    "model_origin",
    "parent_model_id",
    "training_request_id",
    "training_run_id",
    "validation_run_id",
)
LEGACY_OBSERVATION_COLUMNS: tuple[str, ...] = tuple(
    column for column in OBSERVATION_COLUMNS if column not in LINEAGE_COLUMNS
)


def logical_observation_hash(frame: DataFrame) -> str:
    """Hash canonical observation rows in unique-key order."""

    ordered = frame.loc[:, list(OBSERVATION_COLUMNS)].sort_values(
        list(OBSERVATION_KEY), kind="mergesort"
    )
    normalized = ordered.astype(object).where(ordered.notna(), None)
    return canonical_payload_hash(normalized.to_dict("records"))


def observation_content_hash(row: pd.Series) -> str:
    """Hash immutable row content while excluding first-observation timing metadata."""

    payload = {
        column: None if pd.isna(row[column]) else row[column]
        for column in OBSERVATION_COLUMNS
        if column not in {"observation_id", "observation_as_of"}
    }
    return canonical_payload_hash(payload)


def read_observation_artifact(output_dir: Path) -> tuple[DataFrame, dict[str, Any]] | None:
    """Read and verify one complete observation artifact."""

    parquet_path = output_dir / "observation.parquet"
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    if not parquet_path.is_file() or not manifest_path.is_file() or not metrics_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid performance observation manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise DataValidationError("performance observation manifest must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != "performance_observation"
    ):
        raise DataValidationError("invalid performance observation artifact identity")
    if file_sha256(parquet_path) != manifest.get("parquet_file_sha256"):
        raise DataValidationError("performance observation Parquet hash mismatch")
    raw = pd.read_parquet(parquet_path)
    legacy = not set(LINEAGE_COLUMNS).issubset(raw.columns)
    if legacy:
        missing = sorted(set(LEGACY_OBSERVATION_COLUMNS) - set(raw.columns))
        if missing:
            raise DataValidationError(f"performance observations lack columns: {missing}")
        source_hash = _logical_hash_columns(raw, LEGACY_OBSERVATION_COLUMNS)
    else:
        source_hash = logical_observation_hash(raw)
    if source_hash != manifest.get("observation_hash"):
        raise DataValidationError("performance observation logical hash mismatch")
    frame = normalize_observation_lineage(raw)
    _validate_frame(frame)
    return frame, manifest


def load_observation_history(
    root: Path,
    *,
    before_or_on: str,
    exclude_date: str | None = None,
) -> tuple[DataFrame, dict[str, str]]:
    """Load prior immutable batches and reject duplicate append-only identities."""

    frames: list[DataFrame] = []
    hashes: dict[str, str] = {}
    if not root.is_dir():
        return _empty_frame(), hashes
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        if directory.name > before_or_on or directory.name == exclude_date:
            continue
        artifact = read_observation_artifact(directory)
        if artifact is None:
            raise DataValidationError(f"incomplete performance observation artifact: {directory}")
        frame, manifest = artifact
        frames.append(frame)
        hashes[directory.name] = str(manifest["observation_hash"])
    if not frames:
        return _empty_frame(), hashes
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(list(OBSERVATION_KEY)).any():
        raise DataValidationError("append-only observation history contains duplicate identities")
    return combined.sort_values(list(OBSERVATION_KEY), kind="mergesort"), hashes


def publish_observation_artifact(
    *,
    output_dir: Path,
    observations: DataFrame,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Atomically publish Parquet and metrics, writing manifest last in staging."""

    _validate_frame(observations)
    expected_hash = logical_observation_hash(observations)
    if manifest.get("observation_hash") != expected_hash:
        raise DataValidationError("performance observation manifest hash is invalid")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        parquet_path = staging / "observation.parquet"
        observations.to_parquet(parquet_path, index=False)
        reread = pd.read_parquet(parquet_path)
        _validate_frame(reread)
        try:
            pd.testing.assert_frame_equal(
                reread.loc[:, list(OBSERVATION_COLUMNS)],
                observations.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as error:
            raise DataValidationError(
                f"staged performance observations failed reread: {error}"
            ) from error
        atomic_write_json(staging / "metrics.json", metrics)
        completed_manifest = {
            **manifest,
            "parquet_file_sha256": file_sha256(parquet_path),
            "metrics_file_sha256": file_sha256(staging / "metrics.json"),
        }
        atomic_write_json(staging / "manifest.json", completed_manifest)
        if output_dir.exists():
            raise DataValidationError(
                f"performance observation output already exists: {output_dir}"
            )
        os.replace(staging, output_dir)
        return completed_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_frame(frame: DataFrame) -> None:
    missing = sorted(set(OBSERVATION_COLUMNS) - set(frame.columns))
    if missing:
        raise DataValidationError(f"performance observations lack columns: {missing}")
    if frame.duplicated(list(OBSERVATION_KEY)).any():
        raise DataValidationError("performance observation identities are duplicated")
    if frame["observation_id"].duplicated().any():
        raise DataValidationError("performance observation_id is duplicated")


def _empty_frame() -> DataFrame:
    return pd.DataFrame(columns=list(OBSERVATION_COLUMNS))


def normalize_observation_lineage(frame: DataFrame) -> DataFrame:
    """Normalize legacy observations to the explicit model-origin contract."""

    result = frame.copy()
    defaults = (
        result["model_role"]
        .astype(str)
        .map(lambda role: "champion" if role == "champion" else "research_challenger")
    )
    if "model_origin" not in result:
        result["model_origin"] = defaults
    else:
        result["model_origin"] = result["model_origin"].where(
            result["model_origin"].notna() & result["model_origin"].astype(str).ne(""),
            defaults,
        )
    for column in LINEAGE_COLUMNS[1:]:
        if column not in result:
            result[column] = ""
        else:
            result[column] = result[column].fillna("").astype(str)
    return result


def _logical_hash_columns(frame: DataFrame, columns: tuple[str, ...]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(OBSERVATION_KEY), kind="mergesort")
    normalized = ordered.astype(object).where(ordered.notna(), None)
    return canonical_payload_hash(normalized.to_dict("records"))
