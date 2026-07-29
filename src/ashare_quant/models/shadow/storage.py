"""Canonical hashing and immutable atomic shadow publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame


def canonical_payload_hash(payload: object) -> str:
    """Hash canonical UTF-8 JSON with lexical keys and normalized scalar values."""

    encoded = json.dumps(
        _canonical_value(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def logical_prediction_hash(frame: DataFrame) -> str:
    """Hash ordered logical rows, excluding the self-referential prediction_hash."""

    ordered = frame.drop(columns=["prediction_hash"], errors="ignore").sort_values(
        ["trade_date", "model_id", "ts_code"], kind="mergesort"
    )
    return canonical_payload_hash(ordered.to_dict("records"))


def file_sha256(path: Path) -> str:
    """Return a file's physical SHA256."""

    if not path.is_file():
        raise DataValidationError(f"required artifact does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_shadow_bundle(
    *,
    output_dir: Path,
    predictions: DataFrame,
    manifest_without_file_hash: dict[str, Any],
) -> dict[str, Any]:
    """Publish a complete directory atomically with the manifest written last."""

    expected_hash = str(manifest_without_file_hash.get("prediction_hash") or "")
    if not expected_hash or logical_prediction_hash(predictions) != expected_hash:
        raise DataValidationError("shadow manifest logical prediction hash is invalid")
    if set(predictions["prediction_hash"].astype(str)) != {expected_hash}:
        raise DataValidationError("shadow row prediction hashes differ from manifest")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        parquet_path = staging / "predictions.parquet"
        predictions.to_parquet(parquet_path, index=False)
        reread = pd.read_parquet(parquet_path)
        _validate_reread(predictions, reread)
        manifest = {
            **manifest_without_file_hash,
            "parquet_file_sha256": file_sha256(parquet_path),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise DataValidationError(
                f"shadow output already exists and cannot be overwritten: {output_dir}"
            )
        os.replace(staging, output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_complete_manifest(output_dir: Path) -> dict[str, Any] | None:
    """Read only a complete artifact; directories without a manifest are incomplete."""

    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.parquet"
    if not manifest_path.is_file() or not predictions_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid shadow manifest: {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError("shadow manifest must contain an object")
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_name") != "shadow_prediction_bundle"
    ):
        raise DataValidationError("shadow manifest identity or schema version is invalid")
    if payload.get("parquet_file_sha256") != file_sha256(predictions_path):
        raise DataValidationError("shadow Parquet hash differs from manifest")
    predictions = pd.read_parquet(predictions_path)
    expected_hash = str(payload.get("prediction_hash") or "")
    if (
        not expected_hash
        or logical_prediction_hash(predictions) != expected_hash
        or set(predictions["prediction_hash"].astype(str)) != {expected_hash}
    ):
        raise DataValidationError("shadow logical prediction hash differs from manifest")
    return payload


def _validate_reread(expected: DataFrame, actual: DataFrame) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.loc[:, expected.columns],
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise DataValidationError(f"staged shadow predictions failed reread: {error}") from error


def _canonical_value(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            raise DataValidationError("canonical prediction payload contains non-finite float")
        return format(number, ".17g")
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return str(value)
