"""Label-free selection and exact ownership of frozen STRICT_OOS predictions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.walk_forward_evaluation import validate_completed_walk_forward_artifact

SELECTION_CONTRACT = "immutable_fold_manifest_strict_oos_v1"
PREDICTION_COLUMNS = ["trade_date", "ts_code", "prediction_score"]


def file_hash(path: Path) -> str:
    """Hash immutable bytes without loading large files into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def payload_hash(value: object) -> str:
    """Content identity without operational locators or non-finite JSON numbers."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read an object, rejecting malformed JSON roots."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataValidationError(f"CONTINUOUS_INVALID_OBJECT: {path}")
    return value


def frame_hash(frame: pd.DataFrame) -> str:
    """Hash canonical Arrow rows, independent of Parquet compression and file paths."""
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    table = table.replace_schema_metadata(None).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(memoryview(sink.getvalue())).hexdigest()


@dataclass(frozen=True)
class ContinuousSource:
    """One validated source selection; no labels or ranking metrics are consumed."""

    manifest: dict[str, Any]
    predictions: pd.DataFrame
    lineage: pd.DataFrame
    identity: dict[str, Any]


def load_continuous_source(run: Path) -> ContinuousSource:
    """Validate the existing COMPLETE run and select by immutable classification only."""
    manifest = validate_completed_walk_forward_artifact(run)
    if (
        manifest.get("evaluation_contract_version") != 9
        or manifest.get("run_scope") != {"mode": "all_eligible_folds"}
        or manifest["experiment"].get("horizon") != 5
    ):
        raise DataValidationError("CONTINUOUS_SOURCE_CONTRACT_UNSUPPORTED")
    selected = []
    for fold_id, digest in manifest["fold_manifest_hashes"].items():
        folder = run / "folds" / fold_id
        fm = read_json(folder / "manifest.json")
        if fm["research_validity"].get("research_classification") != "STRICT_OOS":
            continue
        if fm.get("horizon") != manifest["experiment"]["horizon"]:
            raise DataValidationError("CONTINUOUS_SOURCE_HORIZON_MISMATCH")
        for key in ("modeling_identity", "execution_identity", "execution_contract"):
            if fm.get(key) != manifest.get(key):
                raise DataValidationError("CONTINUOUS_SOURCE_IDENTITY_MISMATCH")
        if fm["fold"].get("research_validity") != fm["research_validity"]:
            raise DataValidationError("CONTINUOUS_CLASSIFICATION_MISMATCH")
        selected.append((fm["fold"]["evaluation_start"], fold_id, digest, fm))
    if not selected:
        raise DataValidationError("CONTINUOUS_NO_STRICT_OOS_FOLDS")
    selected.sort()
    frames: list[pd.DataFrame] = []
    lineage: list[dict[str, Any]] = []
    inventory = []
    previous_end = ""
    for start, fold_id, digest, fm in selected:
        end = str(fm["fold"]["evaluation_end"])
        for date in (start, end):
            if datetime.strptime(date, "%Y%m%d").strftime("%Y%m%d") != date:
                raise DataValidationError("CONTINUOUS_INVALID_DATE")
        if start > end or start <= previous_end:
            raise DataValidationError("CONTINUOUS_OVERLAPPING_FOLD_OWNERSHIP")
        previous_end = end
        hashes = fm["artifact_hashes"]
        frame = pd.read_parquet(run / "folds" / fold_id / "predictions.parquet")
        if list(frame.columns) != PREDICTION_COLUMNS or frame.empty:
            raise DataValidationError("CONTINUOUS_PREDICTION_SCHEMA_INVALID")
        if frame[PREDICTION_COLUMNS].isna().any().any():
            raise DataValidationError("CONTINUOUS_NULL_PREDICTION")
        if not np.isfinite(frame.prediction_score.to_numpy(dtype=float)).all():
            raise DataValidationError("CONTINUOUS_NONFINITE_SCORE")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise DataValidationError("CONTINUOUS_DUPLICATE_PREDICTION_KEY")
        dates = sorted(frame.trade_date.unique().tolist())
        if any(not isinstance(d, str) or d < start or d > end for d in dates):
            raise DataValidationError("CONTINUOUS_PREDICTION_OUTSIDE_FOLD")
        record = {
            "source_fold_id": fold_id,
            "evaluation_start": start,
            "evaluation_end": end,
            "research_classification": "STRICT_OOS",
            "source_fold_manifest_hash": digest,
            "source_predictions_hash": hashes["predictions.parquet"],
            "source_model_hash": hashes["model.txt"],
        }
        inventory.append({**record, "prediction_rows": len(frame)})
        lineage.extend({"trade_date": date, **record} for date in dates)
        frames.append(
            frame.assign(
                source_fold_id=fold_id,
                source_prediction_hash=hashes["predictions.parquet"],
                source_model_hash=hashes["model.txt"],
            )
        )
    predictions = pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "ts_code"], kind="stable", ignore_index=True
    )
    ownership = pd.DataFrame(lineage).sort_values("trade_date", ignore_index=True)
    if ownership.trade_date.duplicated().any():
        raise DataValidationError("CONTINUOUS_OVERLAPPING_DATE_OWNERSHIP")
    identity = {
        "source_run_id": manifest["run_id"],
        "source_manifest_hash": file_hash(run / "manifest.json"),
        "selection_contract": SELECTION_CONTRACT,
        "selected_folds": inventory,
        "stitched_content_hash": frame_hash(predictions),
        "signal_lineage_hash": frame_hash(ownership),
        "prediction_rows": len(predictions),
        "signal_sessions": len(ownership),
        "signal_start": str(ownership.trade_date.min()),
        "signal_end": str(ownership.trade_date.max()),
    }
    return ContinuousSource(manifest, predictions, ownership, identity)
