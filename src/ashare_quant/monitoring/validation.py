"""Read-only validation for immutable monitoring inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.schemas import MonitoringSources
from ashare_quant.utils.manifest import config_hash

type DataFrame = pd.DataFrame


def validate_monitoring_sources(
    *,
    as_of: str,
    reports_root: Path,
    config_path: Path,
) -> tuple[MonitoringSources, DataFrame, DataFrame]:
    """Load and bind production artifacts without touching labels or model code."""

    _validate_date(as_of)
    report_dir = reports_root / as_of
    summary_path = report_dir / "production_summary.json"
    predictions_path = report_dir / "predictions.parquet"
    candidates_path = report_dir / "candidates.csv"
    prediction_manifest_path = report_dir / "manifest.json"
    candidate_manifest_path = report_dir / "candidates_manifest.json"
    summary = _load_json(summary_path, "production summary")
    prediction_manifest = _load_json(prediction_manifest_path, "prediction manifest")
    candidate_manifest = _load_json(candidate_manifest_path, "candidate manifest")

    expected_config_hash = config_hash(config_path)
    _require_identity(summary, "production_daily_summary", as_of)
    _require_identity(prediction_manifest, "production_predictions", as_of)
    _require_identity(candidate_manifest, "production_candidates", as_of)
    for name, manifest in (
        ("prediction", prediction_manifest),
        ("candidate", candidate_manifest),
    ):
        if manifest.get("config_hash") != expected_config_hash:
            raise DataValidationError(
                f"{name} manifest config hash mismatch: "
                f"expected={expected_config_hash} actual={manifest.get('config_hash')}"
            )

    model_id = _required_string(summary, "model_id", "production summary")
    feature_hash = _required_string(prediction_manifest, "feature_hash", "prediction manifest")
    if prediction_manifest.get("model_id") != model_id:
        raise DataValidationError("production summary and prediction manifest model_id mismatch")
    if candidate_manifest.get("model_id") != model_id:
        raise DataValidationError("candidate manifest model_id mismatch")
    if candidate_manifest.get("feature_hash") != feature_hash:
        raise DataValidationError("candidate and prediction feature hash mismatch")
    embedded = candidate_manifest.get("prediction_manifest")
    if not isinstance(embedded, dict):
        raise DataValidationError("candidate manifest lacks embedded prediction manifest")
    if _payload_hash(embedded) != _payload_hash(prediction_manifest):
        raise DataValidationError("candidate source prediction manifest hash mismatch")

    predictions = _read_parquet(predictions_path, "predictions")
    candidates = _read_csv(candidates_path, "candidates")
    _validate_predictions(predictions, as_of, model_id)
    _validate_candidates(candidates, as_of, model_id)
    if int(prediction_manifest.get("prediction_count", -1)) != len(predictions):
        raise DataValidationError("prediction manifest row count mismatch")
    if int(candidate_manifest.get("candidate_count", -1)) != len(candidates):
        raise DataValidationError("candidate manifest row count mismatch")
    if int(summary.get("candidate_count", -1)) != len(candidates):
        raise DataValidationError("production summary candidate count mismatch")
    _validate_summary_artifacts(summary)

    source_hashes = {
        "production_summary": _file_hash(summary_path),
        "predictions": _file_hash(predictions_path),
        "prediction_manifest": _file_hash(prediction_manifest_path),
        "candidates": _file_hash(candidates_path),
        "candidate_manifest": _file_hash(candidate_manifest_path),
    }
    drift_reference = find_existing_drift_reference(
        reports_root=reports_root,
        model_id=model_id,
        as_of=as_of,
    )
    return (
        MonitoringSources(
            as_of=as_of,
            model_id=model_id,
            feature_hash=feature_hash,
            production_summary=summary,
            prediction_manifest=prediction_manifest,
            candidate_manifest=candidate_manifest,
            predictions_path=predictions_path,
            candidates_path=candidates_path,
            source_hashes=source_hashes,
            drift_reference=drift_reference,
        ),
        predictions,
        candidates,
    )


def find_existing_drift_reference(
    *,
    reports_root: Path,
    model_id: str,
    as_of: str,
) -> dict[str, Any] | None:
    """Reference the latest eligible drift report without recalculating PSI or KS."""

    root = reports_root / "model_diagnostics"
    matches: list[tuple[str, Path, dict[str, Any]]] = []
    if not root.is_dir():
        return None
    for path in root.glob("*/manifest.json"):
        try:
            manifest = _load_json(path, "drift manifest")
        except DataValidationError:
            continue
        end_date = str(manifest.get("requested_end_date") or "")
        if manifest.get("model_id") == model_id and end_date and end_date <= as_of:
            matches.append((end_date, path, manifest))
    if not matches:
        return None
    end_date, path, manifest = max(matches, key=lambda item: (item[0], str(item[1])))
    return {
        "manifest_path": str(path),
        "manifest_hash": _file_hash(path),
        "requested_end_date": end_date,
        "summary_path": str(path.parent / "summary.json"),
        "feature_hash": manifest.get("feature_hash"),
        "implementation": "reused_existing_model_drift_diagnostics",
    }


def _validate_predictions(frame: DataFrame, as_of: str, model_id: str) -> None:
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"predictions lack required columns: {missing}")
    if frame.empty:
        raise DataValidationError("predictions are empty")
    if set(frame["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("predictions contain a different trade_date")
    if set(frame["model_id"].astype(str)) != {model_id}:
        raise DataValidationError("predictions contain a different model_id")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("predictions contain duplicate stock keys")


def _validate_candidates(frame: DataFrame, as_of: str, model_id: str) -> None:
    required = {"trade_date", "ts_code", "prediction_score", "model_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"candidates lack required columns: {missing}")
    if set(frame["trade_date"].astype(str)) not in ({as_of}, set()):
        raise DataValidationError("candidates contain a different trade_date")
    if not frame.empty and set(frame["model_id"].astype(str)) != {model_id}:
        raise DataValidationError("candidates contain a different model_id")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise DataValidationError("candidates contain duplicate stock keys")


def _validate_summary_artifacts(summary: dict[str, Any]) -> None:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise DataValidationError("production summary has no artifact references")
    missing = [
        str(path) for path in artifacts if not isinstance(path, str) or not Path(path).exists()
    ]
    if missing:
        raise DataValidationError(f"production summary references missing artifacts: {missing}")


def _require_identity(payload: dict[str, Any], name: str, as_of: str) -> None:
    if payload.get("artifact_name") != name:
        raise DataValidationError(f"unexpected artifact identity: expected={name}")
    if str(payload.get("as_of")) != as_of:
        raise DataValidationError(f"{name} as_of mismatch")


def _required_string(payload: dict[str, Any], key: str, description: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{description} lacks {key}")
    return value


def _validate_date(value: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"monitoring as_of must use YYYYMMDD: {value}")


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"{description} must contain a JSON object")
    return value


def _read_parquet(path: Path, description: str) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error


def _read_csv(path: Path, description: str) -> DataFrame:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        return pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
