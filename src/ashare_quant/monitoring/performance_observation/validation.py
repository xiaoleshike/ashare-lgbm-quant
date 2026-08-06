"""Read-only source validation for prospective performance observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.schemas import MODEL_ORIGINS, MODEL_ROLES
from ashare_quant.models.shadow.storage import file_sha256, read_complete_manifest
from ashare_quant.monitoring.performance_observation.schemas import SUPPORTED_HORIZONS
from ashare_quant.orchestration.publication import validate_production_publication

type DataFrame = pd.DataFrame

PROHIBITED_SOURCES: tuple[str, ...] = (
    "reports/challenger_predictions",
    "reports/ensemble_evaluation",
)


def load_shadow_sources(
    *,
    reports_root: Path,
    runs_root: Path,
    observation_as_of: str,
) -> tuple[DataFrame, list[dict[str, Any]], dict[str, str]]:
    """Load only complete prospective shadow bundles through the cutoff."""

    root = reports_root / "shadow_predictions"
    frames: list[DataFrame] = []
    manifests: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    if not root.is_dir():
        return pd.DataFrame(), manifests, source_hashes
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        signal_date = directory.name
        if signal_date > observation_as_of:
            continue
        normalized = directory.as_posix()
        if any(value in normalized for value in PROHIBITED_SOURCES):
            raise DataValidationError(f"historical evaluation source is prohibited: {directory}")
        manifest = read_complete_manifest(directory)
        if manifest is None:
            raise DataValidationError(f"incomplete shadow artifact cannot be observed: {directory}")
        summary = validate_production_publication(
            reports_root=reports_root,
            runs_root=runs_root,
            as_of=signal_date,
        )
        if str(summary.get("run_id")) != str(manifest.get("production_run_id")):
            raise DataValidationError(
                f"shadow production_run_id differs from successful production run: {signal_date}"
            )
        sources = [("production", directory, manifest)]
        retrained_root = directory / "retrained"
        if retrained_root.is_dir():
            for sidecar in sorted(path for path in retrained_root.iterdir() if path.is_dir()):
                sidecar_manifest = read_complete_manifest(sidecar)
                if sidecar_manifest is None:
                    raise DataValidationError(
                        f"incomplete retrained shadow artifact cannot be observed: {sidecar}"
                    )
                sources.append((f"retrained:{sidecar.name}", sidecar, sidecar_manifest))
        qualification_root = directory / "qualification"
        if qualification_root.is_dir():
            for qualification in sorted(
                path for path in qualification_root.iterdir() if path.is_dir()
            ):
                for sidecar in sorted(path for path in qualification.iterdir() if path.is_dir()):
                    sidecar_manifest = read_complete_manifest(sidecar)
                    if sidecar_manifest is None:
                        raise DataValidationError(
                            "incomplete qualification shadow artifact cannot be observed: "
                            f"{sidecar}"
                        )
                    if (
                        sidecar_manifest.get("qualification_only") is not True
                        or sidecar_manifest.get("qualification_run_id") != qualification.name
                        or sidecar_manifest.get("promotion_forbidden") is not True
                        or sidecar_manifest.get("trading_forbidden") is not True
                    ):
                        raise DataValidationError(
                            "qualification shadow artifact lacks isolation contract"
                        )
                    sources.append(
                        (
                            f"qualification:{qualification.name}:{sidecar.name}",
                            sidecar,
                            sidecar_manifest,
                        )
                    )
        for source_name, source_dir, source_manifest in sources:
            _validate_shadow_manifest(source_manifest, signal_date)
            if str(source_manifest.get("production_run_id")) != str(summary.get("run_id")):
                raise DataValidationError(
                    "shadow production_run_id differs from successful production run: "
                    f"{signal_date}"
                )
            predictions_path = source_dir / "predictions.parquet"
            frame = normalize_shadow_lineage(pd.read_parquet(predictions_path))
            _validate_shadow_rows(frame, source_manifest, signal_date)
            frames.append(frame)
            source_key = (
                signal_date if source_name == "production" else f"{signal_date}:{source_name}"
            )
            manifests.append(
                {
                    **source_manifest,
                    "source_signal_date": signal_date,
                    "source_key": source_key,
                }
            )
            source_hashes[source_key] = file_sha256(predictions_path)
    if not frames:
        return pd.DataFrame(), manifests, source_hashes
    combined = pd.concat(frames, ignore_index=True)
    return combined, manifests, source_hashes


def _validate_shadow_manifest(manifest: dict[str, Any], signal_date: str) -> None:
    if manifest.get("artifact_name") != "shadow_prediction_bundle":
        raise DataValidationError("invalid shadow artifact identity")
    if manifest.get("schema_version") != 1:
        raise DataValidationError("unsupported shadow manifest schema")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise DataValidationError("shadow manifest contains no model records")
    for model in models:
        if not isinstance(model, dict):
            raise DataValidationError("invalid shadow model manifest record")
        if model.get("access_policy") != "prospective_production":
            raise DataValidationError(
                f"non-prospective shadow source is prohibited: signal_date={signal_date}"
            )
    encoded = json.dumps(manifest, ensure_ascii=True, sort_keys=True)
    if "frozen_oos_evaluation" in encoded:
        raise DataValidationError("frozen_oos_evaluation shadow source is prohibited")
    if any(value in encoded for value in PROHIBITED_SOURCES):
        raise DataValidationError("historical evaluation lineage is prohibited")


def _validate_shadow_rows(
    frame: DataFrame,
    manifest: dict[str, Any],
    signal_date: str,
) -> None:
    required = {
        "trade_date",
        "ts_code",
        "model_id",
        "model_role",
        "native_horizon",
        "prediction_score",
        "rank",
        "score_percentile",
        "production_run_id",
        "shadow_run_id",
        "prediction_hash",
        "feature_hash",
        "universe_hash",
        "access_policy",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(f"shadow predictions lack required columns: {missing}")
    if frame.empty:
        raise DataValidationError(f"shadow predictions are empty: {signal_date}")
    if set(frame["trade_date"].astype(str)) != {signal_date}:
        raise DataValidationError("shadow prediction signal date mismatch")
    if not set(frame["model_role"].astype(str)).issubset(MODEL_ROLES):
        raise DataValidationError("shadow prediction has unsupported model_role")
    if set(frame["access_policy"].astype(str)) != {"prospective_production"}:
        raise DataValidationError("shadow prediction access policy is not prospective")
    if not set(frame["model_origin"].astype(str)).issubset(MODEL_ORIGINS):
        raise DataValidationError("shadow prediction has unsupported model_origin")
    for column, manifest_key in (
        ("production_run_id", "production_run_id"),
        ("shadow_run_id", "shadow_run_id"),
        ("prediction_hash", "prediction_hash"),
        ("feature_hash", "feature_hash"),
        ("universe_hash", "universe_hash"),
    ):
        if set(frame[column].astype(str)) != {str(manifest.get(manifest_key))}:
            raise DataValidationError(f"shadow row {column} differs from manifest")
    if frame.duplicated(["trade_date", "model_id", "ts_code"]).any():
        raise DataValidationError("shadow predictions contain duplicate model-stock keys")
    native = pd.to_numeric(frame["native_horizon"], errors="coerce").dropna().astype(int)
    if not set(native).issubset(SUPPORTED_HORIZONS):
        raise DataValidationError("shadow predictions contain unsupported native horizon")


def normalize_shadow_lineage(frame: DataFrame) -> DataFrame:
    """Add explicit lineage to legacy research shadow rows without changing scores."""

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
    for column in (
        "parent_model_id",
        "training_request_id",
        "training_run_id",
        "validation_run_id",
        "qualification_run_id",
    ):
        if column not in result:
            result[column] = ""
        else:
            result[column] = result[column].fillna("").astype(str)
    if "qualification_only" not in result:
        result["qualification_only"] = False
    else:
        result["qualification_only"] = result["qualification_only"].fillna(False).astype(bool)
    return result
