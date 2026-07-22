"""Immutable, label-free historical predictions for registered challengers."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.inference import (
    PredictionModel,
    score_registered_model_range,
)
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

PREDICTION_MANIFEST_SCHEMA_VERSION = 1

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class ChallengerPredictionResult:
    """Published immutable challenger predictions for one mature final-test scope."""

    model_id: str
    horizon: int
    prediction_rows: int
    prediction_dates: int
    output_dir: Path
    predictions: DataFrame


class ChallengerPredictionEngine:
    """Score a candidate using production-equivalent inputs without loading labels."""

    def __init__(
        self,
        *,
        registry: ModelRegistry,
        processed_root: Path,
        reports_root: Path,
        config_path: Path,
        model_loader: Callable[[Path], PredictionModel] | None = None,
    ) -> None:
        self.registry = registry
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.config_path = config_path
        self._model_loader = model_loader

    def predict(self, model_id: str) -> ChallengerPredictionResult:
        """Publish scores for the mature final-test ranges bound to the challenger."""

        challenger = _registered_challenger(self.registry, model_id)
        artifact = Path(challenger.artifact_path)
        challenger_manifest = _load_json(artifact / "manifest.json", "challenger manifest")
        horizon_record, ranges, maximum_mature_date = _evaluation_contract(challenger_manifest)
        _validate_current_sources(self.processed_root, challenger_manifest)
        start_date = min(start for start, _ in ranges)
        end_date = max(end for _, end in ranges)
        if end_date > maximum_mature_date:
            raise DataValidationError(
                "challenger final-test range contains labels that are not mature: "
                f"end={end_date} maximum_mature={maximum_mature_date}"
            )
        identity = _prediction_identity(
            challenger,
            challenger_manifest,
            horizon_record,
            ranges,
            processed_root=self.processed_root,
            config_path=self.config_path,
        )
        output_dir = self.reports_root / "challenger_predictions" / model_id
        existing = _existing_result(output_dir, identity)
        if existing is not None:
            return existing
        batch = score_registered_model_range(
            challenger,
            processed_root=self.processed_root,
            start_date=start_date,
            end_date=end_date,
            allowed_ranges=ranges,
            model_loader=self._model_loader,
        )
        predictions = batch.predictions
        if predictions["trade_date"].astype(str).max() > maximum_mature_date:
            raise DataValidationError("challenger predictions include an immature signal date")
        manifest = _prediction_manifest(
            identity=identity,
            challenger=challenger,
            challenger_manifest=challenger_manifest,
            horizon_record=horizon_record,
            ranges=ranges,
            maximum_mature_date=maximum_mature_date,
            predictions=predictions,
            processed_root=self.processed_root,
            config_path=self.config_path,
        )
        _publish(output_dir, predictions, manifest)
        return ChallengerPredictionResult(
            model_id=model_id,
            horizon=int(horizon_record["horizon"]),
            prediction_rows=len(predictions),
            prediction_dates=predictions["trade_date"].nunique(),
            output_dir=output_dir,
            predictions=predictions,
        )


def _registered_challenger(registry: ModelRegistry, model_id: str) -> RegisteredModel:
    try:
        record = next(model for model in registry.list_models() if model.model_id == model_id)
    except StopIteration as error:
        raise DataValidationError(f"challenger model is not registered: {model_id}") from error
    if record.status != "candidate":
        raise DataValidationError(
            f"challenger prediction requires candidate status: {model_id}={record.status}"
        )
    manifest = _load_json(Path(record.artifact_path) / "manifest.json", "challenger manifest")
    if manifest.get("artifact_name") != "lightgbm_ranker_challenger":
        raise DataValidationError(f"registered model is not a challenger artifact: {model_id}")
    return record


def _evaluation_contract(
    challenger_manifest: dict[str, Any],
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...], str]:
    source = challenger_manifest.get("source_manifests")
    if not isinstance(source, dict) or not isinstance(source.get("horizon_experiment"), dict):
        raise DataValidationError("challenger manifest lacks horizon experiment provenance")
    reference = source["horizon_experiment"]
    plan_path = Path(str(reference.get("path", "")))
    if _file_hash(plan_path) != reference.get("sha256"):
        raise DataValidationError("horizon experiment manifest hash has changed")
    plan = _load_json(plan_path, "horizon experiment manifest")
    source_id = str(challenger_manifest.get("source_horizon_experiment_id", ""))
    experiments = plan.get("experiments")
    if not isinstance(experiments, list):
        raise DataValidationError("horizon experiment plan contains no experiments")
    try:
        record = next(
            item
            for item in experiments
            if isinstance(item, dict) and str(item.get("experiment_id")) == source_id
        )
    except StopIteration as error:
        raise DataValidationError(
            f"challenger horizon experiment is absent from source plan: {source_id}"
        ) from error
    horizon = _required_int(record, "horizon")
    expected_label = f"future_excess_ret_{horizon}d"
    if record.get("label_name") != expected_label:
        raise DataValidationError("challenger horizon label_name is inconsistent")
    if challenger_manifest.get("label_name") != expected_label:
        raise DataValidationError("challenger model label_name differs from horizon plan")
    if challenger_manifest.get("feature_hash") != record.get("feature_hash"):
        raise DataValidationError("challenger feature_hash differs from horizon plan")
    if challenger_manifest.get("universe_hash") != record.get("universe_hash"):
        raise DataValidationError("challenger universe_hash differs from horizon plan")
    if challenger_manifest.get("execution_rule") != record.get("execution_rule"):
        raise DataValidationError("challenger execution rule differs from horizon plan")
    if challenger_manifest.get("holding_period") != horizon:
        raise DataValidationError("challenger holding period differs from label horizon")
    final_test = record.get("final_test_period")
    if not isinstance(final_test, dict) or final_test.get("may_select_model") is not False:
        raise DataValidationError("challenger final-test contract is invalid")
    raw_folds = final_test.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise DataValidationError("challenger has no final-test fold ranges")
    ranges = tuple(
        (str(fold.get("evaluation_start", "")), str(fold.get("evaluation_end", "")))
        for fold in raw_folds
        if isinstance(fold, dict)
    )
    if len(ranges) != len(raw_folds) or any(
        not _valid_date(start) or not _valid_date(end) or start > end for start, end in ranges
    ):
        raise DataValidationError("challenger final-test fold ranges are invalid")
    if tuple(sorted(ranges)) != ranges:
        raise DataValidationError("challenger final-test fold ranges are not deterministic")
    maximum_mature_date = str(record.get("maximum_mature_evaluation_date", ""))
    if not _valid_date(maximum_mature_date):
        raise DataValidationError("challenger horizon lacks a maturity cutoff")
    return record, ranges, maximum_mature_date


def _validate_current_sources(processed_root: Path, challenger_manifest: dict[str, Any]) -> None:
    source = challenger_manifest.get("source_manifests")
    assert isinstance(source, dict)
    for name in ("features_daily", "universe_daily"):
        reference = source.get(name)
        if not isinstance(reference, dict):
            raise DataValidationError(f"challenger manifest lacks {name} provenance")
        current_path = processed_root / name / "_manifest.json"
        if _file_hash(current_path) != reference.get("sha256"):
            raise DataValidationError(
                f"current {name} manifest differs from challenger training provenance"
            )
    universe_path = processed_root / "universe_daily" / "_manifest.json"
    if _file_hash(universe_path) != challenger_manifest.get("universe_hash"):
        raise DataValidationError("current universe manifest differs from challenger universe_hash")


def _prediction_identity(
    challenger: RegisteredModel,
    challenger_manifest: dict[str, Any],
    horizon_record: dict[str, Any],
    ranges: tuple[tuple[str, str], ...],
    *,
    processed_root: Path,
    config_path: Path,
) -> str:
    git = current_git_info()
    payload = {
        "model_id": challenger.model_id,
        "model_manifest_hash": _file_hash(Path(challenger.artifact_path) / "manifest.json"),
        "feature_hash": challenger.feature_hash,
        "universe_hash": challenger_manifest.get("universe_hash"),
        "features_manifest_hash": _file_hash(processed_root / "features_daily" / "_manifest.json"),
        "horizon": horizon_record.get("horizon"),
        "label_name": horizon_record.get("label_name"),
        "ranges": ranges,
        "maximum_mature_evaluation_date": horizon_record.get("maximum_mature_evaluation_date"),
        "config_hash": config_hash(config_path),
        "git_commit": git["commit"],
    }
    return _payload_hash(payload)


def _prediction_manifest(
    *,
    identity: str,
    challenger: RegisteredModel,
    challenger_manifest: dict[str, Any],
    horizon_record: dict[str, Any],
    ranges: tuple[tuple[str, str], ...],
    maximum_mature_date: str,
    predictions: DataFrame,
    processed_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    git = current_git_info()
    return {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "artifact_name": "challenger_predictions",
        "prediction_identity": identity,
        "model_id": challenger.model_id,
        "model_status": challenger.status,
        "feature_hash": challenger.feature_hash,
        "universe_hash": challenger_manifest["universe_hash"],
        "horizon": horizon_record["horizon"],
        "holding_period": horizon_record["holding_period"],
        "execution_rule": horizon_record["execution_rule"],
        "label_name": horizon_record["label_name"],
        "evaluation_ranges": [{"start_date": start, "end_date": end} for start, end in ranges],
        "maximum_mature_evaluation_date": maximum_mature_date,
        "prediction_rows": len(predictions),
        "prediction_dates": predictions["trade_date"].nunique(),
        "minimum_prediction_date": str(predictions["trade_date"].min()),
        "maximum_prediction_date": str(predictions["trade_date"].max()),
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "config_path": str(config_path),
        "config_hash": config_hash(config_path),
        "input_manifests": {
            "model": _file_hash(Path(challenger.artifact_path) / "manifest.json"),
            "features_daily": _file_hash(processed_root / "features_daily" / "_manifest.json"),
            "universe_daily": _file_hash(processed_root / "universe_daily" / "_manifest.json"),
        },
        "isolation_contract": {
            "labels_loaded": False,
            "future_prices_loaded": False,
            "production_observation_loaded": False,
            "champion_modified": False,
            "candidate_selection_loaded": False,
        },
    }


def _existing_result(output_dir: Path, identity: str) -> ChallengerPredictionResult | None:
    if not output_dir.exists():
        return None
    manifest = _load_json(output_dir / "manifest.json", "challenger prediction manifest")
    if manifest.get("prediction_identity") != identity:
        raise DataValidationError(
            f"immutable challenger prediction artifact has a different identity: {output_dir}"
        )
    path = output_dir / "predictions.parquet"
    if not path.is_file():
        raise DataValidationError("challenger prediction artifact is incomplete")
    predictions = pd.read_parquet(path)
    return ChallengerPredictionResult(
        model_id=str(manifest["model_id"]),
        horizon=int(manifest["horizon"]),
        prediction_rows=len(predictions),
        prediction_dates=predictions["trade_date"].nunique(),
        output_dir=output_dir,
        predictions=predictions,
    )


def _publish(output_dir: Path, predictions: DataFrame, manifest: dict[str, Any]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix=".predictions-") as temporary:
        staging = Path(temporary)
        predictions.to_parquet(staging / "predictions.parquet", index=False)
        atomic_write_json(staging / "manifest.json", manifest)
        if output_dir.exists():
            raise DataValidationError(
                f"immutable challenger prediction artifact already exists: {output_dir}"
            )
        staging.rename(output_dir)


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"challenger {field} must be an integer")
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must be a JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise DataValidationError(f"manifest source does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _valid_date(value: str) -> bool:
    return len(value) == 8 and value.isdigit()
