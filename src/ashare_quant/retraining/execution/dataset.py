"""Leakage-controlled dataset planning and loading for retraining."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger import (
    _load_and_validate_folds,
    _select_training_fold,
    _validate_experiment,
    _validate_fold_for_experiment,
)
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.ranker_data import RankerDataLoader
from ashare_quant.models.registry import RegisteredModel
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.execution.schemas import DatasetManifest, PreparedTrainingData
from ashare_quant.retraining.readiness.schemas import RetrainingReadinessReport
from ashare_quant.utils.manifest import config_hash


class RetrainingDatasetPreparer:
    """Resolve one approved selection fold and load train/validation rows only."""

    def __init__(
        self,
        *,
        processed_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path

    def prepare(
        self,
        *,
        source_model: RegisteredModel,
        horizon: int,
        readiness: RetrainingReadinessReport,
    ) -> PreparedTrainingData:
        features = _feature_list(Path(source_model.artifact_path) / "feature_list.json")
        feature_hash = feature_list_hash(features)
        if source_model.feature_hash != feature_hash:
            raise DataValidationError("source model feature hash differs from feature list")
        _, plan, experiment = self._plan(horizon, feature_hash, readiness)
        folds_manifest = Path(str(plan["folds_manifest"]))
        _, folds = _load_and_validate_folds(
            folds_manifest,
            expected_manifest_hash=str(plan["folds_manifest_hash"]),
            expected_folds_hash=str(plan["folds_hash"]),
            expected_feature_hash=feature_hash,
        )
        fold = _select_training_fold(experiment, folds)
        _validate_fold_for_experiment(experiment, fold)
        final_test = experiment.get("final_test_period")
        if not isinstance(final_test, dict) or final_test.get("may_select_model") is not False:
            raise DataValidationError("retraining experiment lacks isolated final-test contract")
        if str(fold["validation_end"]) >= str(final_test.get("start_date", "")):
            raise DataValidationError("retraining validation overlaps final-test period")
        loader = RankerDataLoader(
            self.processed_root,
            horizon=horizon,
            minimum_group_size=self.settings.ranker.minimum_group_size,
        )
        train = loader.load(
            str(fold["train_start"]),
            str(fold["train_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        validation = loader.load(
            str(fold["validation_start"]),
            str(fold["validation_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        manifest = DatasetManifest(
            feature_hash=feature_hash,
            feature_manifest_hash=str(readiness.feature_hash),
            universe_hash=str(readiness.universe_hash),
            label_hash=str(readiness.label_hash),
            horizon=horizon,  # type: ignore[arg-type]
            label_name=f"future_excess_ret_{horizon}d",
            train_dates={"start": str(fold["train_start"]), "end": str(fold["train_end"])},
            validation_dates={
                "start": str(fold["validation_start"]),
                "end": str(fold["validation_end"]),
            },
            fold_manifest=str(folds_manifest.resolve()),
            fold_manifest_hash=file_sha256(folds_manifest),
            fold_id=str(fold["fold_id"]),
        )
        return PreparedTrainingData(
            manifest,
            features,
            train,
            validation,
            int(experiment["holding_period"]),
            str(experiment["execution_rule"]),
        )

    def _plan(
        self,
        horizon: int,
        feature_hash: str,
        readiness: RetrainingReadinessReport,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        expected_config = config_hash(self.config_path)
        candidates: list[tuple[str, Path, dict[str, Any], dict[str, Any]]] = []
        for path in (self.reports_root / "horizon_experiments").glob("*/experiment_manifest.json"):
            plan = _json(path)
            if (
                plan.get("config_hash") != expected_config
                or plan.get("feature_hash") != feature_hash
            ):
                continue
            experiments = plan.get("experiments")
            if not isinstance(experiments, list):
                continue
            for raw in experiments:
                if isinstance(raw, dict) and raw.get("horizon") == horizon:
                    experiment = raw
                    _validate_experiment(
                        experiment,
                        feature_hash=feature_hash,
                        universe_hash=str(readiness.universe_hash),
                        config_hash_value=str(expected_config),
                    )
                    candidates.append((str(plan.get("created_time", "")), path, plan, experiment))
        if not candidates:
            raise DataValidationError("no compatible horizon experiment exists for retraining")
        _, path, plan, experiment = max(candidates, key=lambda item: (item[0], str(item[1])))
        return path, plan, experiment


def _feature_list(path: Path) -> tuple[str, ...]:
    payload = _json(path)
    raw = payload.get("features")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise DataValidationError("source model feature list is invalid")
    features = tuple(raw)
    if len(features) != len(set(features)):
        raise DataValidationError("source model feature list contains duplicates")
    return features


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required retraining source is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid retraining source {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"retraining source must contain an object: {path}")
    return payload
