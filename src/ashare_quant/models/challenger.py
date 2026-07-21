"""Leakage-controlled multi-horizon challenger training and publication."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.ranker import feature_importance, fit_ranker, ranker_parameters
from ashare_quant.models.ranker_data import RankerDataLoader
from ashare_quant.models.ranker_metrics import evaluate_ranker
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.utils.manifest import config_hash, current_git_info

CHALLENGER_MANIFEST_SCHEMA_VERSION = 1
HORIZON_PLAN_SCHEMA_VERSION = 2
WALK_FORWARD_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class ChallengerTrainingResult:
    """One immutable, candidate-registered challenger artifact."""

    model_id: str
    experiment_id: str
    horizon: int
    output_dir: Path
    training_rows: int
    validation_rows: int
    validation_rank_ic: float


class ChallengerTrainer:
    """Train fixed Rankers from mature selection folds without reading final-test rows."""

    def __init__(
        self,
        *,
        processed_root: Path,
        models_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.processed_root = processed_root
        self.models_root = models_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path
        self.registry = ModelRegistry(models_root)

    def train(
        self,
        *,
        experiment_id: str | None = None,
        all_horizons: bool = False,
        experiment_manifest: Path | None = None,
    ) -> tuple[ChallengerTrainingResult, ...]:
        """Train one requested horizon or every horizon from one immutable plan."""

        if all_horizons == (experiment_id is not None):
            raise DataValidationError("choose exactly one of --experiment-id or --all-horizons")
        plan_path = self._resolve_experiment_manifest(experiment_manifest)
        plan = _load_json(plan_path, "horizon experiment manifest")
        experiments = self._validate_plan(plan_path, plan)
        selected = (
            experiments
            if all_horizons
            else (_select_experiment(experiments, cast(str, experiment_id)),)
        )
        source_model = self._source_model(plan)
        features = _load_features(Path(source_model.artifact_path) / "feature_list.json")
        if feature_list_hash(features) != str(plan["feature_hash"]):
            raise DataValidationError(
                "source model feature list does not match horizon experiment feature_hash"
            )
        fold_manifest_path = Path(str(plan["folds_manifest"]))
        fold_manifest, folds = _load_and_validate_folds(
            fold_manifest_path,
            expected_manifest_hash=str(plan["folds_manifest_hash"]),
            expected_folds_hash=str(plan["folds_hash"]),
            expected_feature_hash=str(plan["feature_hash"]),
        )
        del fold_manifest
        return tuple(
            self._train_one(
                plan_path=plan_path,
                plan=plan,
                experiment=record,
                folds=folds,
                source_model=source_model,
                features=features,
            )
            for record in selected
        )

    def _resolve_experiment_manifest(self, requested: Path | None) -> Path:
        if requested is not None:
            path = requested / "experiment_manifest.json" if requested.is_dir() else requested
            if not path.is_file():
                raise DataValidationError(f"horizon experiment manifest does not exist: {path}")
            return path.resolve()
        candidates = list(
            (self.reports_root / "horizon_experiments").glob("*/experiment_manifest.json")
        )
        if not candidates:
            raise DataValidationError(
                "no horizon experiment manifest found; run `ashare-quant models horizon-plan`"
            )
        dated: list[tuple[str, str, Path]] = []
        for path in candidates:
            payload = _load_json(path, "horizon experiment manifest")
            dated.append((str(payload.get("created_time", "")), str(path), path))
        return max(dated)[2].resolve()

    def _validate_plan(self, plan_path: Path, plan: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        if plan.get("artifact_name") != "multi_horizon_experiment_plan":
            raise DataValidationError(f"not a horizon experiment plan: {plan_path}")
        if _required_int(plan, "schema_version") != HORIZON_PLAN_SCHEMA_VERSION:
            raise DataValidationError(
                "challenger training requires horizon experiment schema version 2"
            )
        current_config_hash = config_hash(self.config_path)
        if current_config_hash is None:
            raise DataValidationError(f"configuration file does not exist: {self.config_path}")
        if plan.get("config_hash") != current_config_hash:
            raise DataValidationError(
                "horizon experiment config_hash does not match current configuration"
            )
        universe_manifest = self.processed_root / "universe_daily" / "_manifest.json"
        if _file_hash(universe_manifest) != plan.get("universe_hash"):
            raise DataValidationError(
                "current universe manifest does not match horizon experiment universe_hash"
            )
        features_manifest = self.processed_root / "features_daily" / "_manifest.json"
        _load_json(features_manifest, "features manifest")
        final_test = _required_mapping(plan, "final_test_period")
        if final_test.get("may_select_model") is not False:
            raise DataValidationError("final_test_period must explicitly prohibit model selection")
        selection = _required_mapping(plan, "selection_period")
        if str(selection.get("end_date", "")) >= str(final_test.get("start_date", "")):
            raise DataValidationError("selection period must end before final-test period")
        raw_experiments = plan.get("experiments")
        if not isinstance(raw_experiments, list) or not raw_experiments:
            raise DataValidationError("horizon experiment plan contains no experiments")
        if not all(isinstance(record, dict) for record in raw_experiments):
            raise DataValidationError("horizon experiment records must be objects")
        records = tuple(cast(dict[str, Any], record) for record in raw_experiments)
        ids = [str(record.get("experiment_id", "")) for record in records]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise DataValidationError("horizon experiment IDs must be non-empty and unique")
        return records

    def _source_model(self, plan: dict[str, Any]) -> RegisteredModel:
        source_model_id = str(plan.get("source_model_id", ""))
        try:
            return next(
                model for model in self.registry.list_models() if model.model_id == source_model_id
            )
        except StopIteration as error:
            raise DataValidationError(
                f"horizon experiment source model is not registered: {source_model_id}"
            ) from error

    def _train_one(
        self,
        *,
        plan_path: Path,
        plan: dict[str, Any],
        experiment: dict[str, Any],
        folds: tuple[dict[str, Any], ...],
        source_model: RegisteredModel,
        features: tuple[str, ...],
    ) -> ChallengerTrainingResult:
        horizon = _validate_experiment(
            experiment,
            feature_hash=str(plan["feature_hash"]),
            universe_hash=str(plan["universe_hash"]),
            config_hash_value=str(plan["config_hash"]),
        )
        selected_fold = _select_training_fold(experiment, folds)
        _validate_fold_for_experiment(experiment, selected_fold)
        final_test = _required_mapping(experiment, "final_test_period")
        if str(selected_fold["validation_end"]) >= str(final_test["start_date"]):
            raise DataValidationError(
                "selected train/validation fold overlaps the final-test period"
            )
        model_id = _challenger_id(plan, experiment, selected_fold, self.settings)
        output_dir = self.models_root / "challengers" / model_id
        if output_dir.exists():
            raise DataValidationError(f"immutable challenger artifact already exists: {output_dir}")
        if any(model.model_id == model_id for model in self.registry.list_models()):
            raise DataValidationError(f"challenger model_id is already registered: {model_id}")

        loader = RankerDataLoader(
            self.processed_root,
            horizon=horizon,
            minimum_group_size=self.settings.ranker.minimum_group_size,
        )
        train = loader.load(
            str(selected_fold["train_start"]),
            str(selected_fold["train_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        validation = loader.load(
            str(selected_fold["validation_start"]),
            str(selected_fold["validation_end"]),
            features,
            self.settings.ranker.relevance_grades,
        )
        model = fit_ranker(train, validation, self.settings.ranker)
        predictions = np.asarray(model.predict(validation.features), dtype=float)
        validation_metrics = evaluate_ranker(
            validation,
            predictions,
            self.settings.ranker.ndcg_at,
            self.settings.ranker.portfolio_fractions,
        )
        importance = feature_importance(model, features)
        manifest = self._manifest(
            model_id=model_id,
            plan_path=plan_path,
            plan=plan,
            experiment=experiment,
            fold=selected_fold,
            source_model=source_model,
            features=features,
            training_rows=len(train.frame),
            validation_rows=len(validation.frame),
        )
        self._publish(
            output_dir=output_dir,
            model=model,
            features=features,
            validation_metrics=validation_metrics,
            importance=importance,
            manifest=manifest,
        )
        registered = self.registry.register_model(
            output_dir,
            model_id=model_id,
            model_type="lightgbm_ranker",
            operator_command=f"train-challenger {model_id}",
        )
        if registered.status != "candidate":
            raise DataValidationError("new challenger was not registered as candidate")
        return ChallengerTrainingResult(
            model_id=model_id,
            experiment_id=model_id,
            horizon=horizon,
            output_dir=output_dir,
            training_rows=len(train.frame),
            validation_rows=len(validation.frame),
            validation_rank_ic=cast(float, validation_metrics["rank_ic"]),
        )

    def _manifest(
        self,
        *,
        model_id: str,
        plan_path: Path,
        plan: dict[str, Any],
        experiment: dict[str, Any],
        fold: dict[str, Any],
        source_model: RegisteredModel,
        features: tuple[str, ...],
        training_rows: int,
        validation_rows: int,
    ) -> dict[str, Any]:
        git = current_git_info()
        feature_manifest = self.processed_root / "features_daily" / "_manifest.json"
        universe_manifest = self.processed_root / "universe_daily" / "_manifest.json"
        return {
            "schema_version": CHALLENGER_MANIFEST_SCHEMA_VERSION,
            "artifact_name": "lightgbm_ranker_challenger",
            "model_id": model_id,
            "experiment_id": model_id,
            "source_horizon_experiment_id": experiment["experiment_id"],
            "source_model_id": source_model.model_id,
            "creation_time": plan.get("created_time"),
            "horizon": experiment["horizon"],
            "holding_period": experiment["holding_period"],
            "execution_rule": experiment["execution_rule"],
            "label_name": experiment["label_name"],
            "feature_hash": feature_list_hash(features),
            "feature_list_hash": feature_list_hash(features),
            "feature_count": len(features),
            "universe_hash": experiment["universe_hash"],
            "fold_manifest": str(Path(str(plan["folds_manifest"])).resolve()),
            "fold_manifest_hash": plan["folds_manifest_hash"],
            "fold_id": fold["fold_id"],
            "train_dates": {"start": fold["train_start"], "end": fold["train_end"]},
            "validation_dates": {
                "start": fold["validation_start"],
                "end": fold["validation_end"],
            },
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "config_hash": plan["config_hash"],
            "git_commit": git["commit"],
            "git_dirty": git["dirty"],
            "training_rows": training_rows,
            "validation_rows": validation_rows,
            "fixed_parameters": ranker_parameters(self.settings.ranker),
            "source_manifests": {
                "horizon_experiment": {
                    "path": str(plan_path.resolve()),
                    "sha256": _file_hash(plan_path),
                },
                "features_daily": {
                    "path": str(feature_manifest.resolve()),
                    "sha256": _file_hash(feature_manifest),
                },
                "universe_daily": {
                    "path": str(universe_manifest.resolve()),
                    "sha256": _file_hash(universe_manifest),
                },
            },
            "isolation_contract": {
                "selection_fold_only": True,
                "final_test_labels_loaded": False,
                "evaluation_labels_loaded": False,
                "production_observation_loaded": False,
                "champion_modified": False,
                "automatic_promotion": False,
            },
        }

    def _publish(
        self,
        *,
        output_dir: Path,
        model: lgb.LGBMRanker,
        features: tuple[str, ...],
        validation_metrics: dict[str, object],
        importance: list[dict[str, object]],
        manifest: dict[str, Any],
    ) -> None:
        parent = output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent, prefix=".challenger-") as temporary:
            directory = Path(temporary)
            model.booster_.save_model(str(directory / "model.txt"))
            _write_json(
                directory / "feature_list.json",
                {
                    "feature_count": len(features),
                    "feature_hash": feature_list_hash(features),
                    "features": list(features),
                },
            )
            _write_json(
                directory / "metrics.json",
                {
                    "metric_scope": "selection-fold validation only; final test not loaded",
                    "validation": validation_metrics,
                    "test": {},
                    "feature_importance": importance,
                },
            )
            _write_json(directory / "manifest.json", manifest)
            if output_dir.exists():
                raise DataValidationError(
                    f"immutable challenger artifact already exists: {output_dir}"
                )
            directory.rename(output_dir)


def _select_experiment(experiments: tuple[dict[str, Any], ...], requested: str) -> dict[str, Any]:
    matches = [
        record
        for record in experiments
        if requested
        in {
            str(record.get("experiment_id", "")),
            str(record.get("name", "")),
            f"experiment_c_{record.get('name', '')}",
        }
    ]
    if len(matches) != 1:
        available = sorted(f"experiment_c_{record.get('name')}" for record in experiments)
        raise DataValidationError(
            f"horizon experiment is not uniquely available: {requested}; available={available}"
        )
    return matches[0]


def _validate_experiment(
    experiment: dict[str, Any],
    *,
    feature_hash: str,
    universe_hash: str,
    config_hash_value: str,
) -> int:
    horizon = _required_int(experiment, "horizon")
    if _required_int(experiment, "holding_period") != horizon:
        raise DataValidationError("challenger holding_period does not match horizon")
    expected_label = f"future_excess_ret_{horizon}d"
    if experiment.get("label_name") != expected_label:
        raise DataValidationError(
            f"label_name does not match horizon {horizon}: expected {expected_label}"
        )
    if experiment.get("execution_rule") != "next_open":
        raise DataValidationError("challenger experiment requires next_open execution semantics")
    if experiment.get("feature_hash") != feature_hash:
        raise DataValidationError("experiment feature_hash differs from plan")
    if experiment.get("universe_hash") != universe_hash:
        raise DataValidationError("experiment universe_hash differs from plan")
    if experiment.get("config_hash") != config_hash_value:
        raise DataValidationError("experiment config_hash differs from plan")
    if _required_int(experiment, "label_maturity_sessions") != horizon + 1:
        raise DataValidationError("experiment label maturity does not match horizon")
    selection = _required_mapping(experiment, "selection_period")
    final_test = _required_mapping(experiment, "final_test_period")
    if selection.get("may_select_model") is not True:
        raise DataValidationError("selection_period must explicitly permit challenger comparison")
    if final_test.get("may_select_model") is not False:
        raise DataValidationError("final_test_period must prohibit model selection")
    selection_ids = _period_fold_ids(selection)
    test_ids = _period_fold_ids(final_test)
    if selection_ids & test_ids:
        raise DataValidationError("selection and final-test fold references overlap")
    return horizon


def _select_training_fold(
    experiment: dict[str, Any], folds: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    selection = _required_mapping(experiment, "selection_period")
    references = selection.get("folds")
    if not isinstance(references, list) or not references:
        raise DataValidationError("horizon selection_period contains no folds")
    reference = max(
        (cast(dict[str, Any], item) for item in references if isinstance(item, dict)),
        key=lambda item: (str(item.get("evaluation_end", "")), str(item.get("fold_id", ""))),
        default=None,
    )
    if reference is None:
        raise DataValidationError("horizon selection fold references are invalid")
    fold_id = str(reference.get("fold_id", ""))
    try:
        return next(fold for fold in folds if str(fold.get("fold_id")) == fold_id)
    except StopIteration as error:
        raise DataValidationError(f"selection fold is absent from folds v2: {fold_id}") from error


def _validate_fold_for_experiment(experiment: dict[str, Any], fold: dict[str, Any]) -> None:
    required_dates = (
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "evaluation_start",
        "evaluation_end",
    )
    if any(not _valid_date(str(fold.get(field, ""))) for field in required_dates):
        raise DataValidationError(f"fold contains invalid date boundaries: {fold.get('fold_id')}")
    if not (
        str(fold["train_start"])
        <= str(fold["train_end"])
        < str(fold["validation_start"])
        <= str(fold["validation_end"])
        < str(fold["evaluation_start"])
        <= str(fold["evaluation_end"])
    ):
        raise DataValidationError(f"fold chronology is invalid: {fold.get('fold_id')}")
    if _required_int(fold, "purge_sessions") < _required_int(experiment, "required_purge_sessions"):
        raise DataValidationError("fold purge is unsafe for challenger horizon")
    if _required_int(fold, "embargo_sessions") < _required_int(
        experiment, "required_embargo_sessions"
    ):
        raise DataValidationError("fold embargo is unsafe for challenger horizon")


def _load_and_validate_folds(
    manifest_path: Path,
    *,
    expected_manifest_hash: str,
    expected_folds_hash: str,
    expected_feature_hash: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if _file_hash(manifest_path) != expected_manifest_hash:
        raise DataValidationError("walk-forward manifest hash differs from horizon plan")
    manifest = _load_json(manifest_path, "walk-forward manifest")
    if manifest.get("artifact_name") != "purged_walk_forward_plan":
        raise DataValidationError("fold manifest is not a purged walk-forward plan")
    if _required_int(manifest, "schema_version") != WALK_FORWARD_SCHEMA_VERSION:
        raise DataValidationError("challenger training requires fold schema version 2")
    if manifest.get("feature_hash") != expected_feature_hash:
        raise DataValidationError("walk-forward feature_hash differs from horizon plan")
    outputs = _required_mapping(manifest, "outputs")
    folds_path = Path(str(outputs.get("folds", "")))
    if not folds_path.is_absolute():
        folds_path = manifest_path.parent / folds_path
    if _file_hash(folds_path) != expected_folds_hash:
        raise DataValidationError("walk-forward folds hash differs from horizon plan")
    payload = _load_json(folds_path, "walk-forward folds")
    if _required_int(payload, "schema_version") != WALK_FORWARD_SCHEMA_VERSION:
        raise DataValidationError("challenger training requires folds schema version 2")
    raw_folds = payload.get("folds")
    if not isinstance(raw_folds, list) or not raw_folds:
        raise DataValidationError("walk-forward folds are missing or empty")
    folds = tuple(cast(dict[str, Any], fold) for fold in raw_folds if isinstance(fold, dict))
    if len(folds) != len(raw_folds):
        raise DataValidationError("walk-forward fold entries must be objects")
    fold_ids = [str(fold.get("fold_id", "")) for fold in folds]
    if any(not fold_id for fold_id in fold_ids) or len(fold_ids) != len(set(fold_ids)):
        raise DataValidationError("walk-forward fold IDs must be non-empty and unique")
    return manifest, folds


def _source_identity(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    fold: dict[str, Any],
    settings: AppSettings,
) -> str:
    payload = {
        "plan_identity_hash": plan.get("plan_identity_hash"),
        "horizon_experiment_id": experiment.get("experiment_id"),
        "fold_id": fold.get("fold_id"),
        "feature_hash": plan.get("feature_hash"),
        "universe_hash": plan.get("universe_hash"),
        "config_hash": plan.get("config_hash"),
        "fixed_parameters": ranker_parameters(settings.ranker),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _challenger_id(
    plan: dict[str, Any],
    experiment: dict[str, Any],
    fold: dict[str, Any],
    settings: AppSettings,
) -> str:
    name = str(experiment.get("name", ""))
    if not name or "/" in name or "\\" in name:
        raise DataValidationError(f"invalid horizon experiment name: {name}")
    return f"experiment_c_{name}_{_source_identity(plan, experiment, fold, settings)[:16]}"


def _period_fold_ids(period: dict[str, Any]) -> set[str]:
    folds = period.get("folds")
    if not isinstance(folds, list):
        raise DataValidationError("experiment period folds must be an array")
    return {
        str(fold.get("fold_id", ""))
        for fold in folds
        if isinstance(fold, dict) and str(fold.get("fold_id", ""))
    }


def _load_features(path: Path) -> tuple[str, ...]:
    payload = _load_json(path, "source feature list")
    raw = payload.get("features")
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise DataValidationError("source feature list must contain non-empty string features")
    features = tuple(str(item) for item in raw)
    if len(features) != len(set(features)):
        raise DataValidationError("source feature list contains duplicate names")
    declared = payload.get("feature_hash")
    if declared is not None and declared != feature_list_hash(features):
        raise DataValidationError("source feature list hash is invalid")
    return features


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


def _required_mapping(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = payload.get(field)
    if not isinstance(value, dict):
        raise DataValidationError(f"challenger {field} must be an object")
    return value


def _required_int(payload: dict[str, Any], field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataValidationError(f"challenger {field} must be an integer")
    return value


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise DataValidationError(f"manifest source does not exist: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_date(value: str) -> bool:
    return len(value) == 8 and value.isdigit()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
