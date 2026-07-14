"""Fixed-parameter LightGBM Ranker baseline experiments."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np

from ashare_quant.config.settings import AppSettings, RankerSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import (
    feature_list_hash,
    load_recommended_features,
    load_robust_features,
)
from ashare_quant.models.ranker_data import RankerDataLoader, RankerDataset
from ashare_quant.models.ranker_metrics import evaluate_ranker
from ashare_quant.utils.manifest import config_hash, current_git_info, read_manifest


@dataclass(frozen=True, slots=True)
class RankerExperimentResult:
    """Summarize one persisted Ranker baseline experiment."""

    experiment_name: str
    experiment_id: str
    output_dir: Path
    feature_count: int
    validation_rank_ic: float
    test_rank_ic: float


class RankerBaselineRunner:
    """Run top-50 and manual robust-subset experiments without test tuning."""

    def __init__(
        self,
        processed_root: Path,
        output_root: Path,
        settings: AppSettings,
        config_path: Path,
    ) -> None:
        self.processed_root = processed_root
        self.output_root = output_root
        self.settings = settings
        self.config_path = config_path

    def run(
        self,
        recommended_features_path: Path | None = None,
        robust_features_path: Path | None = None,
    ) -> tuple[RankerExperimentResult, RankerExperimentResult]:
        """Execute Experiment A then B with independent fixed-parameter models."""

        ranker = self.settings.ranker
        if ranker.label_horizon != 5:
            raise DataValidationError("Phase 7 baseline requires label_horizon=5")
        recommended = load_recommended_features(
            recommended_features_path or ranker.recommended_features_path
        )
        robust = load_robust_features(robust_features_path or ranker.robust_features_path)
        if not set(robust).issubset(recommended):
            raise DataValidationError(
                "Experiment B robust features must be a subset of Experiment A top_50"
            )
        run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        experiment_a = self._run_one("experiment_a_top50", recommended, run_stamp)
        experiment_b = self._run_one("experiment_b_robust", robust, run_stamp)
        return experiment_a, experiment_b

    def _run_one(
        self, experiment_name: str, features: tuple[str, ...], run_stamp: str
    ) -> RankerExperimentResult:
        """Train on train only, then evaluate validation and untouched test periods."""

        ranker = self.settings.ranker
        loader = RankerDataLoader(
            self.processed_root, ranker.label_horizon, ranker.minimum_group_size
        )
        train = loader.load(ranker.train_start, ranker.train_end, features, ranker.relevance_grades)
        validation = loader.load(
            ranker.validation_start,
            ranker.validation_end,
            features,
            ranker.relevance_grades,
        )
        model = fit_ranker(train, validation, ranker)
        validation_predictions = model.predict(validation.features)
        validation_metrics = evaluate_ranker(
            validation,
            np.asarray(validation_predictions),
            ranker.ndcg_at,
            ranker.portfolio_fractions,
        )

        # Test rows are loaded only after the fixed model has been fitted and validation reported.
        test = loader.load(ranker.test_start, ranker.test_end, features, ranker.relevance_grades)
        test_predictions = model.predict(test.features)
        test_metrics = evaluate_ranker(
            test,
            np.asarray(test_predictions),
            ranker.ndcg_at,
            ranker.portfolio_fractions,
        )
        importance = feature_importance(model, features)
        experiment_id = f"{experiment_name}_{run_stamp}"
        output_dir = self._persist(
            experiment_id,
            experiment_name,
            features,
            model,
            train,
            validation_metrics,
            test_metrics,
            importance,
        )
        return RankerExperimentResult(
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            output_dir=output_dir,
            feature_count=len(features),
            validation_rank_ic=cast(float, validation_metrics["rank_ic"]),
            test_rank_ic=cast(float, test_metrics["rank_ic"]),
        )

    def _persist(
        self,
        experiment_id: str,
        experiment_name: str,
        features: tuple[str, ...],
        model: lgb.LGBMRanker,
        train: RankerDataset,
        validation_metrics: dict[str, object],
        test_metrics: dict[str, object],
        importance: list[dict[str, object]],
    ) -> Path:
        """Atomically publish model and provenance only after successful evaluation."""

        final_dir = self.output_root / experiment_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            raise DataValidationError(f"model experiment directory already exists: {final_dir}")
        with tempfile.TemporaryDirectory(dir=self.output_root) as temporary:
            directory = Path(temporary)
            model.booster_.save_model(str(directory / "model.txt"))
            feature_payload = {
                "experiment_name": experiment_name,
                "feature_count": len(features),
                "feature_hash": feature_list_hash(features),
                "features": list(features),
            }
            write_json(directory / "feature_list.json", feature_payload)
            write_json(
                directory / "metrics.json",
                {
                    "metric_scope": "ranking diagnostics and portfolio proxy; not a backtest",
                    "validation": validation_metrics,
                    "test": test_metrics,
                    "feature_importance": importance,
                },
            )
            git_info = current_git_info()
            ranker = self.settings.ranker
            write_json(
                directory / "manifest.json",
                {
                    "schema_version": 1,
                    "artifact_name": "lightgbm_ranker_baseline",
                    "experiment_id": experiment_id,
                    "experiment_name": experiment_name,
                    "completed_at": datetime.now(UTC).isoformat(),
                    "git_commit": git_info["commit"],
                    "git_dirty": git_info["dirty"],
                    "config_path": str(self.config_path),
                    "config_hash": config_hash(self.config_path),
                    "feature_list_hash": feature_list_hash(features),
                    "feature_count": len(features),
                    "label_horizon": ranker.label_horizon,
                    "target": "future_excess_ret_5d",
                    "ranker_relevance": "within-trade-date quantile grade",
                    "train_start": ranker.train_start,
                    "train_end": ranker.train_end,
                    "validation_start": ranker.validation_start,
                    "validation_end": ranker.validation_end,
                    "test_start": ranker.test_start,
                    "test_end": ranker.test_end,
                    "train_rows": len(train.frame),
                    "train_groups": len(train.groups),
                    "fixed_parameters": ranker_parameters(ranker),
                    "source_manifests": {
                        artifact: read_manifest(self.processed_root / artifact)
                        for artifact in ("features_daily", "labels_forward", "universe_daily")
                    },
                },
            )
            directory.rename(final_dir)
        return final_dir


def fit_ranker(
    train: RankerDataset, validation: RankerDataset, settings: RankerSettings
) -> lgb.LGBMRanker:
    """Fit one fixed lambdarank baseline without early stopping or parameter search."""

    parameters = ranker_parameters(settings)
    model = lgb.LGBMRanker(**parameters)
    model.fit(
        train.features,
        train.relevance,
        group=train.groups,
        eval_set=[(validation.features, validation.relevance)],
        eval_group=[validation.groups],
        eval_at=list(settings.ndcg_at),
        feature_name=list(train.feature_names),
    )
    return model


def ranker_parameters(settings: RankerSettings) -> dict[str, Any]:
    """Return the complete fixed baseline parameter set recorded in manifests."""

    return {
        "objective": "lambdarank",
        "metric": "ndcg",
        "n_estimators": settings.n_estimators,
        "learning_rate": settings.learning_rate,
        "num_leaves": settings.num_leaves,
        "min_child_samples": settings.min_child_samples,
        "colsample_bytree": settings.feature_fraction,
        "subsample": settings.bagging_fraction,
        "subsample_freq": settings.bagging_freq,
        "reg_alpha": settings.reg_alpha,
        "reg_lambda": settings.reg_lambda,
        "random_state": settings.random_seed,
        "n_jobs": -1,
        "verbosity": -1,
        "label_gain": [2**grade - 1 for grade in range(settings.relevance_grades)],
    }


def feature_importance(model: lgb.LGBMRanker, features: tuple[str, ...]) -> list[dict[str, object]]:
    """Return gain and split importance in configured feature order."""

    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    splits = booster.feature_importance(importance_type="split")
    return [
        {"feature": feature, "gain": float(gain), "split": int(split)}
        for feature, gain, split in zip(features, gains, splits, strict=True)
    ]


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one JSON file inside an unpublished temporary experiment directory."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
