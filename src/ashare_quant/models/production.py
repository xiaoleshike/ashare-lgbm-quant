"""Final production LightGBM Ranker training pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import (
    feature_list_hash,
    load_robust_features,
)
from ashare_quant.models.ranker import feature_importance, ranker_parameters
from ashare_quant.models.ranker_data import RankerDataLoader, RankerDataset
from ashare_quant.utils.manifest import config_hash, current_git_info, read_manifest


@dataclass(frozen=True, slots=True)
class ProductionTrainingResult:
    """Summary of the published production model artifact."""

    output_dir: Path
    feature_count: int
    train_rows: int
    train_groups: int
    train_start: str
    train_end: str


class ProductionRankerTrainer:
    """Train the final model on the full approved historical period."""

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

    def train(self, feature_list_path: Path | None = None) -> ProductionTrainingResult:
        """Train and atomically publish `models/production`."""

        production = self.settings.production_model
        ranker = self.settings.ranker
        if ranker.label_horizon != 5:
            raise DataValidationError("production Ranker training requires label_horizon=5")
        features = load_robust_features(feature_list_path or production.feature_list_path)
        loader = RankerDataLoader(
            self.processed_root, ranker.label_horizon, ranker.minimum_group_size
        )
        train = loader.load(
            production.train_start,
            production.train_end,
            features,
            ranker.relevance_grades,
        )
        model = fit_production_ranker(train, self.settings)
        output_dir = self._persist(features, model, train)
        return ProductionTrainingResult(
            output_dir=output_dir,
            feature_count=len(features),
            train_rows=len(train.frame),
            train_groups=len(train.groups),
            train_start=production.train_start,
            train_end=production.train_end,
        )

    def _persist(
        self,
        features: tuple[str, ...],
        model: lgb.LGBMRanker,
        train: RankerDataset,
    ) -> Path:
        final_dir = self.output_root / self.settings.production_model.output_dir_name
        self.output_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output_root) as temporary:
            directory = Path(temporary)
            model.booster_.save_model(str(directory / "model.txt"))
            write_json(
                directory / "feature_list.json",
                {
                    "artifact_name": "production_ranker_feature_list",
                    "feature_count": len(features),
                    "feature_hash": feature_list_hash(features),
                    "features": list(features),
                    "source_path": str(self.settings.production_model.feature_list_path),
                },
            )
            write_json(
                directory / "metrics.json",
                {
                    "metric_scope": "training provenance only; no validation or test evaluation",
                    "train_rows": len(train.frame),
                    "train_groups": len(train.groups),
                    "train_min_date": str(train.frame["trade_date"].min()),
                    "train_max_date": str(train.frame["trade_date"].max()),
                    "unique_train_dates": int(train.frame["trade_date"].nunique()),
                    "feature_importance": feature_importance(model, features),
                },
            )
            write_json(directory / "manifest.json", self._manifest(features, train))
            replace_directory_atomically(directory, final_dir)
        return final_dir

    def _manifest(self, features: tuple[str, ...], train: RankerDataset) -> dict[str, Any]:
        git_info = current_git_info()
        production = self.settings.production_model
        ranker = self.settings.ranker
        return {
            "schema_version": 1,
            "artifact_name": "production_lightgbm_ranker",
            "output_dir": str(self.output_root / production.output_dir_name),
            "completed_at": datetime.now(UTC).isoformat(),
            "git_commit": git_info["commit"],
            "git_dirty": git_info["dirty"],
            "config_path": str(self.config_path),
            "config_hash": config_hash(self.config_path),
            "training_start": production.train_start,
            "training_end": production.train_end,
            "training_min_date": str(train.frame["trade_date"].min()),
            "training_max_date": str(train.frame["trade_date"].max()),
            "train_rows": len(train.frame),
            "train_groups": len(train.groups),
            "feature_list_path": str(production.feature_list_path),
            "feature_list_hash": feature_list_hash(features),
            "feature_count": len(features),
            "label_horizon": ranker.label_horizon,
            "target": "future_excess_ret_5d",
            "ranker_relevance": "within-trade-date quantile grade",
            "fixed_parameters": ranker_parameters(ranker),
            "source_manifests": {
                artifact: read_manifest(self.processed_root / artifact)
                for artifact in ("features_daily", "labels_forward", "universe_daily")
            },
        }


def fit_production_ranker(train: RankerDataset, settings: AppSettings) -> lgb.LGBMRanker:
    """Fit one fixed production model without validation or test data."""

    model = lgb.LGBMRanker(**ranker_parameters(settings.ranker))
    model.fit(
        train.features,
        train.relevance,
        group=train.groups,
        feature_name=list(train.feature_names),
    )
    return model


def replace_directory_atomically(source_dir: Path, final_dir: Path) -> None:
    """Replace a published artifact directory only after a complete build exists."""

    backup_dir = final_dir.with_name(f".{final_dir.name}.previous")
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    if final_dir.exists():
        final_dir.rename(backup_dir)
    try:
        source_dir.rename(final_dir)
    except Exception:
        if backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact inside an unpublished temporary directory."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
