from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.config.settings import AppSettings, RankerSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.features.registry import FEATURE_REGISTRY
from ashare_quant.models.feature_lists import (
    feature_list_hash,
    load_recommended_features,
    load_robust_features,
)
from ashare_quant.models.production import ProductionRankerTrainer
from ashare_quant.models.ranker import RankerBaselineRunner
from ashare_quant.models.ranker_data import RankerDataLoader, RankerDataset
from ashare_quant.models.ranker_metrics import evaluate_ranker, ndcg


def test_recommended_feature_list_requires_exact_top_50(tmp_path: Path) -> None:
    path = tmp_path / "recommended_features.json"
    path.write_text(
        json.dumps(
            {
                "recommended_set": "top_50",
                "recommended_features": [spec.name for spec in FEATURE_REGISTRY[:49]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="exactly 50"):
        load_recommended_features(path)


def test_robust_feature_list_rejects_disabled_or_unknown_feature(tmp_path: Path) -> None:
    path = tmp_path / "robust.json"
    path.write_text(json.dumps({"features": ["future_return_leak"]}), encoding="utf-8")

    with pytest.raises(DataValidationError, match="disabled or unknown"):
        load_robust_features(path)


def test_feature_list_hash_is_order_sensitive() -> None:
    assert feature_list_hash(("ret_1d", "ret_3d")) != feature_list_hash(("ret_3d", "ret_1d"))


def test_ranker_loader_builds_contiguous_date_groups_and_relevance(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    write_ranker_fixture(processed, ("ret_1d", "ret_3d"))
    loader = RankerDataLoader(processed, horizon=5, minimum_group_size=3)

    dataset = loader.load("20200102", "20200103", ("ret_1d", "ret_3d"), 5)

    assert dataset.groups == [20, 20]
    assert dataset.relevance.min() == 0
    assert dataset.relevance.max() == 4
    assert dataset.features.dtypes.tolist() == [np.dtype("float32"), np.dtype("float32")]
    assert dataset.frame["trade_date"].is_monotonic_increasing


def test_ndcg_and_ranker_metrics_are_perfect_for_perfect_ordering() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": ["20200102"] * 5,
            "ts_code": [f"{index:06d}.SZ" for index in range(5)],
            "signal": np.arange(5, dtype=np.float32),
            "future_excess_ret_5d": np.arange(5, dtype=float) / 100.0,
            "relevance": np.arange(5, dtype=np.int32),
        }
    )
    dataset = RankerDataset(frame, ("signal",))
    predictions = np.arange(5, dtype=float)

    metrics = evaluate_ranker(dataset, predictions, (10, 50), (0.05, 0.10))

    assert ndcg(np.arange(5), predictions, 10) == pytest.approx(1.0)
    assert metrics["rank_ic"] == pytest.approx(1.0)
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
    assert metrics["ndcg_at_50"] == pytest.approx(1.0)
    assert metrics["top_10pct_mean_future_excess_ret"] == pytest.approx(0.04)


def test_ranker_runner_writes_two_complete_experiments(tmp_path: Path) -> None:
    all_features = tuple(spec.name for spec in FEATURE_REGISTRY[:50])
    robust_features = all_features[:5]
    processed = tmp_path / "processed"
    output = tmp_path / "models"
    recommended_path = tmp_path / "recommended_features.json"
    robust_path = tmp_path / "robust_features.json"
    config_path = tmp_path / "config.yaml"
    write_ranker_fixture(processed, all_features)
    recommended_path.write_text(
        json.dumps({"recommended_set": "top_50", "recommended_features": list(all_features)}),
        encoding="utf-8",
    )
    robust_path.write_text(
        json.dumps({"name": "test_robust", "features": list(robust_features)}),
        encoding="utf-8",
    )
    config_path.write_text("project_name: ranker-test\n", encoding="utf-8")
    settings = AppSettings(
        ranker=RankerSettings(
            train_start="20200102",
            train_end="20200103",
            validation_start="20200104",
            validation_end="20200105",
            test_start="20200106",
            test_end="20200107",
            recommended_features_path=recommended_path,
            robust_features_path=robust_path,
            n_estimators=3,
            num_leaves=3,
            min_child_samples=2,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            minimum_group_size=3,
        )
    )

    results = RankerBaselineRunner(processed, output, settings, config_path).run()

    assert [result.feature_count for result in results] == [50, 5]
    for result in results:
        assert result.output_dir.exists()
        assert {path.name for path in result.output_dir.iterdir()} == {
            "model.txt",
            "feature_list.json",
            "metrics.json",
            "manifest.json",
        }
        metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert metrics["metric_scope"].endswith("not a backtest")
        assert set(metrics) == {"metric_scope", "validation", "test", "feature_importance"}
        assert manifest["target"] == "future_excess_ret_5d"
        assert manifest["train_start"] == "20200102"
        assert manifest["train_end"] == "20200103"
        assert manifest["feature_list_hash"]


def test_production_trainer_uses_all_configured_dates_and_frozen_features(tmp_path: Path) -> None:
    all_features = tuple(spec.name for spec in FEATURE_REGISTRY[:8])
    robust_features = all_features[:4]
    processed = tmp_path / "processed"
    output = tmp_path / "models"
    robust_path = tmp_path / "robust_features.json"
    config_path = tmp_path / "config.yaml"
    write_ranker_fixture(processed, all_features)
    robust_path.write_text(
        json.dumps({"name": "production_robust", "features": list(robust_features)}),
        encoding="utf-8",
    )
    config_path.write_text("project_name: production-test\n", encoding="utf-8")
    settings = AppSettings(
        ranker=RankerSettings(
            n_estimators=3,
            num_leaves=3,
            min_child_samples=2,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            minimum_group_size=3,
        ),
        production_model={
            "train_start": "20200102",
            "train_end": "20200107",
            "feature_list_path": robust_path,
        },
    )

    result = ProductionRankerTrainer(processed, output, settings, config_path).train()

    assert result.output_dir == output / "production"
    assert result.train_start == "20200102"
    assert result.train_end == "20200107"
    assert result.train_groups == 6
    feature_payload = json.loads((result.output_dir / "feature_list.json").read_text("utf-8"))
    metrics = json.loads((result.output_dir / "metrics.json").read_text("utf-8"))
    manifest = json.loads((result.output_dir / "manifest.json").read_text("utf-8"))
    assert tuple(feature_payload["features"]) == robust_features
    assert feature_payload["feature_hash"] == feature_list_hash(robust_features)
    assert metrics["train_min_date"] == "20200102"
    assert metrics["train_max_date"] == "20200107"
    assert metrics["unique_train_dates"] == 6
    assert "validation" not in metrics
    assert "test" not in metrics
    assert manifest["training_start"] == "20200102"
    assert manifest["training_end"] == "20200107"
    assert "validation_start" not in manifest
    assert "test_start" not in manifest


def test_production_training_replaces_artifacts_reproducibly(tmp_path: Path) -> None:
    all_features = tuple(spec.name for spec in FEATURE_REGISTRY[:6])
    robust_features = all_features[:3]
    processed = tmp_path / "processed"
    output = tmp_path / "models"
    robust_path = tmp_path / "robust_features.json"
    config_path = tmp_path / "config.yaml"
    write_ranker_fixture(processed, all_features)
    robust_path.write_text(
        json.dumps({"name": "production_robust", "features": list(robust_features)}),
        encoding="utf-8",
    )
    config_path.write_text("project_name: production-test\n", encoding="utf-8")
    settings = AppSettings(
        ranker=RankerSettings(
            n_estimators=3,
            num_leaves=3,
            min_child_samples=2,
            feature_fraction=1.0,
            bagging_fraction=1.0,
            minimum_group_size=3,
        ),
        production_model={
            "train_start": "20200102",
            "train_end": "20200107",
            "feature_list_path": robust_path,
        },
    )
    trainer = ProductionRankerTrainer(processed, output, settings, config_path)

    first = trainer.train()
    first_feature_list = (first.output_dir / "feature_list.json").read_text("utf-8")
    first_metrics = json.loads((first.output_dir / "metrics.json").read_text("utf-8"))
    second = trainer.train()
    second_feature_list = (second.output_dir / "feature_list.json").read_text("utf-8")
    second_metrics = json.loads((second.output_dir / "metrics.json").read_text("utf-8"))

    assert first.output_dir == second.output_dir == output / "production"
    assert first_feature_list == second_feature_list
    assert first_metrics["train_rows"] == second_metrics["train_rows"]
    assert first_metrics["train_groups"] == second_metrics["train_groups"]
    assert first_metrics["feature_importance"] == second_metrics["feature_importance"]
    assert (output / "production" / "model.txt").exists()


def write_ranker_fixture(processed: Path, feature_names: tuple[str, ...]) -> None:
    dates = [f"202001{day:02d}" for day in range(2, 8)]
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    universe: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(dates):
        for stock_index in range(20):
            code = f"{stock_index:06d}.SZ"
            row: dict[str, object] = {"trade_date": trade_date, "ts_code": code}
            for feature_index, feature in enumerate(feature_names):
                row[feature] = (
                    stock_index * (feature_index % 5 + 1)
                    + date_index
                    + np.sin(stock_index + feature_index)
                )
            features.append(row)
            labels.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "horizon": 5,
                    "future_excess_ret": (stock_index - 9.5) / 100.0,
                    "is_label_available": True,
                }
            )
            universe.append({"trade_date": trade_date, "ts_code": code, "in_model_universe": True})
    write_parquet(processed / "features_daily" / "data.parquet", pd.DataFrame(features))
    write_parquet(processed / "labels_forward" / "data.parquet", pd.DataFrame(labels))
    write_parquet(processed / "universe_daily" / "data.parquet", pd.DataFrame(universe))


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
