from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger_evaluation import (
    ChallengerEvaluationEngine,
    ChallengerEvaluationResult,
)
from ashare_quant.models.challenger_prediction import (
    ChallengerPredictionEngine,
    ChallengerPredictionResult,
)
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json

DATES = ("20240102", "20240103", "20240104")


class _FixtureModel:
    def __init__(self, *, challenger: bool) -> None:
        self.challenger = challenger

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        column = "f2" if self.challenger and "f2" in data else data.columns[0]
        return data[column].to_numpy(dtype=float)


def test_challenger_prediction_is_immutable_and_never_loads_labels(tmp_path: Path) -> None:
    engines, paths = _evaluation_fixture(tmp_path)
    observation = paths["reports"] / "production_observation" / "poison.json"
    observation.parent.mkdir(parents=True)
    observation.write_bytes(b"not json")
    label_data = paths["label_data"]
    hidden_labels = label_data.with_suffix(".hidden")
    label_data.rename(hidden_labels)

    first = engines["prediction"].predict("challenger_h5")
    before = (first.output_dir / "manifest.json").read_bytes()
    second = engines["prediction"].predict("challenger_h5")

    assert first.prediction_rows == second.prediction_rows == 60
    assert first.prediction_dates == 3
    assert set(first.predictions.columns) == {
        "trade_date",
        "ts_code",
        "prediction_score",
        "model_id",
        "rank",
    }
    assert (first.output_dir / "manifest.json").read_bytes() == before
    manifest = json.loads(before)
    assert manifest["isolation_contract"]["labels_loaded"] is False
    assert manifest["isolation_contract"]["production_observation_loaded"] is False


def test_evaluation_compares_same_dates_rows_labels_and_preserves_registry(
    tmp_path: Path,
) -> None:
    engines, paths = _evaluation_fixture(tmp_path)
    engines["prediction"].predict("challenger_h5")
    registry_before = paths["registry"].read_bytes()

    result = engines["evaluation"].evaluate("challenger_h5")

    assert result.evaluation_dates == 3
    assert result.labelled_rows == 60
    assert paths["registry"].read_bytes() == registry_before
    assert set(path.name for path in result.output_dir.iterdir()) == {
        "summary.json",
        "evaluation_report.md",
        "metrics.csv",
        "manifest.json",
    }
    summary = _read_json(result.output_dir / "summary.json")
    assert summary["scientific_scope"]["same_dates"] is True
    assert summary["scientific_scope"]["same_universe_rows"] is True
    assert summary["scientific_scope"]["labels_used_only_post_hoc"] is True
    assert summary["scientific_scope"]["registry_modified"] is False
    assert summary["promotion_gate"]["automatic_promotion"] is False
    metrics = pd.read_csv(result.output_dir / "metrics.csv")
    assert set(metrics["model_role"]) == {"champion", "challenger"}
    assert {"overall", "year", "month", "regime"} <= set(metrics["period_type"])
    assert {
        "rank_ic",
        "icir",
        "positive_ic_ratio",
        "top_1pct_mean_excess_return",
        "top_5pct_mean_excess_return",
        "top_10pct_mean_excess_return",
        "top_20pct_mean_excess_return",
        "top_50pct_mean_excess_return",
    } <= set(metrics.columns)


def test_evaluation_rejects_different_feature_hashes(tmp_path: Path) -> None:
    engines, _ = _evaluation_fixture(tmp_path, challenger_features=("f2",))
    engines["prediction"].predict("challenger_h5")

    with pytest.raises(DataValidationError, match="feature hashes differ"):
        engines["evaluation"].evaluate("challenger_h5")


def test_evaluation_rejects_changed_universe_manifest(tmp_path: Path) -> None:
    engines, paths = _evaluation_fixture(tmp_path)
    engines["prediction"].predict("challenger_h5")
    atomic_write_json(paths["universe_manifest"], {"artifact_name": "changed"})

    with pytest.raises(DataValidationError, match="universe hash differs"):
        engines["evaluation"].evaluate("challenger_h5")


def test_evaluation_is_byte_deterministic_and_read_only(tmp_path: Path) -> None:
    engines, paths = _evaluation_fixture(tmp_path)
    engines["prediction"].predict("challenger_h5")
    first = engines["evaluation"].evaluate("challenger_h5")
    before = {
        name: (first.output_dir / name).read_bytes()
        for name in ("summary.json", "evaluation_report.md", "metrics.csv", "manifest.json")
    }

    second = engines["evaluation"].evaluate("challenger_h5")

    assert second.output_dir == first.output_dir
    assert {name: (second.output_dir / name).read_bytes() for name in before} == before
    assert not (paths["reports"] / "production_observation").exists()


def test_prediction_rejects_unmatured_final_test_range(tmp_path: Path) -> None:
    engines, paths = _evaluation_fixture(tmp_path)
    plan = _read_json(paths["horizon_plan"])
    plan["experiments"][0]["maximum_mature_evaluation_date"] = "20240103"
    atomic_write_json(paths["horizon_plan"], plan)
    challenger_manifest = _read_json(paths["challenger_manifest"])
    challenger_manifest["source_manifests"]["horizon_experiment"]["sha256"] = _hash(
        paths["horizon_plan"]
    )
    atomic_write_json(paths["challenger_manifest"], challenger_manifest)

    with pytest.raises(DataValidationError, match="not mature"):
        engines["prediction"].predict("challenger_h5")


def test_challenger_prediction_and_evaluation_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prediction_output = tmp_path / "reports" / "challenger_predictions" / "candidate"
    evaluation_output = tmp_path / "reports" / "challenger_evaluation" / "run"

    monkeypatch.setattr(
        ChallengerPredictionEngine,
        "predict",
        lambda self, model_id: ChallengerPredictionResult(
            model_id=model_id,
            horizon=5,
            prediction_rows=100,
            prediction_dates=5,
            output_dir=prediction_output,
            predictions=pd.DataFrame(),
        ),
    )
    monkeypatch.setattr(
        ChallengerEvaluationEngine,
        "evaluate",
        lambda self, model_id: ChallengerEvaluationResult(
            run_id="run",
            champion_model_id="champion",
            challenger_model_id=model_id,
            horizon=5,
            labelled_rows=90,
            evaluation_dates=5,
            eligible_for_manual_review=True,
            output_dir=evaluation_output,
        ),
    )
    common = [
        "--config",
        "config/default.yaml",
        "models",
        "--processed-root",
        str(tmp_path / "processed"),
        "--output-root",
        str(tmp_path / "models"),
        "--reports-root",
        str(tmp_path / "reports"),
    ]

    assert main([*common, "predict-challenger", "--model-id", "candidate"]) == 0
    assert "challenger_predictions: model_id=candidate" in capsys.readouterr().out
    assert main([*common, "evaluate-challenger", "--model-id", "candidate"]) == 0
    assert "challenger_evaluation: run_id=run" in capsys.readouterr().out


def _evaluation_fixture(
    tmp_path: Path,
    *,
    challenger_features: tuple[str, ...] = ("f1", "f2"),
) -> tuple[dict[str, Any], dict[str, Path]]:
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: challenger-evaluation-test\n", encoding="utf-8")
    feature_dir = processed / "features_daily"
    universe_dir = processed / "universe_daily"
    label_dir = processed / "labels_forward"
    for directory in (feature_dir, universe_dir, label_dir):
        directory.mkdir(parents=True)
    feature_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    benchmark = (0.01, -0.01, 0.0)
    for date_index, trade_date in enumerate(DATES):
        for stock_index in range(20):
            code = f"{stock_index:06d}.SZ"
            feature_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "f1": float(-stock_index + date_index),
                    "f2": float(stock_index + date_index),
                }
            )
            universe_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "in_model_universe": True,
                }
            )
            label_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "horizon": 5,
                    "is_label_available": True,
                    "future_excess_ret": (stock_index - 9.5) / 100,
                    "benchmark_forward_ret": benchmark[date_index],
                }
            )
    pd.DataFrame(feature_rows).to_parquet(feature_dir / "data.parquet", index=False)
    pd.DataFrame(universe_rows).to_parquet(universe_dir / "data.parquet", index=False)
    label_data = label_dir / "data.parquet"
    pd.DataFrame(label_rows).to_parquet(label_data, index=False)
    features_manifest = feature_dir / "_manifest.json"
    universe_manifest = universe_dir / "_manifest.json"
    labels_manifest = label_dir / "_manifest.json"
    atomic_write_json(features_manifest, {"artifact_name": "features_daily"})
    atomic_write_json(universe_manifest, {"artifact_name": "universe_daily"})
    atomic_write_json(labels_manifest, {"artifact_name": "labels_forward", "horizons": [5]})

    champion_features = ("f1", "f2")
    champion_artifact = _write_model(
        models,
        "champion",
        champion_features,
        artifact_name="lightgbm_ranker_baseline",
        test_metrics=True,
    )
    horizon_plan = reports / "horizon_experiments" / "fixture" / "experiment_manifest.json"
    horizon_plan.parent.mkdir(parents=True)
    challenger_hash = feature_list_hash(challenger_features)
    horizon_record = {
        "experiment_id": "h5_fixture",
        "name": "h5",
        "horizon": 5,
        "holding_period": 5,
        "execution_rule": "next_open",
        "label_name": "future_excess_ret_5d",
        "feature_hash": challenger_hash,
        "universe_hash": _hash(universe_manifest),
        "maximum_mature_evaluation_date": DATES[-1],
        "final_test_period": {
            "start_date": DATES[0],
            "end_date": DATES[-1],
            "may_select_model": False,
            "folds": [
                {
                    "fold_id": "test_fold",
                    "evaluation_start": DATES[0],
                    "evaluation_end": DATES[-1],
                }
            ],
        },
    }
    atomic_write_json(horizon_plan, {"experiments": [horizon_record]})
    challenger_artifact = _write_model(
        models,
        "challenger_h5",
        challenger_features,
        artifact_name="lightgbm_ranker_challenger",
        test_metrics=False,
        extra_manifest={
            "source_horizon_experiment_id": "h5_fixture",
            "horizon": 5,
            "holding_period": 5,
            "execution_rule": "next_open",
            "label_name": "future_excess_ret_5d",
            "feature_hash": challenger_hash,
            "universe_hash": _hash(universe_manifest),
            "source_manifests": {
                "horizon_experiment": {
                    "path": str(horizon_plan.resolve()),
                    "sha256": _hash(horizon_plan),
                },
                "features_daily": {
                    "path": str(features_manifest.resolve()),
                    "sha256": _hash(features_manifest),
                },
                "universe_daily": {
                    "path": str(universe_manifest.resolve()),
                    "sha256": _hash(universe_manifest),
                },
            },
        },
    )
    registry = ModelRegistry(models)
    registry.register_model(champion_artifact)
    registry.promote_model("champion")
    registry.register_model(challenger_artifact)

    def model_loader(path: Path) -> _FixtureModel:
        return _FixtureModel(challenger="challenger" in str(path))

    settings = AppSettings.model_validate(
        {
            "models": {
                "challenger_evaluation": {
                    "minimum_cross_section": 5,
                    "minimum_labelled_days": 2,
                }
            }
        }
    )
    prediction = ChallengerPredictionEngine(
        registry=registry,
        processed_root=processed,
        reports_root=reports,
        config_path=config,
        model_loader=model_loader,
    )
    evaluation = ChallengerEvaluationEngine(
        registry=registry,
        processed_root=processed,
        reports_root=reports,
        settings=settings,
        config_path=config,
        model_loader=model_loader,
    )
    return (
        {"prediction": prediction, "evaluation": evaluation},
        {
            "reports": reports,
            "registry": models / "registry.json",
            "features_manifest": features_manifest,
            "universe_manifest": universe_manifest,
            "labels_manifest": labels_manifest,
            "label_data": label_data,
            "horizon_plan": horizon_plan,
            "challenger_manifest": challenger_artifact / "manifest.json",
        },
    )


def _write_model(
    models: Path,
    model_id: str,
    features: tuple[str, ...],
    *,
    artifact_name: str,
    test_metrics: bool,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    artifact = models / model_id
    artifact.mkdir(parents=True)
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("fixture model\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json",
        {"features": list(features), "feature_hash": digest},
    )
    metrics: dict[str, object] = {"validation": {"rank_ic": 0.01}, "test": {}}
    if test_metrics:
        metrics["test"] = {"rank_ic": 0.01}
    atomic_write_json(artifact / "metrics.json", metrics)
    manifest: dict[str, Any] = {
        "artifact_name": artifact_name,
        "experiment_id": model_id,
        "creation_time": "2026-07-21T00:00:00+00:00",
        "feature_list_hash": digest,
        "train_start": "20200101",
        "train_end": "20231231",
    }
    manifest.update(extra_manifest or {})
    atomic_write_json(artifact / "manifest.json", manifest)
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
