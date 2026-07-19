from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.inference import InferenceResult, ProductionInferenceEngine
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.orchestration.freshness import GateResult
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20240110"


class FakeModel:
    """Deterministic stand-in for a persisted LightGBM Booster."""

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data.fillna(0.0).sum(axis=1).to_numpy(dtype=float)


class ReadyChecks:
    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready

    def check_all(self, as_of: str) -> tuple[GateResult, ...]:
        failures = () if self.ready else ("fixture is stale",)
        return (GateResult("features_readiness_gate", as_of, failures, (), {}),)


def test_inference_loads_champion_and_publishes_expected_schema(tmp_path: Path) -> None:
    engine, model_path = inference_fixture(tmp_path)
    loaded: list[Path] = []
    engine._model_loader = lambda path: loaded.append(path) or FakeModel()

    result = engine.predict(AS_OF)

    assert loaded == [model_path]
    assert result.model_id == "ranker_a"
    assert result.universe_size == 3
    assert result.prediction_count == 2
    assert list(result.predictions.columns) == [
        "trade_date",
        "ts_code",
        "prediction_score",
        "model_id",
    ]
    assert result.predictions["ts_code"].tolist() == ["000002.SZ", "000001.SZ"]
    assert pd.read_parquet(result.output_dir / "predictions.parquet").equals(result.predictions)
    ranking = pd.read_csv(result.output_dir / "ranking.csv", dtype={"ts_code": str})
    assert ranking["rank"].tolist() == [1, 2]
    assert set(ranking.columns) == {"rank", "ts_code", "prediction_score"}
    summary = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["feature_count"] == 2
    assert summary["universe_size"] == 3
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "ranker_a"
    assert manifest["input_artifact_manifests"]["features_daily"]["artifact_name"] == (
        "features_daily"
    )


def test_missing_champion_fails_before_loading_features(tmp_path: Path) -> None:
    engine = ProductionInferenceEngine(
        registry=ModelRegistry(tmp_path / "models"),
        processed_root=tmp_path / "missing-processed",
        reports_root=tmp_path / "reports",
        config_path=_write_config(tmp_path),
        freshness=ReadyChecks(),
        model_loader=lambda path: FakeModel(),
    )

    with pytest.raises(DataValidationError, match="no champion"):
        engine.predict(AS_OF)


def test_feature_hash_mismatch_fails_fast(tmp_path: Path) -> None:
    engine, _ = inference_fixture(tmp_path)
    artifact = tmp_path / "models" / "ranker_a"
    payload = json.loads((artifact / "feature_list.json").read_text(encoding="utf-8"))
    payload["feature_hash"] = "wrong"
    atomic_write_json(artifact / "feature_list.json", payload)

    with pytest.raises(DataValidationError, match="feature identity mismatch"):
        engine.predict(AS_OF)


def test_missing_model_feature_fails_with_column_name(tmp_path: Path) -> None:
    engine, _ = inference_fixture(tmp_path, features=("f1", "missing_signal"))

    with pytest.raises(DataValidationError, match="missing_signal"):
        engine.predict(AS_OF)


def test_inference_is_deterministic_and_does_not_load_labels(tmp_path: Path) -> None:
    engine, _ = inference_fixture(tmp_path)
    labels = tmp_path / "processed" / "labels_forward"
    labels.mkdir(parents=True)
    (labels / "unreadable.parquet").write_bytes(b"not parquet")

    first = engine.predict(AS_OF).predictions
    second = engine.predict(AS_OF).predictions

    pd.testing.assert_frame_equal(first, second)


def test_readiness_failure_stops_prediction(tmp_path: Path) -> None:
    engine, _ = inference_fixture(tmp_path)
    engine.freshness = ReadyChecks(ready=False)

    with pytest.raises(DataValidationError, match="production readiness failed"):
        engine.predict(AS_OF)


def test_feature_and_universe_key_mismatch_fails(tmp_path: Path) -> None:
    engine, _ = inference_fixture(tmp_path)
    universe_path = next((tmp_path / "processed" / "universe_daily").glob("**/*.parquet"))
    universe = pd.read_parquet(universe_path).iloc[:-1]
    universe.to_parquet(universe_path, index=False)

    with pytest.raises(DataValidationError, match="do not match universe rows"):
        engine.predict(AS_OF)


def test_predict_cli_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    class SuccessfulEngine:
        def __init__(self, **kwargs: object) -> None:
            pass

        def predict(self, as_of: str) -> InferenceResult:
            output = tmp_path / "reports" / as_of
            return InferenceResult(as_of, "ranker_a", 2, 3, 2, output, pd.DataFrame())

    monkeypatch.setattr("ashare_quant.cli.ProductionInferenceEngine", SuccessfulEngine)
    arguments = [
        "--config",
        "config/default.yaml",
        "models",
        "--processed-root",
        str(tmp_path / "processed"),
        "--storage-root",
        str(tmp_path / "raw"),
        "--output-root",
        str(tmp_path / "models"),
        "--reports-root",
        str(tmp_path / "reports"),
        "predict",
        "--as-of",
        AS_OF,
    ]

    assert main(arguments) == 0
    assert "prediction_output: date=20240110 model_id=ranker_a stocks=2" in capsys.readouterr().out

    class FailingEngine(SuccessfulEngine):
        def predict(self, as_of: str) -> InferenceResult:
            raise DataValidationError("no champion")

    monkeypatch.setattr("ashare_quant.cli.ProductionInferenceEngine", FailingEngine)
    assert main(arguments) == 2
    assert "production prediction failed: no champion" in capsys.readouterr().err


def inference_fixture(
    tmp_path: Path, *, features: tuple[str, ...] = ("f1", "f2")
) -> tuple[ProductionInferenceEngine, Path]:
    config_path = _write_config(tmp_path)
    processed = tmp_path / "processed"
    feature_dir = processed / "features_daily" / "year=2024" / "month=01"
    universe_dir = processed / "universe_daily" / "year=2024" / "month=01"
    feature_dir.mkdir(parents=True)
    universe_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "f1": [1.0, 3.0, 10.0],
            "f2": [0.5, 1.0, 10.0],
        }
    ).to_parquet(feature_dir / "data.parquet", index=False)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "in_model_universe": [True, True, False],
        }
    ).to_parquet(universe_dir / "data.parquet", index=False)
    atomic_write_json(
        processed / "features_daily" / "_manifest.json",
        {"artifact_name": "features_daily", "max_date": AS_OF},
    )
    atomic_write_json(
        processed / "universe_daily" / "_manifest.json",
        {"artifact_name": "universe_daily", "max_date": AS_OF},
    )

    models_root = tmp_path / "models"
    artifact = _write_model_artifact(models_root, "ranker_a", features)
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    registry.promote_model("ranker_a")
    engine = ProductionInferenceEngine(
        registry=registry,
        processed_root=processed,
        reports_root=tmp_path / "reports",
        config_path=config_path,
        freshness=ReadyChecks(),
        model_loader=lambda path: FakeModel(),
    )
    return engine, artifact / "model.txt"


def _write_model_artifact(models_root: Path, model_id: str, features: tuple[str, ...]) -> Path:
    artifact = models_root / model_id
    artifact.mkdir(parents=True)
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("fixture model\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json",
        {"features": list(features), "feature_hash": digest},
    )
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.03}, "test": {"rank_ic": 0.02}},
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": model_id,
            "completed_at": "2024-01-11T00:00:00+00:00",
            "git_commit": "fixture",
            "config_hash": "fixture-config",
            "feature_list_hash": digest,
            "train_start": "20200101",
            "train_end": "20231231",
        },
    )
    return artifact


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text("project_name: inference-test\n", encoding="utf-8")
    return path
