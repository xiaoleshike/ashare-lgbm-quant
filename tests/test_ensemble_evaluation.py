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
from ashare_quant.models.ensemble_evaluation import (
    EnsembleEvaluationResult,
    MultiHorizonEnsembleEngine,
    build_rank_percentile_ensemble,
)
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json

DATES = ("20240102", "20240103", "20240104")
HORIZONS = (5, 10, 20, 60)


class _FixtureModel:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data["f1"].to_numpy(dtype=float)


def test_percentile_rank_ensemble_is_deterministic_and_does_not_average_raw_scores() -> None:
    base = pd.DataFrame(
        {
            "trade_date": ["20240102"] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
        }
    )
    predictions = {
        5: base.assign(prediction_score=[1.0, 2.0, 3.0]),
        10: base.assign(prediction_score=[100.0, 200.0, 300.0]),
        20: base.assign(prediction_score=[-1.0, 0.0, 1.0]),
        60: base.assign(prediction_score=[0.1, 0.2, 0.3]),
    }

    first = build_rank_percentile_ensemble(predictions)
    shuffled = {
        horizon: frame.sample(frac=1.0, random_state=horizon).reset_index(drop=True)
        for horizon, frame in predictions.items()
    }
    second = build_rank_percentile_ensemble(shuffled)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[first["rank"] == 1, "ensemble_score"].item() == pytest.approx(1.0)
    assert first["ensemble_score"].max() <= 1.0


def test_ensemble_writes_immutable_outputs_and_preserves_registry(tmp_path: Path) -> None:
    engine, sources = _ensemble_fixture(tmp_path)
    before_registry = sources["registry"].read_bytes()

    first = engine.evaluate(sources["model_ids"])
    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir()}
    second = engine.evaluate(tuple(reversed(sources["model_ids"])))

    assert second.output_dir == first.output_dir
    assert {path.name: path.read_bytes() for path in second.output_dir.iterdir()} == before
    assert sources["registry"].read_bytes() == before_registry
    assert set(before) == {
        "ensemble_predictions.parquet",
        "metrics.json",
        "report.md",
        "manifest.json",
    }
    predictions = pd.read_parquet(first.output_dir / "ensemble_predictions.parquet")
    assert len(predictions) == len(DATES) * 60
    assert {
        "trade_date",
        "ts_code",
        "ensemble_score",
        "rank",
        "h5_rank_percentile",
        "h10_rank_percentile",
        "h20_rank_percentile",
        "h60_rank_percentile",
    } == set(predictions.columns)
    metrics = _read_json(first.output_dir / "metrics.json")
    assert set(metrics["overall"]) == {"5", "10", "20", "60"}
    assert set(metrics["overall"]["5"]) == {
        "champion",
        "h5",
        "h10",
        "h20",
        "h60",
        "ensemble",
    }
    assert {row["period_type"] for row in metrics["rows"]} == {
        "overall",
        "year",
        "regime",
    }
    manifest = _read_json(first.output_dir / "manifest.json")
    assert manifest["method"] == "daily_equal_weight_rank_percentile"
    assert manifest["weights"] == {"5": 0.25, "10": 0.25, "20": 0.25, "60": 0.25}
    assert manifest["isolation_contract"]["raw_scores_averaged"] is False
    assert manifest["isolation_contract"]["registry_modified"] is False


def test_ensemble_rejects_missing_horizon(tmp_path: Path) -> None:
    engine, sources = _ensemble_fixture(tmp_path)

    with pytest.raises(DataValidationError, match="four unique"):
        engine.evaluate(sources["model_ids"][:-1])


def test_ensemble_rejects_different_universe_rows(tmp_path: Path) -> None:
    engine, sources = _ensemble_fixture(tmp_path)
    h60_predictions = sources["reports"] / "challenger_predictions" / "challenger_h60"
    frame = pd.read_parquet(h60_predictions / "predictions.parquet").iloc[:-1].copy()
    frame.to_parquet(h60_predictions / "predictions.parquet", index=False)
    manifest = _read_json(h60_predictions / "manifest.json")
    manifest["prediction_rows"] = len(frame)
    atomic_write_json(h60_predictions / "manifest.json", manifest)

    with pytest.raises(DataValidationError, match="different universe"):
        engine.evaluate(sources["model_ids"])


def test_ensemble_loads_labels_only_after_scores_are_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, sources = _ensemble_fixture(tmp_path)
    import ashare_quant.models.ensemble_evaluation as module

    score_was_built = False
    original_build = module.build_rank_percentile_ensemble
    original_labels = module._load_mature_labels

    def build_scores(predictions: Any) -> pd.DataFrame:
        nonlocal score_was_built
        result = original_build(predictions)
        score_was_built = True
        return result

    def load_labels(*args: Any, **kwargs: Any) -> pd.DataFrame:
        assert score_was_built
        return original_labels(*args, **kwargs)

    monkeypatch.setattr(module, "build_rank_percentile_ensemble", build_scores)
    monkeypatch.setattr(module, "_load_mature_labels", load_labels)
    poison = sources["reports"] / "production_observation" / "poison.json"
    poison.parent.mkdir(parents=True)
    poison.write_bytes(b"not-json")

    result = engine.evaluate(sources["model_ids"])

    assert result.prediction_rows == len(DATES) * 60
    assert poison.read_bytes() == b"not-json"


def test_ensemble_cli_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports" / "ensemble_evaluation" / "run"

    monkeypatch.setattr(
        MultiHorizonEnsembleEngine,
        "evaluate",
        lambda self, model_ids: EnsembleEvaluationResult(
            run_id="run",
            model_ids=tuple(model_ids),
            prediction_rows=100,
            prediction_dates=5,
            output_dir=output,
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
        "evaluate-ensemble",
    ]
    arguments = [
        item for model_id in ("h5", "h10", "h20", "h60") for item in ("--model-id", model_id)
    ]

    assert main([*common, *arguments]) == 0
    assert "ensemble_evaluation: run_id=run" in capsys.readouterr().out

    monkeypatch.setattr(
        MultiHorizonEnsembleEngine,
        "evaluate",
        lambda self, model_ids: (_ for _ in ()).throw(DataValidationError("invalid ensemble")),
    )
    assert main([*common, *arguments]) == 2
    assert "ensemble evaluation failed: invalid ensemble" in capsys.readouterr().err


def _ensemble_fixture(
    tmp_path: Path,
) -> tuple[MultiHorizonEnsembleEngine, dict[str, Any]]:
    processed = tmp_path / "processed"
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: ensemble-test\n", encoding="utf-8")
    features_dir = processed / "features_daily"
    universe_dir = processed / "universe_daily"
    labels_dir = processed / "labels_forward"
    for directory in (features_dir, universe_dir, labels_dir):
        directory.mkdir(parents=True)
    feature_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(DATES):
        for stock_index in range(60):
            ts_code = f"{stock_index:06d}.SZ"
            feature_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "f1": float(stock_index + date_index),
                    "f2": float(60 - stock_index + date_index),
                }
            )
            universe_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "in_model_universe": True,
                }
            )
            for horizon in HORIZONS:
                label_rows.append(
                    {
                        "trade_date": trade_date,
                        "ts_code": ts_code,
                        "horizon": horizon,
                        "is_label_available": True,
                        "future_excess_ret": (stock_index - 29.5) / (1000 / horizon),
                        "benchmark_forward_ret": (-0.01, 0.0, 0.01)[date_index],
                    }
                )
    pd.DataFrame(feature_rows).to_parquet(features_dir / "data.parquet", index=False)
    pd.DataFrame(universe_rows).to_parquet(universe_dir / "data.parquet", index=False)
    pd.DataFrame(label_rows).to_parquet(labels_dir / "data.parquet", index=False)
    features_manifest = features_dir / "_manifest.json"
    universe_manifest = universe_dir / "_manifest.json"
    labels_manifest = labels_dir / "_manifest.json"
    atomic_write_json(features_manifest, {"artifact_name": "features_daily"})
    atomic_write_json(universe_manifest, {"artifact_name": "universe_daily"})
    atomic_write_json(labels_manifest, {"artifact_name": "labels_forward"})

    feature_names = ("f1", "f2")
    champion = _write_model(models, "champion", feature_names, "lightgbm_ranker_baseline")
    registry = ModelRegistry(models)
    registry.register_model(champion)
    registry.promote_model("champion")
    horizon_plan = reports / "horizon_experiments" / "fixture" / "experiment_manifest.json"
    atomic_write_json(horizon_plan, {"artifact_name": "multi_horizon_experiment_plan"})
    model_ids: list[str] = []
    for horizon in HORIZONS:
        model_id = f"challenger_h{horizon}"
        model_ids.append(model_id)
        artifact = _write_model(
            models,
            model_id,
            feature_names,
            "lightgbm_ranker_challenger",
            extra={
                "horizon": horizon,
                "holding_period": horizon,
                "execution_rule": "next_open",
                "universe_hash": _hash(universe_manifest),
                "source_manifests": {
                    "horizon_experiment": {
                        "path": str(horizon_plan.resolve()),
                        "sha256": _hash(horizon_plan),
                    }
                },
            },
        )
        registry.register_model(artifact)
        prediction_dir = reports / "challenger_predictions" / model_id
        prediction_dir.mkdir(parents=True)
        predictions = pd.DataFrame(feature_rows)[["trade_date", "ts_code"]].copy()
        multiplier = HORIZONS.index(horizon) + 1
        predictions["prediction_score"] = predictions["ts_code"].str[:6].astype(float) * multiplier
        predictions["model_id"] = model_id
        predictions = predictions.sort_values(
            ["trade_date", "prediction_score", "ts_code"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        predictions["rank"] = predictions.groupby("trade_date", sort=False).cumcount() + 1
        predictions.to_parquet(prediction_dir / "predictions.parquet", index=False)
        atomic_write_json(
            prediction_dir / "manifest.json",
            {
                "artifact_name": "challenger_predictions",
                "prediction_identity": f"prediction-h{horizon}",
                "model_id": model_id,
                "feature_hash": feature_list_hash(feature_names),
                "universe_hash": _hash(universe_manifest),
                "horizon": horizon,
                "holding_period": horizon,
                "execution_rule": "next_open",
                "evaluation_ranges": [{"start_date": DATES[0], "end_date": DATES[-1]}],
                "maximum_mature_evaluation_date": DATES[-1],
                "prediction_rows": len(predictions),
                "prediction_dates": len(DATES),
                "input_manifests": {
                    "model": _hash(artifact / "manifest.json"),
                    "features_daily": _hash(features_manifest),
                    "universe_daily": _hash(universe_manifest),
                },
            },
        )
    settings = AppSettings.model_validate(
        {"models": {"challenger_evaluation": {"minimum_cross_section": 20}}}
    )
    engine = MultiHorizonEnsembleEngine(
        registry=registry,
        processed_root=processed,
        reports_root=reports,
        settings=settings,
        config_path=config,
        model_loader=lambda _path: _FixtureModel(),
    )
    return engine, {
        "reports": reports,
        "registry": models / "registry.json",
        "model_ids": tuple(model_ids),
    }


def _write_model(
    root: Path,
    model_id: str,
    features: tuple[str, ...],
    artifact_name: str,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    artifact = root / model_id
    artifact.mkdir(parents=True)
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("fixture model\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json", {"features": list(features), "feature_hash": digest}
    )
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.01}, "test": {"rank_ic": 0.01}},
    )
    manifest = {
        "artifact_name": artifact_name,
        "experiment_id": model_id,
        "creation_time": "2026-07-22T00:00:00+00:00",
        "feature_list_hash": digest,
        "train_start": "20200101",
        "train_end": "20231231",
    }
    manifest.update(extra or {})
    atomic_write_json(artifact / "manifest.json", manifest)
    return artifact


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
