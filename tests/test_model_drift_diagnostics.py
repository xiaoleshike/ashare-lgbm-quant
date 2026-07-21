from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import (
    AppSettings,
    DiagnosticSettings,
    ModelDriftDiagnosticSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.drift_diagnostics import (
    ModelDriftDiagnosticEngine,
    ModelDriftDiagnosticResult,
)
from ashare_quant.models.drift_metrics import (
    build_feature_response_drift,
    build_score_drift,
    fit_psi_edges,
    ks_statistic,
    population_stability_index,
)
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.utils.manifest import atomic_write_json

MODEL_ID = "drift_fixture"
EVALUATION_START = "20210104"
EVALUATION_END = "20210202"


def test_psi_and_ks_detect_distribution_shift_and_missingness() -> None:
    reference = pd.Series([0.0, 1.0, 2.0, 3.0, np.nan])
    identical = reference.copy()
    shifted = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    edges = fit_psi_edges(reference, bins=4)

    assert population_stability_index(reference, identical, edges) == pytest.approx(0.0)
    assert population_stability_index(reference, shifted, edges) > 0.25
    assert ks_statistic(reference, identical) == pytest.approx(0.0)
    assert ks_statistic(reference, shifted) == pytest.approx(1.0)


def test_score_drift_reports_concentration_breadth_and_months() -> None:
    frame = prediction_frame()

    result, reference_months = build_score_drift(frame, reference_months=1, psi_bins=5)

    assert reference_months == ("202101",)
    assert result["month"].tolist() == ["202101", "202102"]
    assert result["top1_concentration"].between(0, 1).all()
    assert result["top10_concentration"].between(0, 1).all()
    assert result["normalized_breadth"].between(0, 1).all()
    assert result.loc[result["month"] == "202102", "score_psi"].iloc[0] > 0


def test_feature_response_reports_rank_ic_buckets_and_sign_change() -> None:
    reference = response_frame("20200102", positive=True)
    evaluation = response_frame("20210104", positive=False)
    evaluation["month"] = "202101"
    reference["month"] = "202001"

    result = build_feature_response_drift(
        reference,
        evaluation,
        ("f1",),
        bucket_counts=(5, 10),
        minimum_cross_section=3,
    )

    assert set(result["bucket_count"]) == {5, 10}
    assert result["ic_sign_change"].all()
    assert result["rank_ic"].dropna().iloc[0] == pytest.approx(-1.0)
    assert result["reference_rank_ic"].dropna().iloc[0] == pytest.approx(1.0)
    assert result.loc[result["bucket_count"] == 10, "bucket"].nunique() == 10


def test_engine_publishes_required_read_only_outputs_and_manifest(tmp_path: Path) -> None:
    engine, paths = drift_fixture(tmp_path)
    source_bytes = {path: path.read_bytes() for path in paths}

    result = engine.run(
        model_id=MODEL_ID,
        start_date=EVALUATION_START,
        end_date=EVALUATION_END,
    )

    assert result.model_id == MODEL_ID
    assert result.feature_count == 2
    assert result.months == 2
    assert {path.name for path in result.output_dir.iterdir()} == {
        "feature_drift.parquet",
        "score_drift.parquet",
        "feature_response.parquet",
        "summary.json",
        "diagnostics_report.md",
        "manifest.json",
    }
    feature_drift = pd.read_parquet(result.output_dir / "feature_drift.parquet")
    score_drift = pd.read_parquet(result.output_dir / "score_drift.parquet")
    response = pd.read_parquet(result.output_dir / "feature_response.parquet")
    assert set(feature_drift["feature"]) == {"f1", "f2"}
    assert set(score_drift["month"]) == {"202101", "202102"}
    assert set(response["bucket_count"]) == {5, 10}
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == MODEL_ID
    assert manifest["feature_hash"] == feature_list_hash(("f1", "f2"))
    assert manifest["leakage_contract"]["model_fitted"] is False
    assert manifest["leakage_contract"]["labels_used_only_for_post_hoc_feature_response"] is True
    assert manifest["source_backtest_manifest"]["out_of_sample"] is True
    assert all(path.read_bytes() == source_bytes[path] for path in paths)


def test_labels_cannot_change_feature_or_score_drift(tmp_path: Path) -> None:
    engine, _ = drift_fixture(tmp_path)
    first = engine.run(
        model_id=MODEL_ID,
        start_date=EVALUATION_START,
        end_date=EVALUATION_END,
    )
    feature_before = pd.read_parquet(first.output_dir / "feature_drift.parquet")
    score_before = pd.read_parquet(first.output_dir / "score_drift.parquet")
    response_before = pd.read_parquet(first.output_dir / "feature_response.parquet")
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    evaluation = labels["trade_date"].between(EVALUATION_START, EVALUATION_END)
    labels.loc[evaluation, "future_excess_ret"] *= -1
    labels.to_parquet(label_path, index=False)

    second = engine.run(
        model_id=MODEL_ID,
        start_date=EVALUATION_START,
        end_date=EVALUATION_END,
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(second.output_dir / "feature_drift.parquet"), feature_before
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(second.output_dir / "score_drift.parquet"), score_before
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(
            pd.read_parquet(second.output_dir / "feature_response.parquet"), response_before
        )


def test_non_future_label_is_rejected(tmp_path: Path) -> None:
    engine, _ = drift_fixture(tmp_path)
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    selected = labels["trade_date"] == EVALUATION_START
    labels.loc[selected, "exit_date"] = labels.loc[selected, "trade_date"]
    labels.to_parquet(label_path, index=False)

    with pytest.raises(DataValidationError, match="non-future labels"):
        engine.run(
            model_id=MODEL_ID,
            start_date=EVALUATION_START,
            end_date=EVALUATION_END,
        )


def test_failed_publication_keeps_previous_manifest(tmp_path: Path, monkeypatch) -> None:
    engine, _ = drift_fixture(tmp_path)
    first = engine.run(
        model_id=MODEL_ID,
        start_date=EVALUATION_START,
        end_date=EVALUATION_END,
    )
    manifest_path = first.output_dir / "manifest.json"
    manifest_before = manifest_path.read_bytes()

    def fail_to_parquet(self: pd.DataFrame, path: Path, **kwargs: object) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    with pytest.raises(OSError, match="simulated publication failure"):
        engine.run(
            model_id=MODEL_ID,
            start_date=EVALUATION_START,
            end_date=EVALUATION_END,
        )

    assert manifest_path.read_bytes() == manifest_before


def test_model_drift_cli_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    class SuccessfulEngine:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, **kwargs: str) -> ModelDriftDiagnosticResult:
            return ModelDriftDiagnosticResult("run_fixture", tmp_path / "reports", MODEL_ID, 20, 12)

    monkeypatch.setattr("ashare_quant.cli.ModelDriftDiagnosticEngine", SuccessfulEngine)
    arguments = [
        "--config",
        "config/default.yaml",
        "models",
        "diagnostics",
        "drift",
        "--model-id",
        MODEL_ID,
        "--start-date",
        EVALUATION_START,
        "--end-date",
        EVALUATION_END,
    ]
    assert main(arguments) == 0
    assert "model_drift_diagnostics: run_id=run_fixture" in capsys.readouterr().out

    class FailingEngine(SuccessfulEngine):
        def run(self, **kwargs: str) -> ModelDriftDiagnosticResult:
            raise DataValidationError("missing immutable predictions")

    monkeypatch.setattr("ashare_quant.cli.ModelDriftDiagnosticEngine", FailingEngine)
    assert main(arguments) == 2
    assert "model drift diagnostics failed" in capsys.readouterr().err


def drift_fixture(tmp_path: Path) -> tuple[ModelDriftDiagnosticEngine, tuple[Path, ...]]:
    processed = tmp_path / "processed"
    reports = tmp_path / "reports"
    models = tmp_path / "models"
    feature_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    dates = ("20200102", "20200103", "20210104", "20210105", "20210201", "20210202")
    for date_index, trade_date in enumerate(dates):
        for stock_index in range(10):
            code = f"{stock_index:06d}.SZ"
            f1 = float(stock_index)
            f2 = float(stock_index + (20 if trade_date.startswith("2021") else 0))
            feature_rows.append({"trade_date": trade_date, "ts_code": code, "f1": f1, "f2": f2})
            universe_rows.append(
                {"trade_date": trade_date, "ts_code": code, "in_model_universe": True}
            )
            label_rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": code,
                    "horizon": 5,
                    "exit_date": f"{int(trade_date) + 10:08d}",
                    "future_excess_ret": f1 / 100 + date_index / 1000,
                    "is_label_available": True,
                }
            )
    feature_path = write_partition(processed / "features_daily", pd.DataFrame(feature_rows))
    universe_path = write_partition(processed / "universe_daily", pd.DataFrame(universe_rows))
    label_path = write_partition(processed / "labels_forward", pd.DataFrame(label_rows))
    for name in ("features_daily", "universe_daily", "labels_forward"):
        atomic_write_json(processed / name / "_manifest.json", {"artifact_name": name})

    artifact = write_model_artifact(models)
    registry = ModelRegistry(models)
    registry.register_model(artifact)
    registry.promote_model(MODEL_ID)
    backtest_dir = reports / "backtest" / "fixture_run"
    backtest_dir.mkdir(parents=True)
    predictions = prediction_frame()
    prediction_path = backtest_dir / "predictions.parquet"
    predictions.drop(columns=["score_percentile"]).to_parquet(prediction_path, index=False)
    atomic_write_json(
        backtest_dir / "manifest.json",
        {
            "artifact_name": "historical_champion_backtest",
            "model_id": MODEL_ID,
            "feature_hash": feature_list_hash(("f1", "f2")),
            "requested_start_date": EVALUATION_START,
            "effective_end_date": EVALUATION_END,
            "out_of_sample": True,
        },
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: fixture\n", encoding="utf-8")
    settings = AppSettings(
        diagnostics=DiagnosticSettings(
            model_drift=ModelDriftDiagnosticSettings(
                label_horizon=5,
                psi_bins=5,
                score_reference_months=1,
                reference_sample_rows=100,
                evaluation_sample_rows_per_month=100,
                minimum_daily_cross_section=3,
                response_bucket_counts=(5, 10),
            )
        )
    )
    engine = ModelDriftDiagnosticEngine(
        processed_root=processed,
        models_root=models,
        reports_root=reports,
        settings=settings,
        config_path=config_path,
    )
    return engine, (feature_path, universe_path, label_path, prediction_path)


def prediction_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for date_index, trade_date in enumerate(("20210104", "20210105", "20210201", "20210202")):
        for rank in range(1, 11):
            score = float(11 - rank + (5 if date_index >= 2 else 0))
            records.append(
                {
                    "trade_date": trade_date,
                    "ts_code": f"{rank - 1:06d}.SZ",
                    "prediction_score": score,
                    "rank": rank,
                    "cross_section_size": 10,
                    "score_percentile": 1.0 - (rank - 1) / 10,
                }
            )
    return pd.DataFrame(records)


def response_frame(trade_date: str, *, positive: bool) -> pd.DataFrame:
    factor = np.arange(10, dtype=float)
    target = factor if positive else -factor
    return pd.DataFrame(
        {
            "trade_date": [trade_date] * 10,
            "ts_code": [f"{index:06d}.SZ" for index in range(10)],
            "f1": factor,
            "future_excess_ret": target,
            "exit_date": ["20210301"] * 10,
        }
    )


def write_partition(root: Path, frame: pd.DataFrame) -> Path:
    path = root / "year=fixture" / "data.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    return path


def write_model_artifact(models_root: Path) -> Path:
    artifact = models_root / MODEL_ID
    artifact.mkdir(parents=True)
    digest = feature_list_hash(("f1", "f2"))
    (artifact / "model.txt").write_text("fixture\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json",
        {"features": ["f1", "f2"], "feature_hash": digest},
    )
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.1}, "test": {"rank_ic": 0.1}},
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": MODEL_ID,
            "completed_at": "2021-01-01T00:00:00+00:00",
            "git_commit": "fixture",
            "config_hash": "fixture",
            "feature_list_hash": digest,
            "train_start": "20200101",
            "train_end": "20201231",
        },
    )
    return artifact
