from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import ExplainabilitySettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.research.explainability import ExplainabilityEngine, ExplainabilityResult
from ashare_quant.research.explainability.contributions import (
    ContributionMatrix,
    compute_tree_contributions,
)
from ashare_quant.research.explainability.history import load_same_model_history
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20240110"
MODEL_ID = "ranker_fixture"
FEATURES = ("market_excess_ret_120d", "amihud_20d")


class FakeTreeModel:
    """Deterministic additive model with LightGBM-compatible local contributions."""

    def predict(self, data: pd.DataFrame, *, pred_contrib: bool = False) -> np.ndarray:
        first = data.iloc[:, 0].fillna(0.0).to_numpy(dtype=float)
        second = data.iloc[:, 1].fillna(0.0).to_numpy(dtype=float) * 2.0
        if pred_contrib:
            return np.column_stack((first, second, np.zeros(len(data), dtype=float)))
        return first + second


def test_explanations_preserve_scores_ranks_and_render_contributions(tmp_path: Path) -> None:
    engine, report_dir = explainability_fixture(tmp_path)
    prediction_bytes = (report_dir / "predictions.parquet").read_bytes()
    candidate_bytes = (report_dir / "candidates.csv").read_bytes()

    result = engine.explain(AS_OF)

    assert result.candidate_count == 2
    assert result.method == "fixture_tree_shap"
    payload = json.loads((report_dir / "explanations.json").read_text(encoding="utf-8"))
    assert payload["model_id"] == MODEL_ID
    assert payload["history_sessions"] == 1
    assert [stock["ts_code"] for stock in payload["stocks"]] == ["000002.SZ", "000001.SZ"]
    first = payload["stocks"][0]
    assert first["model_rank"] == 2
    assert first["candidate_rank"] == 1
    assert first["prediction_score"] == 3.0
    assert first["score_percentile"] == pytest.approx(2 / 3)
    assert first["signal_strength"] == "strong"
    assert first["confidence"] == "low"
    assert first["historical_score_percentile"] == 1.0
    descriptions = {item["description"] for item in first["positive_contributions"]}
    assert descriptions == {"长期市场超额收益", "流动性压力"}
    second = payload["stocks"][1]
    assert second["negative_contributions"][0]["feature"] == "amihud_20d"
    markdown = (report_dir / "explanations.md").read_text(encoding="utf-8")
    assert "not causal conclusions" in markdown
    assert "buy/sell instructions" in markdown
    assert (report_dir / "predictions.parquet").read_bytes() == prediction_bytes
    assert (report_dir / "candidates.csv").read_bytes() == candidate_bytes


def test_explanation_output_is_deterministic_and_does_not_read_labels(tmp_path: Path) -> None:
    engine, report_dir = explainability_fixture(tmp_path)
    labels = tmp_path / "processed" / "labels_forward"
    labels.mkdir(parents=True)
    (labels / "unreadable.parquet").write_bytes(b"not parquet")

    engine.explain(AS_OF)
    first_json = (report_dir / "explanations.json").read_bytes()
    first_markdown = (report_dir / "explanations.md").read_bytes()
    engine.explain(AS_OF)

    assert (report_dir / "explanations.json").read_bytes() == first_json
    assert (report_dir / "explanations.md").read_bytes() == first_markdown


def test_score_recomputation_mismatch_fails_without_publication(tmp_path: Path) -> None:
    engine, report_dir = explainability_fixture(tmp_path)
    predictions = pd.read_parquet(report_dir / "predictions.parquet")
    predictions.loc[predictions["ts_code"] == "000002.SZ", "prediction_score"] = 3.5
    predictions.to_parquet(report_dir / "predictions.parquet", index=False)
    candidates = pd.read_csv(report_dir / "candidates.csv")
    candidates.loc[candidates["ts_code"] == "000002.SZ", "prediction_score"] = 3.5
    candidates.to_csv(report_dir / "candidates.csv", index=False)

    with pytest.raises(DataValidationError, match="recomputed model scores differ"):
        engine.explain(AS_OF)

    assert not (report_dir / "explanations.json").exists()
    assert not (report_dir / "explanations.md").exists()


def test_candidate_csv_rounding_within_tolerance_is_accepted(tmp_path: Path) -> None:
    engine, report_dir = explainability_fixture(tmp_path)
    candidates = pd.read_csv(report_dir / "candidates.csv")
    candidates.loc[candidates["ts_code"] == "000002.SZ", "prediction_score"] += 1e-10
    candidates.to_csv(report_dir / "candidates.csv", index=False)

    result = engine.explain(AS_OF)

    assert result.candidate_count == 2


def test_non_additive_contributions_fail(tmp_path: Path) -> None:
    engine, _ = explainability_fixture(tmp_path)
    engine._contribution_provider = lambda model, matrix: ContributionMatrix(
        values=np.zeros((len(matrix), len(matrix.columns))),
        base_values=np.zeros(len(matrix)),
        method="broken",
    )

    with pytest.raises(DataValidationError, match="do not reconstruct"):
        engine.explain(AS_OF)


def test_lightgbm_native_treeshap_is_used_when_shap_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "ashare_quant.research.explainability.contributions._optional_shap_factory",
        lambda: None,
    )
    matrix = pd.DataFrame({"f1": [1.0], "f2": [-0.5]})

    result = compute_tree_contributions(FakeTreeModel(), matrix)

    assert result.method == "lightgbm_pred_contrib"
    np.testing.assert_allclose(result.values, [[1.0, -1.0]])
    np.testing.assert_allclose(result.base_values, [0.0])


def test_candidate_date_and_prediction_feature_hash_are_validated(tmp_path: Path) -> None:
    engine, report_dir = explainability_fixture(tmp_path)
    candidates = pd.read_csv(report_dir / "candidates.csv", dtype={"trade_date": str})
    candidates["trade_date"] = "20240109"
    candidates.to_csv(report_dir / "candidates.csv", index=False)

    with pytest.raises(DataValidationError, match="candidate dates"):
        engine.explain(AS_OF)

    candidates["trade_date"] = AS_OF
    candidates.to_csv(report_dir / "candidates.csv", index=False)
    manifest = json.loads((report_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["feature_hash"] = "wrong"
    atomic_write_json(report_dir / "manifest.json", manifest)
    with pytest.raises(DataValidationError, match="feature_hash"):
        engine.explain(AS_OF)


def test_history_uses_only_prior_same_model_predictions(tmp_path: Path) -> None:
    _, report_dir = explainability_fixture(tmp_path)
    _write_predictions(
        tmp_path / "reports" / "20240109" / "predictions.parquet",
        "20240109",
        "other_model",
        [100.0],
    )
    _write_predictions(
        tmp_path / "reports" / "20240111" / "predictions.parquet",
        "20240111",
        MODEL_ID,
        [200.0],
    )

    scores, sessions = load_same_model_history(
        tmp_path / "reports", as_of=AS_OF, model_id=MODEL_ID, maximum_sessions=252
    )

    assert sessions == 1
    assert sorted(scores.tolist()) == [-1.0, 0.0, 2.0]
    assert report_dir.exists()


def test_research_explain_cli_success_and_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    class SuccessfulEngine:
        def __init__(self, **kwargs: object) -> None:
            pass

        def explain(self, as_of: str) -> ExplainabilityResult:
            return ExplainabilityResult(
                as_of,
                MODEL_ID,
                2,
                "lightgbm_pred_contrib",
                str(tmp_path / "explanations.json"),
                str(tmp_path / "explanations.md"),
            )

    monkeypatch.setattr("ashare_quant.cli.ExplainabilityEngine", SuccessfulEngine)
    arguments = [
        "--config",
        "config/default.yaml",
        "research",
        "--processed-root",
        str(tmp_path / "processed"),
        "--reports-root",
        str(tmp_path / "reports"),
        "--models-root",
        str(tmp_path / "models"),
        "explain",
        "--as-of",
        AS_OF,
    ]

    assert main(arguments) == 0
    assert "research_explanations: date=20240110 candidates=2" in capsys.readouterr().out

    class FailingEngine(SuccessfulEngine):
        def explain(self, as_of: str) -> ExplainabilityResult:
            raise DataValidationError("feature hash mismatch")

    monkeypatch.setattr("ashare_quant.cli.ExplainabilityEngine", FailingEngine)
    assert main(arguments) == 2
    assert "research explanation failed" in capsys.readouterr().err


def explainability_fixture(tmp_path: Path) -> tuple[ExplainabilityEngine, Path]:
    models_root = tmp_path / "models"
    artifact = _write_model_artifact(models_root)
    registry = ModelRegistry(models_root)
    registry.register_model(artifact)
    registry.promote_model(MODEL_ID)
    processed_root = tmp_path / "processed"
    feature_dir = processed_root / "features_daily" / "year=2024" / "month=01"
    feature_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": [AS_OF] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "market_excess_ret_120d": [1.0, 2.0, 4.0],
            "amihud_20d": [-0.5, 0.5, 1.0],
            "unrelated_feature": [999.0] * 3,
        }
    ).to_parquet(feature_dir / "data.parquet", index=False)
    report_dir = tmp_path / "reports" / AS_OF
    report_dir.mkdir(parents=True)
    _write_predictions(report_dir / "predictions.parquet", AS_OF, MODEL_ID, [0.0, 3.0, 6.0])
    pd.DataFrame(
        {
            "rank": [1, 2],
            "ts_code": ["000002.SZ", "000001.SZ"],
            "prediction_score": [3.0, 0.0],
            "selection_reason": ["passed_configured_filters"] * 2,
            "trade_date": [AS_OF] * 2,
            "model_id": [MODEL_ID] * 2,
        }
    ).to_csv(report_dir / "candidates.csv", index=False)
    atomic_write_json(
        report_dir / "manifest.json",
        {"model_id": MODEL_ID, "feature_hash": feature_list_hash(FEATURES)},
    )
    _write_predictions(
        tmp_path / "reports" / "20240108" / "predictions.parquet",
        "20240108",
        MODEL_ID,
        [-1.0, 0.0, 2.0],
    )
    settings = ExplainabilitySettings(
        strong_percentile=0.6,
        moderate_percentile=0.4,
        minimum_history_sessions=2,
        high_confidence_history_sessions=3,
        maximum_history_sessions=10,
    )
    return (
        ExplainabilityEngine(
            registry=registry,
            processed_root=processed_root,
            reports_root=tmp_path / "reports",
            settings=settings,
            model_loader=lambda path: FakeTreeModel(),
            contribution_provider=_fixture_contributions,
        ),
        report_dir,
    )


def _fixture_contributions(model: object, matrix: pd.DataFrame) -> ContributionMatrix:
    first = matrix.iloc[:, 0].fillna(0.0).to_numpy(dtype=float)
    second = matrix.iloc[:, 1].fillna(0.0).to_numpy(dtype=float) * 2.0
    return ContributionMatrix(
        values=np.column_stack((first, second)),
        base_values=np.zeros(len(matrix), dtype=float),
        method="fixture_tree_shap",
    )


def _write_model_artifact(models_root: Path) -> Path:
    artifact = models_root / MODEL_ID
    artifact.mkdir(parents=True)
    digest = feature_list_hash(FEATURES)
    (artifact / "model.txt").write_text("fixture model\n", encoding="utf-8")
    atomic_write_json(
        artifact / "feature_list.json", {"features": list(FEATURES), "feature_hash": digest}
    )
    atomic_write_json(
        artifact / "metrics.json",
        {"validation": {"rank_ic": 0.03}, "test": {"rank_ic": 0.02}},
    )
    atomic_write_json(
        artifact / "manifest.json",
        {
            "artifact_name": "lightgbm_ranker_baseline",
            "experiment_id": MODEL_ID,
            "completed_at": "2024-01-11T00:00:00+00:00",
            "git_commit": "fixture",
            "config_hash": "fixture-config",
            "feature_list_hash": digest,
            "train_start": "20200101",
            "train_end": "20231231",
        },
    )
    return artifact


def _write_predictions(
    path: Path,
    trade_date: str,
    model_id: str,
    scores: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": [trade_date] * len(scores),
            "ts_code": [f"{index + 1:06d}.SZ" for index in range(len(scores))],
            "prediction_score": scores,
            "model_id": [model_id] * len(scores),
        }
    ).to_parquet(path, index=False)
