"""Governed retrained-Challenger shadow and observation integration tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, PathSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import RegisteredModel
from ashare_quant.models.shadow.service import PREDICTION_COLUMNS
from ashare_quant.models.shadow.storage import (
    logical_prediction_hash,
    publish_shadow_bundle,
    read_complete_manifest,
)
from ashare_quant.monitoring.performance.aggregation import aggregate_performance
from ashare_quant.monitoring.performance_observation.validation import load_shadow_sources
from ashare_quant.retraining.shadow.schemas import (
    RetrainedModelLineage,
    RetrainedShadowContext,
    RetrainedShadowResult,
)
from ashare_quant.retraining.shadow.service import RetrainedChallengerShadowService

AS_OF = "20260729"
MODEL_ID = "challenger_refresh_h10_fixture"


def test_retrained_shadow_preserves_lineage_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, context = shadow_service(tmp_path, monkeypatch)

    first = service.predict(MODEL_ID, as_of=AS_OF)
    second = service.predict(MODEL_ID, as_of=AS_OF)

    assert first.idempotent is False
    assert second.idempotent is True
    manifest = read_complete_manifest(first.output_dir)
    assert manifest is not None
    assert manifest["model_origin"] == "retrained_challenger"
    assert manifest["training_request_id"] == "request-1"
    assert manifest["training_run_id"] == "training-1"
    assert manifest["validation_run_id"] == "validation-1"
    rows = pd.read_parquet(first.output_dir / "predictions.parquet")
    assert set(rows["model_origin"]) == {"retrained_challenger"}
    assert set(rows["production_run_id"]) == {"production-1"}
    assert set(rows["training_run_id"]) == {"training-1"}
    assert set(rows["validation_run_id"]) == {"validation-1"}
    assert set(rows["prediction_hash"]) == {manifest["prediction_hash"]}
    assert context.lineage.parent_model_id == "parent-champion"


def test_retrained_shadow_rejects_different_identity_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, context = shadow_service(tmp_path, monkeypatch)
    result = service.predict(MODEL_ID, as_of=AS_OF)
    before = (result.output_dir / "manifest.json").read_bytes()
    changed = replace(context, validation_manifest_hash="changed-validation-hash")
    monkeypatch.setattr(
        "ashare_quant.retraining.shadow.service.validate_retrained_shadow_eligibility",
        lambda **kwargs: (changed, champion_keys()),
    )

    with pytest.raises(DataValidationError, match="cannot overwrite"):
        service.predict(MODEL_ID, as_of=AS_OF)

    assert (result.output_dir / "manifest.json").read_bytes() == before


def test_shadow_not_eligible_stops_before_scoring_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = shadow_service(tmp_path, monkeypatch)

    def rejected(**kwargs: object) -> None:
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: artifact hash mismatch")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("scoring or forbidden state mutation was called")

    monkeypatch.setattr(
        "ashare_quant.retraining.shadow.service.validate_retrained_shadow_eligibility",
        rejected,
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.shadow.service.score_challenger",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.models.registry.ModelRegistry.promote_model",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.paper_trading.service.PaperTradingService.execute",
        forbidden,
    )

    with pytest.raises(DataValidationError, match="SHADOW_NOT_ELIGIBLE"):
        service.predict(MODEL_ID, as_of=AS_OF)

    assert not service._output_dir(AS_OF, MODEL_ID).exists()


def test_monitoring_keeps_retrained_origin_separate() -> None:
    rows = pd.DataFrame(
        [
            observation_row("same-id", "champion", 0.2, "000001.SZ"),
            observation_row("same-id", "champion", -0.1, "000002.SZ"),
            observation_row("same-id", "retrained_challenger", 0.3, "000003.SZ"),
            observation_row("same-id", "retrained_challenger", -0.2, "000004.SZ"),
        ]
    )
    lineage = {
        "same-id": {
            "source_models": [],
            "fusion_method": None,
        }
    }

    metrics, _, _ = aggregate_performance(rows, lineage)

    assert set(metrics["model_origin"]) == {"champion", "retrained_challenger"}
    assert len(metrics) == 2


def test_observation_source_loader_accepts_only_complete_retrained_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reports/shadow_predictions" / AS_OF
    champion = production_champion_rows()
    champion_hash = logical_prediction_hash(champion)
    champion["prediction_hash"] = champion_hash
    publish_shadow_bundle(
        output_dir=root,
        predictions=champion,
        manifest_without_file_hash={
            "schema_version": 1,
            "artifact_name": "shadow_prediction_bundle",
            "production_run_id": "production-1",
            "shadow_run_id": "production-shadow-1",
            "prediction_hash": champion_hash,
            "feature_hash": "feature-hash",
            "universe_hash": "current-universe-hash",
            "prediction_rows": len(champion),
            "models": [
                {
                    "model_id": "champion-model",
                    "model_role": "champion",
                    "model_origin": "champion",
                    "feature_hash": "feature-hash",
                    "universe_hash": "current-universe-hash",
                    "source_models": [],
                    "fusion_method": None,
                    "access_policy": "prospective_production",
                }
            ],
        },
    )
    service, _ = shadow_service(tmp_path, monkeypatch)
    service.predict(MODEL_ID, as_of=AS_OF)
    monkeypatch.setattr(
        "ashare_quant.monitoring.performance_observation.validation.validate_production_publication",
        lambda **kwargs: {"run_id": "production-1"},
    )

    rows, manifests, hashes = load_shadow_sources(
        reports_root=tmp_path / "reports",
        runs_root=tmp_path / "runs",
        observation_as_of=AS_OF,
    )

    assert set(rows["model_origin"]) == {"champion", "retrained_challenger"}
    assert len(manifests) == 2
    assert set(hashes) == {AS_OF, f"{AS_OF}:retrained:{MODEL_ID}"}


def test_retraining_shadow_cli_success_and_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "reports/shadow_predictions" / AS_OF / "retrained" / MODEL_ID

    class FakeShadow:
        def __init__(self, **kwargs: object) -> None:
            pass

        def predict(self, model_id: str, *, as_of: str | None = None) -> RetrainedShadowResult:
            assert model_id == MODEL_ID
            assert as_of == AS_OF
            return RetrainedShadowResult(model_id, AS_OF, "shadow-id", 2, output)

        def status(self, model_id: str, *, as_of: str | None = None) -> dict[str, object]:
            return {"model_id": model_id, "as_of": as_of, "status": "complete"}

    monkeypatch.setattr("ashare_quant.cli.load_settings", lambda path: make_settings(tmp_path))
    monkeypatch.setattr("ashare_quant.cli.RetrainedChallengerShadowService", FakeShadow)

    common = ["retraining", "shadow", "--model-id", MODEL_ID, "--as-of", AS_OF]
    assert main(common) == 0
    assert "retraining_shadow:" in capsys.readouterr().out
    status = ["retraining", "shadow-status", "--model-id", MODEL_ID, "--as-of", AS_OF]
    assert main(status) == 0


def production_champion_rows() -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "trade_date": [AS_OF, AS_OF],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "model_id": ["champion-model", "champion-model"],
            "model_role": ["champion", "champion"],
            "model_origin": ["champion", "champion"],
            "native_horizon": [5, 5],
            "prediction_score": [0.9, 0.1],
            "rank": [1, 2],
            "score_percentile": [1.0, 0.5],
            "production_run_id": ["production-1", "production-1"],
            "shadow_run_id": ["production-shadow-1", "production-shadow-1"],
            "prediction_hash": ["", ""],
            "feature_hash": ["feature-hash", "feature-hash"],
            "universe_hash": ["current-universe-hash", "current-universe-hash"],
            "access_policy": ["prospective_production", "prospective_production"],
            "generated_at": ["2026-07-29T10:00:00+00:00"] * 2,
            "parent_model_id": ["", ""],
            "training_request_id": ["", ""],
            "training_run_id": ["", ""],
            "validation_run_id": ["", ""],
        }
    )
    return rows.loc[:, list(PREDICTION_COLUMNS)]


def shadow_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RetrainedChallengerShadowService, RetrainedShadowContext]:
    config = tmp_path / "config.yaml"
    config.write_text("environment: test\n", encoding="utf-8")
    settings = make_settings(tmp_path)
    model = RegisteredModel(
        model_id=MODEL_ID,
        experiment_id=MODEL_ID,
        model_type="lightgbm_ranker",
        feature_hash="feature-hash",
        feature_count=1,
        training_date_range={"start": "20200101", "end": "20251231"},
        validation_metrics={},
        test_metrics={},
        git_commit="commit",
        config_hash="config",
        creation_time="created",
        artifact_path=str(tmp_path / "models/challengers" / MODEL_ID),
        status="candidate",
    )
    lineage = RetrainedModelLineage(
        model_id=MODEL_ID,
        parent_model_id="parent-champion",
        training_request_id="request-1",
        training_run_id="training-1",
        validation_run_id="validation-1",
    )
    context = RetrainedShadowContext(
        as_of=AS_OF,
        production_run_id="production-1",
        production_shadow_run_id="production-shadow-1",
        current_universe_hash="current-universe-hash",
        generated_at="2026-07-29T10:00:00+00:00",
        model=model,
        horizon=10,
        artifact_hash="artifact-hash",
        feature_hash="feature-hash",
        training_universe_hash="training-universe-hash",
        validation_manifest_hash="validation-manifest-hash",
        lineage=lineage,
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.shadow.service.validate_retrained_shadow_eligibility",
        lambda **kwargs: (context, champion_keys()),
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.shadow.service.score_challenger",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "trade_date": [AS_OF, AS_OF],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "prediction_score": [0.2, 0.8],
                "rank": [2, 1],
            }
        ),
    )
    return RetrainedChallengerShadowService(settings=settings, config_path=config), context


def champion_keys() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [AS_OF, AS_OF],
            "ts_code": ["000001.SZ", "000002.SZ"],
        }
    )


def observation_row(
    model_id: str, model_origin: str, future_return: float, ts_code: str
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "model_role": "challenger_h10" if model_origin != "champion" else "champion",
        "model_origin": model_origin,
        "horizon": 10,
        "signal_date": AS_OF,
        "ts_code": ts_code,
        "prediction_score": 1.0 if future_return > 0 else 0.0,
        "rank": 1 if future_return > 0 else 2,
        "score_percentile": 1.0 if future_return > 0 else 0.5,
        "future_excess_ret": future_return,
        "label_status": "available",
        "feature_hash": "feature-hash",
        "universe_hash": "universe-hash",
    }


def make_settings(tmp_path: Path) -> AppSettings:
    return AppSettings.model_validate(
        {
            "paths": PathSettings(
                raw_data=tmp_path / "raw",
                processed_data=tmp_path / "processed",
                parquet_store=tmp_path / "parquet",
                duckdb_path=tmp_path / "test.duckdb",
                reports=tmp_path / "reports",
                models=tmp_path / "models",
                backtests=tmp_path / "backtests",
                paper_trading=tmp_path / "paper_trading",
                data_quality_logs=tmp_path / "logs",
            ).model_dump(mode="python")
        }
    )
