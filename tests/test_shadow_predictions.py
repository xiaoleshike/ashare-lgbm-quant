from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import (
    AppSettings,
    ModelExperimentSettings,
    ShadowChallengerModelSettings,
    ShadowPredictionSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.shadow.configuration import configured_model_ids
from ashare_quant.models.shadow.ensemble import (
    build_percentile_ensemble,
    ensemble_model_id,
)
from ashare_quant.models.shadow.model_loader import (
    _reject_historical_source_path,
    load_shadow_challengers,
)
from ashare_quant.models.shadow.schemas import (
    MODEL_ROLES,
    ReadinessResult,
    ShadowContext,
    ShadowPredictionResult,
)
from ashare_quant.models.shadow.service import (
    PREDICTION_COLUMNS,
    ShadowPredictionService,
)
from ashare_quant.models.shadow.storage import (
    canonical_payload_hash,
    logical_prediction_hash,
    publish_shadow_bundle,
    read_complete_manifest,
)

AS_OF = "20260729"
FEATURE_HASH = "feature-hash"
UNIVERSE_HASH = "universe-hash"
PRODUCTION_RUN_ID = "production-run-1"


class FakeReadiness:
    def __init__(self, context: ShadowContext | None, failure: str | None = None) -> None:
        self.context = context
        self.failure = failure

    def require_ready(self, as_of: str) -> ShadowContext:
        if self.failure is not None or self.context is None:
            raise DataValidationError(self.failure or "not ready")
        assert as_of == self.context.as_of
        return self.context

    def validate(self, as_of: str) -> tuple[ReadinessResult, ShadowContext | None]:
        if self.failure is not None:
            return ReadinessResult(False, (self.failure,), {}), None
        assert self.context is not None
        return self.context.readiness, self.context


def test_shadow_service_references_champion_and_scores_only_challengers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, context = _service(tmp_path)
    scored_models: list[str] = []

    def fake_score(model: RegisteredModel, **_: object) -> pd.DataFrame:
        scored_models.append(model.model_id)
        return _challenger_scores(context.as_of, int(model.experiment_id[1:]))

    monkeypatch.setattr("ashare_quant.models.shadow.service.score_challenger", fake_score)
    result = service.predict(AS_OF)

    assert scored_models == ["candidate_h5", "candidate_h10", "candidate_h20", "candidate_h60"]
    assert result.model_count == 6
    output = pd.read_parquet(result.output_dir / "predictions.parquet")
    assert tuple(output.columns) == PREDICTION_COLUMNS
    assert set(output["model_role"]) == MODEL_ROLES
    assert set(output["production_run_id"]) == {PRODUCTION_RUN_ID}
    assert set(output["shadow_run_id"]) == {result.shadow_run_id}
    champion = output.loc[output["model_role"] == "champion"]
    assert champion["prediction_score"].tolist() == [0.9, 0.2]
    assert champion["rank"].tolist() == [1, 2]
    manifest = json.loads((result.output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(output["prediction_hash"]) == {manifest["prediction_hash"]}
    assert manifest["schema_version"] == 1
    assert manifest["artifact_name"] == "shadow_prediction_bundle"
    assert manifest["production_run_id"] == PRODUCTION_RUN_ID
    assert all(model["prediction_hash"] for model in manifest["models"])
    assert len({model["prediction_hash"] for model in manifest["models"]}) == 6
    assert manifest["contracts"] == {
        "champion_recomputed": False,
        "labels_loaded": False,
        "future_data_loaded": False,
        "registry_modified": False,
        "candidate_selection_called": False,
        "paper_trading_called": False,
        "promotion_called": False,
        "hash_scope": "per-model rows exclude prediction_hash; bundle hash covers all rows",
    }


def test_champion_model_loader_is_never_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_model_loader(_: Path) -> object:
        raise AssertionError("Champion predict/model loading must not be called")

    service, context = _service(tmp_path, model_loader=forbidden_model_loader)
    monkeypatch.setattr(
        "ashare_quant.models.shadow.service.score_challenger",
        lambda model, **kwargs: _challenger_scores(context.as_of, int(model.experiment_id[1:])),
    )

    service.predict(AS_OF)


def test_shadow_scoring_does_not_use_labels_future_or_research_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, context = _service(tmp_path)
    labels = tmp_path / "processed" / "labels_forward" / "forbidden.parquet"
    labels.parent.mkdir(parents=True)
    labels.write_bytes(b"must not be read")
    original_read = pd.read_parquet

    def guarded_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if "labels_forward" in str(path):
            raise AssertionError("labels must not be read")
        return cast(pd.DataFrame, original_read(path, *args, **kwargs))  # type: ignore[call-overload]

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    monkeypatch.setattr(
        "ashare_quant.models.shadow.service.score_challenger",
        lambda model, **kwargs: _challenger_scores(context.as_of, int(model.experiment_id[1:])),
    )
    monkeypatch.setattr(
        ModelRegistry,
        "promote_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("promotion must not be called")
        ),
    )
    monkeypatch.setattr(
        "ashare_quant.strategy.candidate_selector.CandidateSelector.select",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("candidate selection must not be called")
        ),
    )
    monkeypatch.setattr(
        "ashare_quant.paper_trading.service.PaperTradingService.run_daily",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paper trading must not be called")
        ),
    )

    result = service.predict(AS_OF)

    assert result.prediction_rows == 12


def test_readiness_failure_prevents_scoring_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = _service(tmp_path)
    service.readiness = cast(Any, FakeReadiness(None, "frozen_oos_evaluation is prohibited"))
    monkeypatch.setattr(
        "ashare_quant.models.shadow.service.score_challenger",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("scoring must not start")),
    )

    with pytest.raises(DataValidationError, match="frozen_oos_evaluation"):
        service.predict(AS_OF)

    assert not (tmp_path / "reports" / "shadow_predictions" / AS_OF).exists()


def test_future_or_mismatched_challenger_keys_fail_without_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, context = _service(tmp_path)

    def future_score(model: RegisteredModel, **_: object) -> pd.DataFrame:
        frame = _challenger_scores(context.as_of, int(model.experiment_id[1:]))
        if model.experiment_id == "h20":
            frame["trade_date"] = "20260730"
        return frame

    monkeypatch.setattr("ashare_quant.models.shadow.service.score_challenger", future_score)

    with pytest.raises(DataValidationError, match="keys differ"):
        service.predict(AS_OF)

    assert not (tmp_path / "reports" / "shadow_predictions" / AS_OF).exists()


def test_percentile_ensemble_and_ids_are_deterministic() -> None:
    frames = {
        horizon: _challenger_scores(AS_OF, horizon).assign(score_percentile=[1.0, 0.5])
        for horizon in (5, 10, 20, 60)
    }
    result = build_percentile_ensemble(frames)
    assert result["prediction_score"].tolist() == [1.0, 0.5]
    first = ensemble_model_id(["b", "a"], ["hash-b", "hash-a"], "percentile_mean")
    second = ensemble_model_id(["a", "b"], ["hash-a", "hash-b"], "percentile_mean")
    assert first == second
    assert first.startswith("ensemble_")
    assert len(first) == len("ensemble_") + 64
    assert canonical_payload_hash({"flag": True}) != canonical_payload_hash({"flag": 1})


def test_ensemble_rejects_missing_horizon_and_different_universe() -> None:
    frames = {
        horizon: _challenger_scores(AS_OF, horizon).assign(score_percentile=[1.0, 0.5])
        for horizon in (5, 10, 20, 60)
    }
    with pytest.raises(DataValidationError, match="requires horizons"):
        build_percentile_ensemble({key: value for key, value in frames.items() if key != 60})
    frames[60] = frames[60].iloc[:1]
    with pytest.raises(DataValidationError, match="stock keys differ"):
        build_percentile_ensemble(frames)


def test_shadow_run_identity_is_environment_independent(tmp_path: Path) -> None:
    service, context = _service(tmp_path)
    first = service._shadow_run_id(context)
    changed_execution_metadata = replace(
        context,
        generated_at="2099-01-01T00:00:00+00:00",
        champion_prediction_file_hash="different-parquet-serialization",
        readiness=ReadinessResult(True, (), {"hostname": "another-host"}),
    )
    second = service._shadow_run_id(changed_execution_metadata)
    assert first == second
    assert first.startswith(f"shadow_{AS_OF}_")


def test_duplicate_identity_is_idempotent_and_different_identity_cannot_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, context = _service(tmp_path)
    monkeypatch.setattr(
        "ashare_quant.models.shadow.service.score_challenger",
        lambda model, **kwargs: _challenger_scores(context.as_of, int(model.experiment_id[1:])),
    )
    first = service.predict(AS_OF)
    monkeypatch.setattr(
        "ashare_quant.models.shadow.service.score_challenger",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("idempotent execution must not rescore")
        ),
    )
    second = service.predict(AS_OF)
    assert second.idempotent
    assert second.shadow_run_id == first.shadow_run_id

    service.readiness = cast(Any, FakeReadiness(replace(context, universe_hash="changed")))
    with pytest.raises(DataValidationError, match="different logical input identity"):
        service.predict(AS_OF)


def test_partial_artifact_is_not_accepted_as_complete(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    output = tmp_path / "reports" / "shadow_predictions" / AS_OF
    output.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(output / "predictions.parquet", index=False)

    assert service.status(AS_OF)["status"] == "incomplete"
    with pytest.raises(DataValidationError, match="incomplete shadow output"):
        service.predict(AS_OF)


def test_atomic_publication_failure_leaves_no_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [AS_OF],
            "model_id": ["model"],
            "ts_code": ["000001.SZ"],
            "prediction_score": [0.1],
        }
    )
    output = tmp_path / "shadow" / AS_OF
    monkeypatch.setattr(
        "ashare_quant.models.shadow.storage.atomic_write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failure")),
    )

    with pytest.raises(OSError, match="disk failure"):
        publish_shadow_bundle(
            output_dir=output,
            predictions=_self_hashed(frame),
            manifest_without_file_hash={
                "schema_version": 1,
                "artifact_name": "shadow_prediction_bundle",
                "prediction_hash": logical_prediction_hash(frame),
            },
        )

    assert not output.exists()
    assert not list(output.parent.glob(f".{AS_OF}.*"))


def test_manifest_is_written_last_and_hashes_are_verified(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "trade_date": [AS_OF],
            "model_id": ["model"],
            "ts_code": ["000001.SZ"],
            "prediction_score": [0.1],
        }
    )
    output = tmp_path / "shadow" / AS_OF
    publish_shadow_bundle(
        output_dir=output,
        predictions=_self_hashed(frame),
        manifest_without_file_hash={
            "schema_version": 1,
            "artifact_name": "shadow_prediction_bundle",
            "prediction_hash": logical_prediction_hash(frame),
        },
    )
    assert read_complete_manifest(output) is not None
    pd.DataFrame({"tampered": [1]}).to_parquet(output / "predictions.parquet", index=False)
    with pytest.raises(DataValidationError, match="hash differs"):
        read_complete_manifest(output)


def test_configuration_rejects_frozen_policy_and_duplicate_models() -> None:
    settings = _shadow_settings()
    assert set(configured_model_ids(settings)) == {5, 10, 20, 60}
    with pytest.raises(DataValidationError, match="frozen_oos_evaluation"):
        configured_model_ids(settings.model_copy(update={"access_policy": "frozen_oos_evaluation"}))
    with pytest.raises(ValueError, match="must be unique"):
        ShadowPredictionSettings(
            challenger_models={
                key: ShadowChallengerModelSettings(model_id="same")
                for key in ("h5", "h10", "h20", "h60")
            }
        )


def test_challenger_loader_rejects_historical_evaluation_source(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="historical evaluation artifact"):
        _reject_historical_source_path(tmp_path / "reports" / "challenger_predictions" / "model")
    with pytest.raises(DataValidationError, match="historical evaluation artifact"):
        _reject_historical_source_path(tmp_path / "reports" / "ensemble_evaluation" / "run")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("frozen", "frozen_oos_evaluation"),
        ("wrong_horizon", "horizon mismatch"),
        ("wrong_feature_hash", "feature identity mismatch"),
        ("not_candidate", "must have candidate status"),
    ],
)
def test_challenger_loader_enforces_artifact_contract(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    settings = _shadow_settings()
    models = {horizon: _write_candidate_artifact(tmp_path, horizon) for horizon in (5, 10, 20, 60)}
    target = models[20]
    artifact = Path(target.artifact_path)
    if mutation == "frozen":
        payload = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        payload["access_policy"] = "frozen_oos_evaluation"
        (artifact / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong_horizon":
        payload = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        payload["horizon"] = 10
        (artifact / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    elif mutation == "wrong_feature_hash":
        payload = json.loads((artifact / "feature_list.json").read_text(encoding="utf-8"))
        payload["feature_hash"] = "wrong"
        (artifact / "feature_list.json").write_text(json.dumps(payload), encoding="utf-8")
    else:
        models[20] = replace(target, status="champion")

    registry = FakeRegistry(tuple(models.values()))
    with pytest.raises(DataValidationError, match=message):
        load_shadow_challengers(registry, settings)  # type: ignore[arg-type]


def test_shadow_cli_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeService:
        def __init__(self, **_: object) -> None:
            pass

        def predict(self, as_of: str) -> ShadowPredictionResult:
            return ShadowPredictionResult(
                as_of,
                PRODUCTION_RUN_ID,
                "shadow-id",
                12,
                6,
                tmp_path / "reports" / "shadow_predictions" / as_of,
            )

        def status(self, as_of: str) -> dict[str, Any]:
            return {
                "status": "missing",
                "as_of": as_of,
                "shadow_run_id": None,
                "prediction_rows": 0,
                "output": "missing",
            }

        def validate(self, as_of: str) -> tuple[bool, tuple[str, ...], dict[str, Any]]:
            return False, ("not ready",), {}

    monkeypatch.setattr("ashare_quant.cli.ShadowPredictionService", FakeService)
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
    assert main([*common, "shadow-predict", "--as-of", AS_OF]) == 0
    assert "models=6" in capsys.readouterr().out
    assert main([*common, "shadow-status", "--as-of", AS_OF]) == 1
    assert main([*common, "shadow-validate", "--as-of", AS_OF]) == 1


def _service(
    tmp_path: Path,
    *,
    model_loader: Any = None,
) -> tuple[ShadowPredictionService, ShadowContext]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("environment: test\n", encoding="utf-8")
    settings = AppSettings(models=ModelExperimentSettings(shadow_predictions=_shadow_settings()))
    registry = ModelRegistry(tmp_path / "models")
    service = ShadowPredictionService(
        settings=settings,
        config_path=config_path,
        registry=registry,
        processed_root=tmp_path / "processed",
        reports_root=tmp_path / "reports",
        runs_root=tmp_path / "runs",
        model_loader=model_loader,
    )
    context = ShadowContext(
        as_of=AS_OF,
        production_run_id=PRODUCTION_RUN_ID,
        champion_model_id="champion-model",
        champion_feature_hash=FEATURE_HASH,
        champion_prediction_hash="champion-logical-hash",
        champion_prediction_file_hash="champion-file-hash",
        feature_hash=FEATURE_HASH,
        universe_hash=UNIVERSE_HASH,
        generated_at="2026-07-29T10:00:00+00:00",
        champion_predictions=pd.DataFrame(
            {
                "trade_date": [AS_OF, AS_OF],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "prediction_score": [0.9, 0.2],
                "rank": [1, 2],
            }
        ),
        challenger_models={horizon: _registered_model(horizon) for horizon in (5, 10, 20, 60)},
        challenger_manifest_hashes={
            horizon: f"manifest-hash-{horizon}" for horizon in (5, 10, 20, 60)
        },
        readiness=ReadinessResult(
            True,
            (),
            {
                "labels_loaded": False,
                "historical_evaluation_sources_used": False,
            },
        ),
    )
    service.readiness = cast(Any, FakeReadiness(context))
    return service, context


def _shadow_settings() -> ShadowPredictionSettings:
    return ShadowPredictionSettings(
        challenger_models={
            f"h{horizon}": ShadowChallengerModelSettings(model_id=f"candidate_h{horizon}")
            for horizon in (5, 10, 20, 60)
        }
    )


def _registered_model(horizon: int) -> RegisteredModel:
    return RegisteredModel(
        model_id=f"candidate_h{horizon}",
        experiment_id=f"h{horizon}",
        model_type="lightgbm_ranker",
        feature_hash=FEATURE_HASH,
        feature_count=2,
        training_date_range={"start": "20100101", "end": "20251231"},
        validation_metrics={},
        test_metrics={},
        git_commit="commit",
        config_hash="config",
        creation_time="2026-07-01T00:00:00+00:00",
        artifact_path=f"/models/candidate_h{horizon}",
        status="candidate",
    )


class FakeRegistry:
    def __init__(self, records: tuple[RegisteredModel, ...]) -> None:
        self.records = records

    def list_models(self) -> tuple[RegisteredModel, ...]:
        return self.records


def _write_candidate_artifact(tmp_path: Path, horizon: int) -> RegisteredModel:
    artifact = tmp_path / "models" / f"candidate_h{horizon}"
    artifact.mkdir(parents=True)
    features = ("f1", "f2")
    digest = feature_list_hash(features)
    (artifact / "model.txt").write_text("model", encoding="utf-8")
    (artifact / "feature_list.json").write_text(
        json.dumps({"features": list(features), "feature_hash": digest}),
        encoding="utf-8",
    )
    (artifact / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": "lightgbm_ranker_challenger",
                "horizon": horizon,
                "feature_list_hash": digest,
            }
        ),
        encoding="utf-8",
    )
    return replace(
        _registered_model(horizon),
        feature_hash=digest,
        artifact_path=str(artifact),
    )


def _challenger_scores(as_of: str, horizon: int) -> pd.DataFrame:
    offset = horizon / 1000.0
    return pd.DataFrame(
        {
            "trade_date": [as_of, as_of],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "prediction_score": [0.8 + offset, 0.1 + offset],
            "rank": [1, 2],
        }
    )


def _self_hashed(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["prediction_hash"] = logical_prediction_hash(result)
    return result
