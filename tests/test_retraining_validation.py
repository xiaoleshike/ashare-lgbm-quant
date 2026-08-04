"""Governed retrained-Challenger validation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, PathSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.ranker_data import RankerDataset
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.retraining.execution.artifact import write_staged_artifact
from ashare_quant.retraining.execution.schemas import (
    CandidateRegistration,
    DatasetManifest,
    PreparedTrainingData,
    TrainedRanker,
)
from ashare_quant.retraining.validation.artifact_validation import (
    validate_candidate_artifact,
)
from ashare_quant.retraining.validation.executable import RetrainingExecutableValidator
from ashare_quant.retraining.validation.offline import RetrainingOfflineValidator
from ashare_quant.retraining.validation.schemas import (
    ExecutableValidationEvidence,
    OfflineValidationEvidence,
    OfflineValidationRun,
    ShadowEligibilityEvidence,
)
from ashare_quant.retraining.validation.service import RetrainingValidationService
from ashare_quant.utils.manifest import atomic_write_json, config_hash

MODEL_ID = "challenger_refresh_h5_fixture"
TRAINING_RUN_ID = "retraining_fixture"
EVALUATION_DATES = ("20230103", "20230104")


class _Booster:
    def save_model(self, path: str) -> None:
        Path(path).write_text("fixture model\n", encoding="utf-8")


class _Model:
    booster_ = _Booster()


class _PredictionModel:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data["f1"].to_numpy(dtype=float)


class _Offline:
    def evaluate(self, context) -> OfflineValidationRun:
        rows = [
            {
                "trade_date": date,
                "ts_code": f"{index:06d}.SZ",
                "prediction_score": float(index),
                "model_id": context.model.model_id,
                "rank": 20 - index,
            }
            for date in EVALUATION_DATES
            for index in range(20)
        ]
        evidence = OfflineValidationEvidence(
            model_id=context.model.model_id,
            horizon=5,
            evaluation_start=EVALUATION_DATES[0],
            evaluation_end=EVALUATION_DATES[-1],
            prediction_rows=40,
            labelled_rows=40,
            evaluation_sessions=20,
            overall_metrics={"rank_ic": 0.1, "icir": 1.0},
            stability_metrics=(),
        )
        return OfflineValidationRun(evidence, pd.DataFrame(rows))


class _Executable:
    def evaluate(self, context, offline) -> ExecutableValidationEvidence:
        del offline
        return ExecutableValidationEvidence(
            model_id=context.model.model_id,
            horizon=5,
            holding_period=5,
            signal_dates=20,
            minimum_signal_date=EVALUATION_DATES[0],
            maximum_signal_date=EVALUATION_DATES[-1],
            top_n=(10, 20, 50),
            execution_config={"commission": 0.00025, "slippage": 0.0005},
            metrics={"10": {}, "20": {}, "50": {}},
        )


class _Shadow:
    def evaluate(self, context) -> ShadowEligibilityEvidence:
        return ShadowEligibilityEvidence(
            model_id=context.model.model_id,
            shadow_eligible=True,
            feature_hash_compatible=True,
            universe_compatible=True,
            deployment_contract_compatible=True,
            inference_adapter_available=True,
        )


def test_validation_publishes_immutable_evidence_and_is_idempotent(tmp_path: Path) -> None:
    paths = validation_fixture(tmp_path)
    service = validation_service(tmp_path)
    registry_before = paths["registry"].read_bytes()

    first = service.validate(MODEL_ID)
    second = service.validate(MODEL_ID)

    assert first.status == "COMPLETED"
    assert first.promotion_ready is True
    assert second.idempotent is True
    assert paths["registry"].read_bytes() == registry_before
    assert set(
        path.relative_to(first.output_dir).as_posix() for path in first.output_dir.rglob("*")
    ) >= {
        "offline/metrics.json",
        "executable/summary.json",
        "shadow/eligibility.json",
        "evidence.json",
        "manifest.json",
    }
    manifest = read_json(first.output_dir / "manifest.json")
    assert manifest["promotion_ready"] is True
    assert manifest["registry_modified"] is False
    assert manifest["promotion_executed"] is False
    assert manifest["trading_executed"] is False


@pytest.mark.parametrize("failure", ["model", "feature", "universe"])
def test_candidate_artifact_validation_rejects_identity_failures(
    tmp_path: Path, failure: str
) -> None:
    paths = validation_fixture(tmp_path)
    if failure == "model":
        paths["model"].unlink()
    elif failure == "feature":
        atomic_write_json(paths["feature_list"], {"features": ["wrong"]})
    else:
        atomic_write_json(paths["universe_manifest"], {"artifact_name": "changed"})

    with pytest.raises(DataValidationError, match="VALIDATION_FAILED|artifact"):
        load_context(tmp_path)


def test_final_test_fold_and_dataset_misuse_are_rejected(tmp_path: Path) -> None:
    paths = validation_fixture(tmp_path)
    plan = read_json(paths["horizon_plan"])
    experiment = plan["experiments"][0]
    experiment["final_test_period"]["folds"] = experiment["selection_period"]["folds"]
    atomic_write_json(paths["horizon_plan"], plan)

    with pytest.raises(DataValidationError, match="selection|final-test"):
        load_context(tmp_path)

    validation_fixture(tmp_path / "misuse")
    dataset = tmp_path / "misuse/models/challengers" / MODEL_ID / "dataset_manifest.json"
    payload = read_json(dataset)
    payload["production_observation_loaded"] = True
    atomic_write_json(dataset, payload)
    with pytest.raises(DataValidationError, match="artifact hash|schema"):
        load_context(tmp_path / "misuse")


def test_offline_validation_generates_rank_and_stability_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_fixture(tmp_path)
    context = load_context(tmp_path)
    predictions = pd.DataFrame(
        [
            {
                "trade_date": date,
                "ts_code": f"{index:06d}.SZ",
                "prediction_score": float(index),
                "model_id": MODEL_ID,
                "rank": 20 - index,
            }
            for date in EVALUATION_DATES
            for index in range(20)
        ]
    )
    labels = predictions[["trade_date", "ts_code"]].copy()
    labels["future_excess_ret"] = predictions["prediction_score"] / 100
    labels["benchmark_forward_ret"] = np.repeat([0.02, -0.02], 20)
    batch = type("Batch", (), {"predictions": predictions})()
    monkeypatch.setattr(
        "ashare_quant.retraining.validation.offline.score_registered_model_range",
        lambda *args, **kwargs: batch,
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.validation.offline._load_mature_labels",
        lambda *args, **kwargs: labels,
    )

    result = RetrainingOfflineValidator(
        processed_root=tmp_path / "processed",
        settings=make_settings(tmp_path),
    ).evaluate(context)

    assert result.evidence.overall_metrics["rank_ic"] == pytest.approx(1.0)
    assert {row["period"] for row in result.evidence.stability_metrics} >= {
        "all",
        "2023",
        "bull",
        "bear",
    }
    assert result.evidence.final_test_loaded is False
    assert result.evidence.production_observation_loaded is False


def test_executable_validation_uses_frozen_scores_and_execution_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_fixture(tmp_path)
    context = load_context(tmp_path)
    offline = _Offline().evaluate(context)
    calendar = EVALUATION_DATES + tuple(f"202302{day:02d}" for day in range(1, 32))
    prices = pd.DataFrame(
        {
            "trade_date": np.repeat(calendar, 2),
            "ts_code": ["000000.SZ", "000001.SZ"] * len(calendar),
            "open": 10.0,
            "close": 10.0,
            "can_buy": True,
            "can_sell": True,
        }
    )
    benchmark = pd.DataFrame({"trade_date": calendar, "close": 100.0})

    class Result:
        def __init__(self, top_n: int) -> None:
            self.top_n = top_n
            self.metrics = {"annual_return": 0.01, "average_turnover": 0.1}
            self.holdings = pd.DataFrame()

    monkeypatch.setattr(
        "ashare_quant.retraining.validation.executable.load_calendar",
        lambda *args, **kwargs: list(calendar),
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.validation.executable.load_execution_prices",
        lambda *args, **kwargs: prices,
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.validation.executable.load_benchmark",
        lambda *args, **kwargs: benchmark,
    )
    monkeypatch.setattr(
        "ashare_quant.retraining.validation.executable.simulate_portfolio",
        lambda inputs, top_n, settings: Result(top_n),
    )

    evidence = RetrainingExecutableValidator(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        settings=make_settings(tmp_path),
    ).evaluate(context, offline)

    assert evidence.execution_rule == "next_open"
    assert evidence.holding_period == 5
    assert evidence.execution_config["commission"] == pytest.approx(0.00025)
    assert evidence.execution_config["stamp_duty"] == pytest.approx(0.001)
    assert evidence.execution_config["slippage"] == pytest.approx(0.0005)
    assert evidence.labels_loaded is False
    assert evidence.trading_state_modified is False


def test_shadow_eligibility_and_cli_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = validation_fixture(tmp_path)
    from ashare_quant.models.inference import ProductionInferenceEngine
    from ashare_quant.models.promotion.apply import PromotionApplyService
    from ashare_quant.models.promotion.rollback import RollbackService
    from ashare_quant.models.registry import ModelRegistry
    from ashare_quant.paper_trading import PaperTradingService
    from ashare_quant.strategy import CandidateSelector

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden stateful service called")

    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(PromotionApplyService, "apply", forbidden)
    monkeypatch.setattr(RollbackService, "apply", forbidden)
    monkeypatch.setattr(ModelRegistry, "register_model", forbidden)
    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)
    monkeypatch.setattr(CandidateSelector, "select", forbidden)
    fake = validation_service(tmp_path)
    monkeypatch.setattr("ashare_quant.cli.RetrainingValidationService", lambda **kwargs: fake)
    before = paths["registry"].read_bytes()

    code = main(
        [
            "--config",
            str(tmp_path / "config/default.yaml"),
            "retraining",
            "validate",
            "--model-id",
            MODEL_ID,
        ]
    )

    assert code == 0
    assert "promotion_ready=True" in capsys.readouterr().out
    assert paths["registry"].read_bytes() == before


def test_root_manifest_is_written_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    validation_fixture(tmp_path)
    from ashare_quant.retraining.validation import storage as storage_module

    writes: list[str] = []
    original = storage_module.atomic_write_json

    def tracked(path: Path, payload: dict[str, Any]) -> None:
        writes.append(path.relative_to(path.parents[1]).as_posix())
        original(path, payload)

    monkeypatch.setattr(storage_module, "atomic_write_json", tracked)

    validation_service(tmp_path).validate(MODEL_ID)

    assert writes[-1].endswith("manifest.json")


def test_atomic_failure_does_not_publish_and_changed_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validation_fixture(tmp_path)
    service = validation_service(tmp_path)
    from ashare_quant.retraining.validation import storage as storage_module

    original_replace = storage_module.os.replace

    def fail_replace(source: Path, target: Path) -> None:
        if "retraining_validation_" in str(target):
            raise OSError("forced validation publish failure")
        original_replace(source, target)

    monkeypatch.setattr(storage_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced validation publish failure"):
        service.validate(MODEL_ID)
    assert not list((tmp_path / "reports/retraining_validation").glob("retraining_validation_*"))

    monkeypatch.setattr(storage_module.os, "replace", original_replace)
    result = service.validate(MODEL_ID)
    atomic_write_json(result.output_dir / "evidence.json", {"tampered": True})
    with pytest.raises(DataValidationError, match="hash mismatch"):
        service.validate(MODEL_ID)


def validation_service(tmp_path: Path) -> RetrainingValidationService:
    return RetrainingValidationService(
        settings=make_settings(tmp_path),
        config_path=tmp_path / "config/default.yaml",
        offline_validator=_Offline(),
        executable_validator=_Executable(),
        shadow_validator=_Shadow(),
    )


def load_context(tmp_path: Path):
    return validate_candidate_artifact(
        model_id=MODEL_ID,
        models_root=tmp_path / "models",
        reports_root=tmp_path / "reports",
        processed_root=tmp_path / "processed",
        config_path=tmp_path / "config/default.yaml",
    )


def validation_fixture(tmp_path: Path) -> dict[str, Path]:
    settings = make_settings(tmp_path)
    config = tmp_path / "config/default.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "raw_data": str(settings.paths.raw_data),
                    "processed_data": str(settings.paths.processed_data),
                    "reports": str(settings.paths.reports),
                    "models": str(settings.paths.models),
                },
                "models": {"challenger_evaluation": {"minimum_labelled_days": 20}},
            }
        ),
        encoding="utf-8",
    )
    processed = settings.paths.processed_data
    hashes: dict[str, str] = {}
    for name in ("features_daily", "universe_daily", "labels_forward"):
        path = processed / name / "_manifest.json"
        atomic_write_json(path, {"artifact_name": name, "max_date": "20260731"})
        hashes[name] = file_sha256(path)
    features = ("f1",)
    feature_hash = feature_list_hash(features)
    folds_dir = tmp_path / "reports/walk_forward/fold"
    folds_path = folds_dir / "folds.json"
    fold = {
        "fold_id": "fold_selection",
        "train_start": "20100104",
        "train_end": "20191231",
        "validation_start": "20200102",
        "validation_end": "20221230",
        "evaluation_start": EVALUATION_DATES[0],
        "evaluation_end": EVALUATION_DATES[-1],
        "purge_sessions": 61,
        "embargo_sessions": 61,
    }
    atomic_write_json(folds_path, {"schema_version": 2, "folds": [fold]})
    folds_manifest = folds_dir / "manifest.json"
    atomic_write_json(
        folds_manifest,
        {
            "schema_version": 2,
            "artifact_name": "purged_walk_forward_plan",
            "feature_hash": feature_hash,
            "outputs": {"folds": "folds.json"},
        },
    )
    dataset = DatasetManifest(
        feature_hash=feature_hash,
        feature_manifest_hash=hashes["features_daily"],
        universe_hash=hashes["universe_daily"],
        label_hash=hashes["labels_forward"],
        horizon=5,
        label_name="future_excess_ret_5d",
        train_dates={"start": "20100104", "end": "20191231"},
        validation_dates={"start": "20200102", "end": "20221230"},
        fold_manifest=str(folds_manifest),
        fold_manifest_hash=file_sha256(folds_manifest),
        fold_id="fold_selection",
    )
    frame = pd.DataFrame(
        {
            "trade_date": ["20200102", "20200102"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "f1": [0.1, 0.2],
            "relevance": [0, 1],
        }
    )
    ranker = RankerDataset(frame, features)
    prepared = PreparedTrainingData(dataset, features, ranker, ranker, 5, "next_open")
    artifact_dir = tmp_path / "models/challengers" / MODEL_ID
    manifest = write_staged_artifact(
        directory=artifact_dir,
        model_id=MODEL_ID,
        training_run_id=TRAINING_RUN_ID,
        request_hash="r" * 64,
        prepared=prepared,
        trained=TrainedRanker(_Model(), {"rank_ic": 0.1}, []),
        config_hash_value=str(config_hash(config)),
        git_commit="abc",
        git_dirty=False,
    )
    registration_id = (
        "candidate_"
        + canonical_payload_hash(
            {
                "model_id": MODEL_ID,
                "training_run_id": TRAINING_RUN_ID,
                "artifact_hash": manifest.artifact_hash,
            }
        )[:16]
    )
    registration = CandidateRegistration(
        model_id=MODEL_ID,
        candidate_registration_id=registration_id,
        training_run_id=TRAINING_RUN_ID,
        artifact_path=str(artifact_dir),
        artifact_hash=manifest.artifact_hash,
        feature_hash=feature_hash,
        horizon=5,
    )
    registration_dir = tmp_path / "models/candidate_registrations" / MODEL_ID
    registration_path = registration_dir / "registration.json"
    atomic_write_json(registration_path, registration.model_dump(mode="json"))
    atomic_write_json(
        registration_dir / "manifest.json",
        {"registration_sha256": file_sha256(registration_path)},
    )
    execution_dir = tmp_path / "reports/retraining/executions" / TRAINING_RUN_ID
    atomic_write_json(
        execution_dir / "manifest.json",
        {
            "status": "COMPLETED",
            "model_id": MODEL_ID,
            "artifact_hash": manifest.artifact_hash,
        },
    )
    experiment = {
        "experiment_id": "h5_fixture",
        "horizon": 5,
        "holding_period": 5,
        "execution_rule": "next_open",
        "label_name": "future_excess_ret_5d",
        "feature_hash": feature_hash,
        "universe_hash": hashes["universe_daily"],
        "config_hash": str(config_hash(config)),
        "label_maturity_sessions": 6,
        "required_purge_sessions": 6,
        "required_embargo_sessions": 6,
        "maximum_mature_evaluation_date": EVALUATION_DATES[-1],
        "selection_period": {
            "may_select_model": True,
            "folds": [{"fold_id": "fold_selection"}],
        },
        "final_test_period": {"may_select_model": False, "folds": []},
    }
    horizon_plan = tmp_path / "reports/horizon_experiments/fixture/experiment_manifest.json"
    atomic_write_json(
        horizon_plan,
        {
            "config_hash": str(config_hash(config)),
            "feature_hash": feature_hash,
            "universe_hash": hashes["universe_daily"],
            "folds_manifest": str(folds_manifest),
            "folds_manifest_hash": file_sha256(folds_manifest),
            "folds_hash": file_sha256(folds_path),
            "experiments": [experiment],
        },
    )
    registry = tmp_path / "models/registry.json"
    atomic_write_json(registry, {"schema_version": 1, "models": []})
    return {
        "model": artifact_dir / "model.txt",
        "feature_list": artifact_dir / "feature_list.json",
        "universe_manifest": processed / "universe_daily/_manifest.json",
        "horizon_plan": horizon_plan,
        "registry": registry,
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
            ).model_dump(mode="python"),
            "models": {"challenger_evaluation": {"minimum_labelled_days": 20}},
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
