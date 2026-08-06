"""Governed retrained Challenger lifecycle orchestration tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, PathSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.execution.schemas import CandidateRegistration, ExecutionResult
from ashare_quant.retraining.orchestration.lifecycle import require_transition
from ashare_quant.retraining.orchestration.schemas import (
    LifecycleInput,
    ObservationProgress,
)
from ashare_quant.retraining.orchestration.service import RetrainingLifecycleOrchestrator
from ashare_quant.retraining.readiness.schemas import (
    ReadinessCheck,
    ReadinessResult,
    RetrainingReadinessReport,
)
from ashare_quant.retraining.schemas import (
    EvidenceReference,
    RetrainingEvidence,
    TrainingRequest,
    TrainingTarget,
)
from ashare_quant.retraining.shadow.schemas import RetrainedShadowResult
from ashare_quant.retraining.validation.schemas import RetrainingValidationResult
from ashare_quant.utils.manifest import atomic_write_json

REQUEST_ID = "training_lifecycle_fixture"
PARENT_ID = "champion-parent"
MODEL_ID = "challenger_refresh_h10_fixture"
AS_OF = "20260731"
NOW = datetime(2026, 8, 1, tzinfo=UTC)


def test_state_machine_rejects_skips_and_promotion() -> None:
    require_transition("REQUEST_ACCEPTED", "READINESS_CHECKING")
    with pytest.raises(DataValidationError, match="forbidden"):
        require_transition("REQUEST_ACCEPTED", "TRAINING_COMPLETED")
    with pytest.raises(DataValidationError, match="forbidden"):
        require_transition("VALIDATION_FAILED", "SHADOW_ENROLLING")
    with pytest.raises(DataValidationError, match="forbidden"):
        require_transition("OBSERVATION_PENDING", "EVIDENCE_READY")


def test_full_lifecycle_reuses_services_and_stops_at_observation_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch)

    result = service.run(REQUEST_ID)

    assert result.current_state == "OBSERVATION_PENDING"
    assert result.model_id == MODEL_ID
    assert calls == ["readiness", "training", "validation", "shadow"]
    snapshot = service.storage.read(result.lifecycle_run_id)
    assert snapshot is not None
    assert [event.state for event in snapshot.events] == [
        "REQUEST_ACCEPTED",
        "READINESS_CHECKING",
        "READINESS_READY",
        "TRAINING",
        "TRAINING_COMPLETED",
        "VALIDATING",
        "VALIDATION_COMPLETED",
        "SHADOW_ENROLLING",
        "SHADOW_ENROLLED",
        "OBSERVATION_PENDING",
    ]
    assert snapshot.summary.parent_model_id == PARENT_ID
    assert snapshot.summary.training_run_id == "training-run-1"
    assert snapshot.summary.validation_run_id == "validation-run-1"
    assert snapshot.summary.shadow_run_id == "shadow-run-1"
    assert snapshot.manifest is not None
    assert snapshot.manifest.model_origin == "retrained_challenger"


def test_stop_after_and_resume_preserve_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch)
    first = service.run(REQUEST_ID, stop_after="training")
    assert first.current_state == "TRAINING_COMPLETED"
    assert calls == ["readiness", "training"]

    resumed = service.resume(first.lifecycle_run_id)

    assert resumed.lifecycle_run_id == first.lifecycle_run_id
    assert resumed.current_state == "OBSERVATION_PENDING"
    assert calls == ["readiness", "training", "readiness", "validation", "shadow"]


def test_readiness_failure_prevents_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch, readiness_status="FAILED")

    result = service.run(REQUEST_ID)

    assert result.current_state == "READINESS_FAILED"
    assert calls == ["readiness"]
    assert not (tmp_path / "models/challengers" / MODEL_ID).exists()


@pytest.mark.parametrize(
    ("failure", "expected", "forbidden_call"),
    [
        ("training", "TRAINING_FAILED", "validation"),
        ("validation", "VALIDATION_FAILED", "shadow"),
        ("shadow", "SHADOW_FAILED", "observation"),
    ],
)
def test_stage_failure_blocks_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: str,
    forbidden_call: str,
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch, failure=failure)

    result = service.run(REQUEST_ID)

    assert result.current_state == expected
    assert forbidden_call not in calls
    if failure in {"validation", "shadow"}:
        assert (tmp_path / "models/challengers" / MODEL_ID / "manifest.json").is_file()


def test_observation_threshold_excludes_other_origins_and_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    first = service.run(REQUEST_ID)
    observation_fixture(tmp_path, sessions=3)
    # Policy objects are frozen; use a direct tracker call with a small threshold.
    from ashare_quant.retraining.orchestration.stages import track_prospective_observations

    progress = track_prospective_observations(
        reports_root=tmp_path / "reports",
        model_id=MODEL_ID,
        horizon=10,
        training_run_id="training-run-1",
        validation_run_id="validation-run-1",
        required_sessions=3,
    )

    assert first.current_state == "OBSERVATION_PENDING"
    assert progress.status == "OBSERVATION_SUFFICIENT"
    assert progress.mature_sessions == 3


def test_same_identity_is_idempotent_and_recovery_is_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    first = service.run(REQUEST_ID, stop_after="readiness")
    before = (first.output_dir / "manifest.json").read_bytes()

    second = service.run(REQUEST_ID, stop_after="readiness")
    recovery = service.recovery(first.lifecycle_run_id)

    assert second.lifecycle_run_id == first.lifecycle_run_id
    assert second.idempotent is True
    assert recovery.status == "CLEAN"
    assert (first.output_dir / "manifest.json").read_bytes() == before


def test_sufficient_observation_only_marks_evidence_ready_when_sources_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.track_prospective_observations",
        lambda **kwargs: ObservationProgress(
            "OBSERVATION_SUFFICIENT",
            60,
            60,
            {"observation": str(tmp_path / "observation-manifest.json")},
            {"observation": "o" * 64},
            ("shadow-run-1",),
        ),
    )
    atomic_write_json(tmp_path / "observation-manifest.json", {"status": "success"})
    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.resolve_promotion_evidence_references",
        lambda **kwargs: (True, {}, {}, ()),
    )

    result = service.run(REQUEST_ID)

    assert result.current_state == "EVIDENCE_READY"
    snapshot = service.storage.read(result.lifecycle_run_id)
    assert snapshot is not None
    assert snapshot.summary.promotion_evidence_status == "READY_FOR_PREPARATION"
    assert not (tmp_path / "models/promotion_requests").exists()


def test_atomic_update_failure_preserves_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    first = service.run(REQUEST_ID, stop_after="readiness")
    before = (first.output_dir / "manifest.json").read_bytes()
    from ashare_quant.retraining.orchestration import storage as storage_module

    original_replace = storage_module.os.replace
    calls = 0

    class OsProxy:
        def replace(self, source: object, target: object) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("forced lifecycle publication failure")
            original_replace(source, target)

    monkeypatch.setattr(storage_module, "os", OsProxy())

    with pytest.raises(OSError, match="forced lifecycle"):
        service.run(REQUEST_ID, stop_after="training")

    assert (first.output_dir / "manifest.json").read_bytes() == before
    assert service.storage.read(first.lifecycle_run_id) is not None


def test_lifecycle_cli_delegates_without_business_logic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeLifecycle:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, request_id: str, *, stop_after: str | None = None):
            from ashare_quant.retraining.orchestration.schemas import LifecycleRunResult

            return LifecycleRunResult(
                "lifecycle-1", request_id, "READINESS_READY", None, tmp_path / "output"
            )

    monkeypatch.setattr("ashare_quant.cli.load_settings", lambda path: make_settings(tmp_path))
    monkeypatch.setattr("ashare_quant.cli.RetrainingLifecycleOrchestrator", FakeLifecycle)

    code = main(
        [
            "retraining",
            "lifecycle-run",
            "--request-id",
            REQUEST_ID,
            "--stop-after",
            "readiness",
        ]
    )

    assert code == 0
    assert "retraining_lifecycle:" in capsys.readouterr().out


def test_lifecycle_never_calls_promotion_registry_trading_or_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ashare_quant.models.promotion.apply import PromotionApplyService
    from ashare_quant.models.promotion.approval import HumanReviewService
    from ashare_quant.models.promotion.rollback import RollbackService
    from ashare_quant.models.registry import ModelRegistry
    from ashare_quant.paper_trading import PaperTradingService
    from ashare_quant.strategy import CandidateSelector

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden governance or trading API called")

    monkeypatch.setattr(PromotionApplyService, "apply", forbidden)
    monkeypatch.setattr(HumanReviewService, "approve", forbidden)
    monkeypatch.setattr(RollbackService, "apply", forbidden)
    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)
    monkeypatch.setattr(CandidateSelector, "select", forbidden)
    service, _ = lifecycle_service(tmp_path, monkeypatch)

    result = service.run(REQUEST_ID)

    assert result.current_state == "OBSERVATION_PENDING"


def lifecycle_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    readiness_status: str = "READY",
    failure: str | None = None,
) -> tuple[RetrainingLifecycleOrchestrator, list[str]]:
    settings = make_settings(tmp_path)
    config = tmp_path / "config/default.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("environment: test\n", encoding="utf-8")
    calls: list[str] = []
    readiness = FakeReadiness(tmp_path, calls, readiness_status)
    execution = FakeExecution(tmp_path, calls, failure)
    validation = FakeValidation(tmp_path, calls, failure)
    shadow = FakeShadow(tmp_path, calls, failure)
    service = RetrainingLifecycleOrchestrator(
        settings=settings,
        config_path=config,
        retraining_policy_path=Path("config/retraining_policy.yaml"),
        promotion_policy_path=Path("config/promotion_policy.yaml"),
        readiness=readiness,
        execution=execution,
        validation=validation,
        shadow=shadow,
        now=lambda: NOW,
    )
    frozen = lifecycle_input(service)
    monkeypatch.setattr(service, "_input", lambda request_id: frozen)
    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.track_prospective_observations",
        lambda **kwargs: ObservationProgress("OBSERVATION_PENDING", 0, 60, {}, {}, ()),
    )
    return service, calls


class FakeReadiness:
    def __init__(self, root: Path, calls: list[str], status: str) -> None:
        self.root = root
        self.calls = calls
        self.status = status

    def validate(self, as_of: str, *, request_id: str | None = None) -> ReadinessResult:
        self.calls.append("readiness")
        output = self.root / "reports/retraining/readiness" / as_of
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output / "manifest.json", {"status": self.status})
        check_status = "PASS" if self.status == "READY" else "FAIL"
        report = RetrainingReadinessReport(
            run_id="readiness-run-1",
            as_of=as_of,
            request_id=request_id,
            status=self.status,
            checks={"scheduler": check_status},
            check_details=(ReadinessCheck(name="scheduler", status=check_status, message="test"),),
            production_run_id="production-run-1" if self.status == "READY" else None,
            governance_snapshot_hash="g" * 64,
            promotion_policy_hash="p" * 64,
            request_hash="r" * 64,
            feature_hash="f" * 64,
            universe_hash="u" * 64,
            label_hash="l" * 64,
        )
        return ReadinessResult(report, output)


class FakeExecution:
    def __init__(self, root: Path, calls: list[str], failure: str | None) -> None:
        self.root = root
        self.calls = calls
        self.failure = failure

    def execute(self, request_id: str) -> ExecutionResult:
        self.calls.append("training")
        if self.failure == "training":
            raise RuntimeError("training fixture failure")
        artifact = self.root / "models/challengers" / MODEL_ID
        execution = self.root / "reports/retraining/executions/training-run-1"
        registration = self.root / "models/candidate_registrations" / MODEL_ID
        atomic_write_json(artifact / "manifest.json", {"model_id": MODEL_ID})
        atomic_write_json(execution / "manifest.json", {"training_run_id": "training-run-1"})
        record = CandidateRegistration(
            model_id=MODEL_ID,
            candidate_registration_id="candidate-registration-1",
            training_run_id="training-run-1",
            artifact_path=str(artifact),
            artifact_hash="a" * 64,
            feature_hash="f" * 64,
            horizon=10,
        )
        atomic_write_json(registration / "registration.json", record.model_dump(mode="json"))
        return ExecutionResult("training-run-1", MODEL_ID, "COMPLETED", execution, artifact)


class FakeValidation:
    def __init__(self, root: Path, calls: list[str], failure: str | None) -> None:
        self.root = root
        self.calls = calls
        self.failure = failure

    def validate(self, model_id: str) -> RetrainingValidationResult:
        self.calls.append("validation")
        if self.failure == "validation":
            raise RuntimeError("validation fixture failure")
        output = self.root / "reports/retraining_validation/validation-run-1"
        offline = output / "offline/metrics.json"
        executable = output / "executable/summary.json"
        eligibility = output / "shadow/eligibility.json"
        atomic_write_json(offline, {"status": "PASS"})
        atomic_write_json(executable, {"status": "PASS"})
        atomic_write_json(eligibility, {"shadow_eligible": True})
        atomic_write_json(
            output / "manifest.json",
            {
                "model_id": model_id,
                "training_run_id": "training-run-1",
                "offline_validation_hash": file_sha256(offline),
                "executable_validation_hash": file_sha256(executable),
            },
        )
        return RetrainingValidationResult("validation-run-1", model_id, "COMPLETED", True, output)


class FakeShadow:
    def __init__(self, root: Path, calls: list[str], failure: str | None) -> None:
        self.root = root
        self.calls = calls
        self.failure = failure

    def predict(self, model_id: str, *, as_of: str | None = None) -> RetrainedShadowResult:
        self.calls.append("shadow")
        if self.failure == "shadow":
            raise RuntimeError("shadow fixture failure")
        date = as_of or AS_OF
        output = self.root / f"reports/shadow_predictions/{date}/retrained/{model_id}"
        atomic_write_json(
            output / "manifest.json",
            {
                "model_origin": "retrained_challenger",
                "training_run_id": "training-run-1",
                "validation_run_id": "validation-run-1",
                "access_policy": "prospective_production",
                "production_run_id": "production-run-1",
            },
        )
        return RetrainedShadowResult(model_id, date, "shadow-run-1", 10, output)


def lifecycle_input(service: RetrainingLifecycleOrchestrator) -> LifecycleInput:
    reference = EvidenceReference(
        path="model_monitor/20260731/manifest.json",
        sha256="e" * 64,
        artifact_name="fixture",
        as_of=AS_OF,
    )
    request = TrainingRequest(
        request_id=REQUEST_ID,
        created_at=NOW.isoformat(),
        as_of=AS_OF,
        target_models=(TrainingTarget(model_id=PARENT_ID, model_role="champion", horizon=10),),
        trigger_reason=("manual_request",),
        evidence=RetrainingEvidence(
            monitor_snapshot=reference,
            performance_observation=reference,
            alerts=reference,
        ),
        evidence_hash="e" * 64,
        policy_hash=service.retraining_policy.policy_hash,
        policy_version=service.retraining_policy.policy_version,
        promotion_policy_hash=service.promotion_policy.policy_hash,
        promotion_policy_version=service.promotion_policy.policy_version,
        generation_mode="manual",
    )
    return LifecycleInput(
        request,
        "r" * 64,
        service.retraining_policy.policy_hash,
        service.retraining_policy.lifecycle_policy_hash,
        service.promotion_policy.policy_hash,
    )


def observation_fixture(root: Path, *, sessions: int) -> None:
    reports = root / "reports/performance_observation"
    for index in range(sessions):
        date = f"202607{index + 1:02d}"
        output = reports / date
        rows = pd.DataFrame(
            [
                observation_row(date, "retrained_challenger", "available", MODEL_ID),
                observation_row(date, "retrained_challenger", "entry_not_buyable", MODEL_ID),
                observation_row(date, "research_challenger", "available", MODEL_ID),
                observation_row(date, "champion", "available", "champion-model"),
            ]
        )
        from ashare_quant.monitoring.performance_observation.storage import (
            logical_observation_hash,
            publish_observation_artifact,
        )

        manifest = {
            "schema_version": 1,
            "artifact_name": "performance_observation",
            "observation_as_of": date,
            "observation_hash": logical_observation_hash(rows),
            "access_policy": "prospective_production",
            "contracts": {
                "historical_predictions_used": False,
                "inference_called": False,
                "backtest_called": False,
                "paper_trading_called": False,
                "registry_modified": False,
                "labels_used_only_after_maturity": True,
            },
        }
        publish_observation_artifact(
            output_dir=output,
            observations=rows,
            metrics={},
            manifest=manifest,
        )


def observation_row(date: str, origin: str, status: str, model_id: str) -> dict[str, object]:
    from ashare_quant.monitoring.performance_observation.schemas import OBSERVATION_COLUMNS

    code_index = {
        ("retrained_challenger", "available"): 1,
        ("retrained_challenger", "entry_not_buyable"): 2,
        ("research_challenger", "available"): 3,
        ("champion", "available"): 4,
    }[(origin, status)]
    record: dict[str, object] = {
        "observation_id": f"{date}-{origin}-{status}-{model_id}",
        "signal_date": date,
        "observation_as_of": date,
        "model_id": model_id,
        "model_role": "challenger_h10" if origin != "champion" else "champion",
        "model_origin": origin,
        "horizon": 10,
        "ts_code": f"{code_index:06d}.SZ",
        "prediction_score": 0.5,
        "rank": 1,
        "score_percentile": 1.0,
        "future_excess_ret": 0.01 if status == "available" else None,
        "entry_date": date,
        "exit_date": date,
        "label_status": status,
        "feature_hash": "f" * 64,
        "universe_hash": "u" * 64,
        "prediction_hash": "p" * 64,
        "production_run_id": "production-run-1",
        "shadow_run_id": "shadow-run-1",
        "parent_model_id": PARENT_ID if origin == "retrained_challenger" else "",
        "training_request_id": REQUEST_ID if origin == "retrained_challenger" else "",
        "training_run_id": "training-run-1" if origin == "retrained_challenger" else "",
        "validation_run_id": "validation-run-1" if origin == "retrained_challenger" else "",
    }
    return {column: record[column] for column in OBSERVATION_COLUMNS}


def make_settings(root: Path) -> AppSettings:
    return AppSettings.model_validate(
        {
            "paths": PathSettings(
                raw_data=root / "raw",
                processed_data=root / "processed",
                parquet_store=root / "parquet",
                duckdb_path=root / "test.duckdb",
                reports=root / "reports",
                models=root / "models",
                backtests=root / "backtests",
                paper_trading=root / "paper_trading",
                data_quality_logs=root / "logs",
            ).model_dump(mode="python")
        }
    )


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
