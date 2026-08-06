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
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.orchestration.lock import detect_production_lock_owner
from ashare_quant.retraining.execution.schemas import (
    CandidateRegistration,
    ExecutionResult,
    QualificationExecutionContext,
)
from ashare_quant.retraining.orchestration.controls import LifecycleOperationalControls
from ashare_quant.retraining.orchestration.dry_run import LifecycleDryRunService
from ashare_quant.retraining.orchestration.lifecycle import require_transition
from ashare_quant.retraining.orchestration.schemas import (
    LifecycleEvent,
    LifecycleInput,
    LifecycleSnapshot,
    LifecycleSummary,
    ObservationProgress,
)
from ashare_quant.retraining.orchestration.service import RetrainingLifecycleOrchestrator
from ashare_quant.retraining.orchestration.stages import (
    latest_successful_shadow_path,
    resolve_promotion_evidence_references,
)
from ashare_quant.retraining.orchestration.storage import LifecycleStorage
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
    assert calls == ["readiness", "training", "validation", "shadow"]


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
        accepted_shadow_run_ids=("shadow-run-1",),
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
        lambda **kwargs: (True, {}, {}, (), ()),
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


def test_failed_shadow_refresh_preserves_successful_enrollment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    first = service.run(REQUEST_ID)
    before = service.storage.read(first.lifecycle_run_id)
    assert before is not None
    assert before.stage_results["shadow_enrollment"].status == "success"
    assert isinstance(service.shadow, FakeShadow)
    service.shadow.failure = "shadow"

    resumed = service.resume(first.lifecycle_run_id)
    after = service.storage.read(first.lifecycle_run_id)

    assert resumed.current_state == "OBSERVATION_PENDING"
    assert after is not None
    assert after.stage_results["shadow_enrollment"] == before.stage_results["shadow_enrollment"]
    assert after.summary.successful_shadow_run_ids == ("shadow-run-1",)
    assert latest_successful_shadow_path(after.stage_results).is_file()
    assert any("Shadow refresh failed" in event.message for event in after.events)
    assert any(name.startswith("shadow_refresh_attempt:") for name in after.stage_results)


def test_policy_drift_keeps_identity_and_explicit_revalidation_never_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.track_prospective_observations",
        lambda **kwargs: ObservationProgress(
            "OBSERVATION_SUFFICIENT",
            60,
            60,
            {"observation": str(tmp_path / "observation-manifest.json")},
            {"observation": file_sha256(tmp_path / "observation-manifest.json")},
            ("shadow-run-1",),
            "20260101",
            "20260331",
            "20260331",
            "a" * 64,
        ),
    )
    atomic_write_json(tmp_path / "observation-manifest.json", {"status": "success"})
    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.resolve_promotion_evidence_references",
        lambda **kwargs: (True, {}, {}, (), ()),
    )
    first = service.run(REQUEST_ID)
    assert first.current_state == "EVIDENCE_READY"
    training_calls = calls.count("training")
    service.promotion_policy = service.promotion_policy.model_copy(
        update={"policy_version": "changed"}
    )

    resumed = service.resume(first.lifecycle_run_id)
    assert resumed.lifecycle_run_id == first.lifecycle_run_id
    assert resumed.current_state == "POLICY_REVIEW_REQUIRED"
    assert calls.count("training") == training_calls

    revalidated = service.revalidate_evidence(first.lifecycle_run_id)
    status = service.status(first.lifecycle_run_id)
    assert revalidated.current_state == "EVIDENCE_READY"
    assert status["evaluated_promotion_policy_hash"] == service.promotion_policy.policy_hash
    assert status["policy_drift"] is True
    assert status["evidence_stale"] is False
    assert calls.count("training") == training_calls


def test_observation_progress_appends_only_when_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    sessions = {"value": 10}

    def progress(**kwargs: object) -> ObservationProgress:
        count = sessions["value"]
        return ObservationProgress(
            "OBSERVATION_ACCUMULATING",
            count,
            60,
            {},
            {},
            ("shadow-run-1",),
            "20260101",
            f"202601{count:02d}",
            f"202601{count:02d}",
            str(count).zfill(64),
        )

    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.track_prospective_observations",
        progress,
    )
    first = service.run(REQUEST_ID)
    snapshot_10 = service.storage.read(first.lifecycle_run_id)
    assert snapshot_10 is not None
    sessions["value"] = 11
    service.resume(first.lifecycle_run_id)
    snapshot_11 = service.storage.read(first.lifecycle_run_id)
    assert snapshot_11 is not None
    assert len(snapshot_11.events) == len(snapshot_10.events) + 1
    assert snapshot_11.events[-1].details["previous_mature_sessions"] == 10
    assert snapshot_11.events[-1].details["current_mature_sessions"] == 11
    service.resume(first.lifecycle_run_id)
    unchanged = service.storage.read(first.lifecycle_run_id)
    assert unchanged is not None
    assert unchanged.events == snapshot_11.events


def test_budget_uses_shanghai_date_and_counts_failed_attempts(tmp_path: Path) -> None:
    storage = LifecycleStorage(tmp_path / "reports")
    snapshot = control_snapshot(
        "prior",
        parent=PARENT_ID,
        horizon=10,
        timestamps=("2026-08-01T16:10:00+00:00", "2026-08-01T16:20:00+00:00"),
    )
    controls = LifecycleOperationalControls(
        storage=storage,
        timezone="Asia/Shanghai",
        max_daily_training_runs=2,
        cooldown_days=30,
        now=datetime.fromisoformat("2026-08-01T16:30:00+00:00"),
    )
    controls._snapshots = lambda: (snapshot,)  # type: ignore[method-assign]

    decision = controls.budget()

    assert decision.operational_date == "2026-08-02"
    assert decision.observed_attempts_before == 2
    assert decision.allowed is False


def test_cooldown_is_parent_and_horizon_specific(tmp_path: Path) -> None:
    storage = LifecycleStorage(tmp_path / "reports")
    previous = control_snapshot(
        "prior",
        parent=PARENT_ID,
        horizon=10,
        timestamps=("2026-07-25T02:00:00+00:00",),
    )
    controls = LifecycleOperationalControls(
        storage=storage,
        timezone="Asia/Shanghai",
        max_daily_training_runs=5,
        cooldown_days=30,
        now=datetime.fromisoformat("2026-08-01T02:00:00+00:00"),
    )
    controls._snapshots = lambda: (previous,)  # type: ignore[method-assign]

    blocked = controls.cooldown(lifecycle_run_id="new", parent_model_id=PARENT_ID, horizon=10)
    other_horizon = controls.cooldown(lifecycle_run_id="new", parent_model_id=PARENT_ID, horizon=20)
    same_run = controls.cooldown(lifecycle_run_id="prior", parent_model_id=PARENT_ID, horizon=10)

    assert blocked.allowed is False
    assert blocked.cooldown_expiry_date == "2026-08-24"
    assert other_horizon.allowed is True
    assert same_run.allowed is True


def test_corrupted_budget_history_fails_closed(tmp_path: Path) -> None:
    storage = LifecycleStorage(tmp_path / "reports")
    (storage.root / "broken").mkdir(parents=True)
    controls = LifecycleOperationalControls(
        storage=storage,
        timezone="Asia/Shanghai",
        max_daily_training_runs=1,
        cooldown_days=30,
        now=NOW,
    )

    with pytest.raises(DataValidationError, match="recovery required"):
        controls.budget()


def test_lifecycle_dry_run_is_deterministic_and_does_not_execute_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch)
    request_path = service.request_storage.requests_root / REQUEST_ID / "training_request.json"
    atomic_write_json(request_path, {"request_id": REQUEST_ID})
    dry_run = LifecycleDryRunService(service)

    first = dry_run.run(REQUEST_ID)
    second = dry_run.run(REQUEST_ID)

    assert first.status == "READY_TO_EXECUTE"
    assert second.dry_run_id == first.dry_run_id
    assert second.idempotent is True
    assert calls == ["readiness", "readiness"]
    assert not (tmp_path / "models/challengers" / MODEL_ID).exists()
    manifest = _json(first.output_dir / "manifest.json")
    assert manifest["manifest_written_last"] is True


def test_lifecycle_dry_run_reports_failed_readiness_as_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls = lifecycle_service(tmp_path, monkeypatch, readiness_status="FAILED")
    request_path = service.request_storage.requests_root / REQUEST_ID / "training_request.json"
    atomic_write_json(request_path, {"request_id": REQUEST_ID})

    result = LifecycleDryRunService(service).run(REQUEST_ID)

    assert result.status == "BLOCKED"
    assert calls == ["readiness"]
    assert not (tmp_path / "models/challengers" / MODEL_ID).exists()


def test_resume_uses_frozen_identity_without_calling_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    first = service.run(REQUEST_ID, stop_after="training")
    (tmp_path / "config/default.yaml").write_text("environment: changed\n", encoding="utf-8")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume recursively called lifecycle-run")

    monkeypatch.setattr(service, "run", forbidden)
    resumed = service.resume(first.lifecycle_run_id)

    assert resumed.lifecycle_run_id == first.lifecycle_run_id
    assert resumed.current_state == "OBSERVATION_PENDING"


def test_changed_or_removed_observation_evidence_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _ = lifecycle_service(tmp_path, monkeypatch)
    source = tmp_path / "observation.json"
    atomic_write_json(source, {"version": 1})
    current = {
        "progress": ObservationProgress(
            "OBSERVATION_ACCUMULATING",
            10,
            60,
            {"observation": str(source)},
            {"observation": file_sha256(source)},
            ("shadow-run-1",),
            "20260101",
            "20260110",
            "20260110",
            "a" * 64,
        )
    }
    monkeypatch.setattr(
        "ashare_quant.retraining.orchestration.service.track_prospective_observations",
        lambda **kwargs: current["progress"],
    )
    first = service.run(REQUEST_ID)
    atomic_write_json(source, {"version": 2})

    with pytest.raises(DataValidationError, match="source hash changed"):
        service.resume(first.lifecycle_run_id)


def test_exact_evidence_rejects_unrelated_paper_portfolio(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    execution = tmp_path / "execution.json"
    validation = tmp_path / "validation.json"
    observation = reports / "performance_observation/20260731/manifest.json"
    shadow = reports / f"shadow_predictions/20260731/retrained/{MODEL_ID}/manifest.json"
    performance = reports / "model_monitor/20260731/performance/manifest.json"
    alerts = reports / "model_monitor/20260731/alerts/manifest.json"
    monitor = reports / "model_monitor/20260731/manifest.json"
    for path, payload in (
        (execution, {"training_run_id": "training-run-1"}),
        (validation, {"validation_run_id": "validation-run-1"}),
        (observation, {"observation_as_of": "20260731"}),
    ):
        atomic_write_json(path, payload)
    atomic_write_json(
        shadow,
        {
            "model_id": MODEL_ID,
            "model_origin": "retrained_challenger",
            "training_request_id": REQUEST_ID,
            "training_run_id": "training-run-1",
            "validation_run_id": "validation-run-1",
            "production_run_id": "production-run-1",
            "shadow_run_id": "shadow-run-1",
            "access_policy": "prospective_production",
            "models": [{"model_id": MODEL_ID, "native_horizon": 10}],
        },
    )
    atomic_write_json(
        performance,
        {
            "artifact_name": "performance_monitor",
            "as_of": "20260731",
            "models": [
                {
                    "model_id": MODEL_ID,
                    "model_origin": "retrained_challenger",
                    "horizon": 10,
                    "training_run_id": "training-run-1",
                    "validation_run_id": "validation-run-1",
                }
            ],
        },
    )
    atomic_write_json(
        alerts,
        {
            "as_of": "20260731",
            "source_metrics": [
                {"path": "performance/manifest.json", "hash": file_sha256(performance)}
            ],
        },
    )
    atomic_write_json(
        monitor,
        {
            "artifact_name": "production_monitor_manifest",
            "as_of": "20260731",
            "monitor_metric_file_hashes": {"performance_manifest": file_sha256(performance)},
        },
    )
    atomic_write_json(
        reports / "paper_trading_daily/20260731/manifest.json",
        {"model_id": "unrelated-champion", "horizon": 10},
    )
    progress = ObservationProgress(
        "OBSERVATION_SUFFICIENT",
        60,
        60,
        {"performance_observation:20260731": str(observation)},
        {"performance_observation:20260731": file_sha256(observation)},
        ("shadow-run-1",),
        "20260501",
        "20260731",
        "20260731",
        "o" * 64,
    )

    ready, _, _, warnings, references = resolve_promotion_evidence_references(
        reports_root=reports,
        lifecycle_run_id="lifecycle-1",
        request_id=REQUEST_ID,
        model_id=MODEL_ID,
        parent_model_id=PARENT_ID,
        horizon=10,
        training_run_id="training-run-1",
        validation_run_id="validation-run-1",
        execution_path=execution,
        validation_path=validation,
        shadow_path=shadow,
        observation=progress,
        policy=PromotionGatePolicy(require={"paper_trading": True}),
    )

    assert ready is False
    assert "missing policy-required retrained Challenger paper-trading evidence" in warnings
    assert all(reference.lifecycle_run_id == "lifecycle-1" for reference in references)


def control_snapshot(
    run_id: str, *, parent: str, horizon: int, timestamps: tuple[str, ...]
) -> LifecycleSnapshot:
    events = tuple(
        LifecycleEvent(
            sequence=index,
            state="TRAINING",
            created_at=value,
            message="attempt",
        )
        for index, value in enumerate(timestamps, start=1)
    )
    summary = LifecycleSummary(
        lifecycle_run_id=run_id,
        request_id=f"request-{run_id}",
        model_id=None,
        parent_model_id=parent,
        horizon=horizon,  # type: ignore[arg-type]
        trigger_reasons=("manual_request",),
        current_state="TRAINING",
        readiness_run_id="readiness",
        required_sessions=60,
        created_at=timestamps[0],
        updated_at=timestamps[-1],
    )
    return LifecycleSnapshot(summary, events, {})


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

    def execute(
        self,
        request_id: str,
        *,
        qualification: QualificationExecutionContext | None = None,
    ) -> ExecutionResult:
        self.calls.append("training")
        assert (
            detect_production_lock_owner(self.root / "runs/.retraining-lifecycle.lock") is not None
        )
        if self.failure == "training":
            raise RuntimeError("training fixture failure")
        artifact = self.root / "models/challengers" / MODEL_ID
        execution = self.root / "reports/retraining/executions/training-run-1"
        registration = self.root / "models/candidate_registrations" / MODEL_ID
        atomic_write_json(
            artifact / "manifest.json",
            {
                "model_id": MODEL_ID,
                "qualification_run_id": (
                    qualification.qualification_run_id if qualification else None
                ),
                "qualification_only": qualification is not None,
            },
        )
        atomic_write_json(execution / "manifest.json", {"training_run_id": "training-run-1"})
        record = CandidateRegistration(
            model_id=MODEL_ID,
            candidate_registration_id="candidate-registration-1",
            training_run_id="training-run-1",
            artifact_path=str(artifact),
            artifact_hash="a" * 64,
            feature_hash="f" * 64,
            horizon=10,
            qualification_run_id=(qualification.qualification_run_id if qualification else None),
            qualification_only=qualification is not None,
            qualification_phase=(qualification.qualification_phase if qualification else None),
            promotion_forbidden=qualification is not None,
            trading_forbidden=qualification is not None,
        )
        atomic_write_json(registration / "registration.json", record.model_dump(mode="json"))
        return ExecutionResult("training-run-1", MODEL_ID, "COMPLETED", execution, artifact)


class FakeValidation:
    def __init__(self, root: Path, calls: list[str], failure: str | None) -> None:
        self.root = root
        self.calls = calls
        self.failure = failure

    def validate(
        self,
        model_id: str,
        *,
        qualification: QualificationExecutionContext | None = None,
    ) -> RetrainingValidationResult:
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
                "qualification_run_id": (
                    qualification.qualification_run_id if qualification else None
                ),
                "qualification_only": qualification is not None,
            },
        )
        return RetrainingValidationResult("validation-run-1", model_id, "COMPLETED", True, output)


class FakeShadow:
    def __init__(self, root: Path, calls: list[str], failure: str | None) -> None:
        self.root = root
        self.calls = calls
        self.failure = failure

    def predict(
        self,
        model_id: str,
        *,
        as_of: str | None = None,
        qualification: QualificationExecutionContext | None = None,
    ) -> RetrainedShadowResult:
        self.calls.append("shadow")
        if self.failure == "shadow":
            raise RuntimeError("shadow fixture failure")
        date = as_of or AS_OF
        output = (
            self.root
            / f"reports/shadow_predictions/{date}"
            / (
                f"qualification/{qualification.qualification_run_id}/{model_id}"
                if qualification
                else f"retrained/{model_id}"
            )
        )
        atomic_write_json(
            output / "manifest.json",
            {
                "model_id": model_id,
                "model_origin": "retrained_challenger",
                "training_request_id": REQUEST_ID,
                "training_run_id": "training-run-1",
                "validation_run_id": "validation-run-1",
                "access_policy": "prospective_production",
                "production_run_id": "production-run-1",
                "shadow_run_id": "shadow-run-1",
                "generated_at": NOW.isoformat(),
                "models": [{"model_id": model_id, "native_horizon": 10}],
                "qualification_run_id": (
                    qualification.qualification_run_id if qualification else None
                ),
                "qualification_only": qualification is not None,
                "promotion_forbidden": qualification is not None,
                "trading_forbidden": qualification is not None,
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
        "qualification_run_id": "",
        "qualification_only": False,
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
