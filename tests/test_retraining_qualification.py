"""Fixture-only tests for controlled operational qualification."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.evidence_resolver import PromotionEvidenceResolver
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.monitoring.performance_observation.storage import (
    logical_observation_hash,
    publish_observation_artifact,
)
from ashare_quant.retraining.orchestration.dry_run import LifecycleDryRunResult
from ashare_quant.retraining.qualification.authorization import (
    QualificationAuthorizationConflictError,
)
from ashare_quant.retraining.qualification.authorization_schemas import AuthorizationStage
from ashare_quant.retraining.qualification.lifecycle import require_qualification_transition
from ashare_quant.retraining.qualification.schemas import QualificationCheck
from ashare_quant.retraining.qualification.service import OperationalQualificationService
from ashare_quant.utils.manifest import atomic_write_json, config_hash
from test_retraining_lifecycle import (
    AS_OF,
    MODEL_ID,
    NOW,
    REQUEST_ID,
    lifecycle_input,
    lifecycle_service,
    observation_row,
)


class FakeDryRun:
    def __init__(
        self, root: Path, proposed_lifecycle_run_id: str, status: str = "READY_TO_EXECUTE"
    ) -> None:
        self.root = root
        self.proposed_lifecycle_run_id = proposed_lifecycle_run_id
        self.status = status
        self.calls = 0

    def run(self, request_id: str, *, as_of: str | None = None) -> LifecycleDryRunResult:
        self.calls += 1
        output = self.root / "reports/retraining/lifecycle_dry_runs/dry-fixture"
        atomic_write_json(
            output / "dry_run.json",
            {
                "request_id": request_id,
                "as_of": as_of,
                "proposed_lifecycle_run_id": self.proposed_lifecycle_run_id,
                "no_mutation_confirmed": True,
                "cooldown_status": "PASS",
                "budget_status": "PASS",
                "lock_status": "AVAILABLE",
                "source_hashes": {"training_request": "r" * 64},
            },
        )
        atomic_write_json(
            output / "manifest.json",
            {
                "status": "READY_TO_EXECUTE",
                "dry_run_sha256": file_sha256(output / "dry_run.json"),
            },
        )
        return LifecycleDryRunResult("dry-fixture", self.status, output, self.calls > 1)


class QualificationReadiness:
    def __init__(self, delegate: object, hashes: dict[str, str]) -> None:
        self.delegate = delegate
        self.hashes = hashes

    def validate(self, as_of: str, *, request_id: str | None = None):
        result = self.delegate.validate(as_of, request_id=request_id)  # type: ignore[attr-defined]
        report = result.report.model_copy(
            update={
                "feature_hash": self.hashes["feature_manifest"],
                "universe_hash": self.hashes["universe_manifest"],
                "label_hash": self.hashes["label_manifest"],
            }
        )
        return type(result)(report, result.output_dir, result.idempotent)


def qualification_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    real_training: bool = False,
    real_shadow: bool = False,
    failure: str | None = None,
    dry_run_status: str = "READY_TO_EXECUTE",
) -> tuple[OperationalQualificationService, list[str], FakeDryRun]:
    lifecycle, calls = lifecycle_service(tmp_path, monkeypatch, failure=failure)
    frozen = replace(
        lifecycle_input(lifecycle), frozen_config_hash=config_hash(lifecycle.config_path)
    )
    monkeypatch.setattr(lifecycle, "_input", lambda request_id: frozen)
    monkeypatch.setattr(
        lifecycle,
        "frozen_input_for_request",
        lambda request_id, **kwargs: frozen,
    )
    proposed, _ = lifecycle.proposed_identity(REQUEST_ID)
    dry_run = FakeDryRun(tmp_path, proposed, dry_run_status)
    service = OperationalQualificationService(
        settings=lifecycle.settings,
        config_path=lifecycle.config_path,
        retraining_policy_path=Path("config/retraining_policy.yaml"),
        promotion_policy_path=Path("config/promotion_policy.yaml"),
        lifecycle=lifecycle,
        dry_run=dry_run,
        readiness=lifecycle.readiness,
        execution=lifecycle.execution,
        validation=lifecycle.validation,
        shadow=lifecycle.shadow,
        now=lifecycle.now,
    )
    service.policy = service.policy.model_copy(
        update={"allow_real_training": real_training, "allow_real_shadow": real_shadow}
    )
    sources = {
        name: tmp_path / f"{name}.json"
        for name in ("fixture", "feature_manifest", "universe_manifest", "label_manifest")
    }
    for name, source in sources.items():
        atomic_write_json(source, {"fixture": name})
    source_hashes = {name: file_sha256(path) for name, path in sources.items()}
    service.readiness = QualificationReadiness(service.readiness, source_hashes)
    monkeypatch.setattr(
        "ashare_quant.retraining.qualification.service.run_preflight",
        lambda **kwargs: (
            (QualificationCheck(name="fixture", status="PASS", message="validated"),),
            {
                name: {"path": str(path), "sha256": source_hashes[name]}
                for name, path in sources.items()
            },
        ),
    )
    return service, calls, dry_run


def authorize_stage(
    service: OperationalQualificationService, run_id: str, stage: AuthorizationStage
) -> str:
    result = service.authorize(
        run_id,
        stage=stage,
        approved_by="fixture-operator",
        reason=f"fixture {stage} authorization",
    )
    return result.authorization_id


def set_capabilities(
    service: OperationalQualificationService,
    *,
    training: bool | None = None,
    shadow: bool | None = None,
) -> None:
    updates = {}
    if training is not None:
        updates["allow_real_training"] = training
    if shadow is not None:
        updates["allow_real_shadow"] = shadow
    service.policy = service.policy.model_copy(update=updates)
    service.authorization.policy = service.policy


def test_qualification_state_machine_rejects_skipped_stages() -> None:
    require_qualification_transition("CREATED", "PREFLIGHT_CHECKING")
    with pytest.raises(DataValidationError, match="forbidden"):
        require_qualification_transition("CREATED", "TRAINING")
    with pytest.raises(DataValidationError, match="forbidden"):
        require_qualification_transition("TRAINING_COMPLETED", "SHADOW_ENROLLING")


def test_start_reuses_dry_run_and_readiness_then_stops_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, dry_run = qualification_service(tmp_path, monkeypatch)

    result = service.start(REQUEST_ID, as_of=AS_OF)

    assert result.state == "TRAINING_PENDING_APPROVAL"
    assert dry_run.calls == 1
    assert calls == ["readiness"]
    snapshot = service.storage.read(result.qualification_run_id)
    assert snapshot is not None
    assert snapshot.summary.qualification_only is True
    assert snapshot.checkpoints["dry_run"].status == "success"
    assert snapshot.checkpoints["readiness"].status == "success"
    assert not (tmp_path / "models/challengers" / MODEL_ID).exists()


def test_start_is_deterministic_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    first = service.start(REQUEST_ID, as_of=AS_OF)
    before = (first.output_dir / "qualification_events.parquet").read_bytes()

    second = service.start(REQUEST_ID, as_of=AS_OF)

    assert second.qualification_run_id == first.qualification_run_id
    assert second.idempotent is True
    assert (first.output_dir / "qualification_events.parquet").read_bytes() == before


def test_blocked_dry_run_prevents_readiness_and_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch, dry_run_status="BLOCKED")

    result = service.start(REQUEST_ID, as_of=AS_OF)

    assert result.state == "DRY_RUN_BLOCKED"
    assert calls == []


def test_training_is_disabled_by_default_and_cli_intent_cannot_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)

    result = service.advance(started.qualification_run_id, target="training")

    assert result.state == "TRAINING_PENDING_APPROVAL"
    assert "training" not in calls


def test_cuda_probe_failure_cannot_be_bypassed_by_qualification_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch, real_training=True)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorization_id = authorize_stage(service, started.qualification_run_id, "training")
    monkeypatch.setattr(
        "ashare_quant.retraining.qualification.service.resolve_training_backend",
        lambda _settings: (_ for _ in ()).throw(DataValidationError("CUDA unavailable")),
    )

    with pytest.raises(DataValidationError, match="CUDA unavailable"):
        service.advance(started.qualification_run_id, target="training")

    assert calls == ["readiness"]
    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert status.authorization_id == authorization_id
    assert status.status == "ACTIVE"


def test_full_fixture_qualification_is_explicit_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, real_shadow=True
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    trained = service.advance(started.qualification_run_id, target="training")
    validated = service.advance(started.qualification_run_id, target="validation")
    authorize_stage(service, started.qualification_run_id, "shadow")
    shadowed = service.advance(started.qualification_run_id, target="shadow")
    observed = service.advance(started.qualification_run_id, target="observation")

    assert trained.state == "VALIDATION_PENDING_APPROVAL"
    assert validated.state == "SHADOW_PENDING_APPROVAL"
    assert shadowed.state == "SHADOW_ENROLLED"
    assert observed.state == "QUALIFIED"
    assert calls == ["readiness", "training", "validation", "shadow"]
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None
    assert snapshot.summary.observation_status == "OBSERVATION_PENDING"
    assert snapshot.invariant_results["changed"] == []
    artifact = json.loads(
        (tmp_path / "models/challengers" / MODEL_ID / "manifest.json").read_text()
    )
    assert artifact["qualification_only"] is True
    assert artifact["qualification_run_id"] == started.qualification_run_id
    sidecar = (
        tmp_path
        / f"reports/shadow_predictions/{AS_OF}/qualification"
        / started.qualification_run_id
        / MODEL_ID
        / "manifest.json"
    )
    assert sidecar.is_file()
    assert not (tmp_path / f"reports/shadow_predictions/{AS_OF}/retrained/{MODEL_ID}").exists()


def test_observation_qualification_accepts_only_exact_manifest_valid_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, real_shadow=True
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    service.advance(started.qualification_run_id, target="training")
    service.advance(started.qualification_run_id, target="validation")
    authorize_stage(service, started.qualification_run_id, "shadow")
    service.advance(started.qualification_run_id, target="shadow")
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None

    exact = observation_row(AS_OF, "retrained_challenger", "available", MODEL_ID)
    exact.update(
        {
            "qualification_run_id": started.qualification_run_id,
            "qualification_only": True,
            "training_run_id": snapshot.summary.training_run_id,
            "validation_run_id": snapshot.summary.validation_run_id,
            "shadow_run_id": snapshot.summary.shadow_run_id,
        }
    )
    unrelated = {**exact, "observation_id": "unrelated", "ts_code": "000002.SZ"}
    unrelated["qualification_run_id"] = "qualification_other"
    observations = pd.DataFrame([exact, unrelated])
    output = tmp_path / f"reports/performance_observation/{AS_OF}"
    publish_observation_artifact(
        output_dir=output,
        observations=observations,
        metrics={},
        manifest={
            "schema_version": 1,
            "artifact_name": "performance_observation",
            "observation_hash": logical_observation_hash(observations),
        },
    )

    result = service.advance(started.qualification_run_id, target="observation")

    assert result.state == "QUALIFIED"
    completed = service.storage.read(started.qualification_run_id)
    assert completed is not None
    assert completed.summary.observation_status == "OBSERVATION_ACCUMULATING"
    assert completed.checkpoints["observation"].metrics["mature_sessions"] == 1
    assert set(completed.checkpoints["observation"].artifact_hashes) == {
        f"{AS_OF}:manifest",
        f"{AS_OF}:parquet",
    }


def test_validation_and_shadow_cannot_be_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch, real_training=True)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    with pytest.raises(DataValidationError, match="cannot skip training"):
        service.advance(started.qualification_run_id, target="validation")
    with pytest.raises(DataValidationError, match="cannot skip validation"):
        service.advance(started.qualification_run_id, target="shadow")


@pytest.mark.parametrize(
    ("failure", "expected", "next_stage"),
    [
        ("training", "TRAINING_FAILED", "validation"),
        ("validation", "VALIDATION_FAILED", "shadow"),
        ("shadow", "SHADOW_FAILED", "observation"),
    ],
)
def test_failed_checkpoint_blocks_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected: str,
    next_stage: str,
) -> None:
    service, _, _ = qualification_service(
        tmp_path,
        monkeypatch,
        real_training=True,
        real_shadow=True,
        failure=failure,
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    result = service.advance(started.qualification_run_id, target="training")
    if failure != "training":
        result = service.advance(started.qualification_run_id, target="validation")
    if failure == "shadow":
        authorize_stage(service, started.qualification_run_id, "shadow")
        result = service.advance(started.qualification_run_id, target="shadow")
    assert result.state == expected
    with pytest.raises(DataValidationError):
        service.advance(started.qualification_run_id, target=next_stage)


def test_shadow_is_disabled_by_default_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, real_shadow=False
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    service.advance(started.qualification_run_id, target="training")
    service.advance(started.qualification_run_id, target="validation")

    result = service.advance(started.qualification_run_id, target="shadow")

    assert result.state == "SHADOW_PENDING_APPROVAL"
    assert "shadow" not in calls


def test_referenced_source_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch, real_training=True)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None
    source = Path(str(snapshot.source_inventory["fixture"]["path"]))
    atomic_write_json(source, {"mutated": True})

    with pytest.raises(DataValidationError, match="qualification source changed"):
        service.advance(started.qualification_run_id, target="training")


def test_protected_artifact_mutation_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch, real_training=True)
    registry = tmp_path / "models/registry.json"
    atomic_write_json(registry, {"schema_version": 1, "models": []})
    started = service.start(REQUEST_ID, as_of=AS_OF)
    atomic_write_json(registry, {"schema_version": 1, "models": [{"changed": True}]})

    with pytest.raises(DataValidationError, match="protected qualification invariants changed"):
        service.advance(started.qualification_run_id, target="training")


def test_cancellation_is_terminal_and_append_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    cancelled = service.cancel(started.qualification_run_id, reason="operator stop")
    assert cancelled.state == "CANCELLED"
    with pytest.raises(DataValidationError, match="terminal"):
        service.advance(started.qualification_run_id, target="training")


def test_recovery_detects_incomplete_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    incomplete = service.storage.output_dir("qualification_incomplete")
    incomplete.mkdir(parents=True)

    recovery = service.recovery("qualification_incomplete")

    assert recovery.status == "ACTION_REQUIRED"
    assert any("incomplete qualification" in issue for issue in recovery.issues)


def test_promotion_resolver_rejects_qualification_only_model(tmp_path: Path) -> None:
    artifact = tmp_path / "models/challengers/qualification-model"
    atomic_write_json(artifact / "manifest.json", {"qualification_only": True})

    with pytest.raises(DataValidationError, match="qualification-only"):
        PromotionEvidenceResolver(
            models_root=tmp_path / "models", reports_root=tmp_path / "reports"
        ).prepare("qualification-model")


def test_runtime_capabilities_do_not_change_static_identity_or_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    static_hash = service.policy.static_policy_hash
    full_hash = service.policy.policy_hash

    set_capabilities(service, training=True, shadow=True)
    repeated = service.start(REQUEST_ID, as_of=AS_OF)

    assert repeated.qualification_run_id == started.qualification_run_id
    assert repeated.idempotent is True
    assert service.policy.static_policy_hash == static_hash
    assert service.policy.policy_hash != full_hash
    status = service.status(started.qualification_run_id)
    assert status["runtime_training_enabled"] is True
    assert status["runtime_shadow_enabled"] is True
    assert status["static_policy_drift"] is False


def test_static_safety_policy_change_still_blocks_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    service.policy = service.policy.model_copy(
        update={
            "allow_real_training": True,
            "allowed_stop_points": tuple(
                point for point in service.policy.allowed_stop_points if point != "observation"
            ),
        }
    )
    service.authorization.policy = service.policy

    with pytest.raises(DataValidationError, match="static policy"):
        service.advance(started.qualification_run_id, target="training")
    assert calls == ["readiness"]


def test_training_requires_capability_authorization_and_explicit_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)

    set_capabilities(service, training=True)
    without_authorization = service.advance(started.qualification_run_id, target="training")
    assert without_authorization.state == "TRAINING_PENDING_APPROVAL"
    assert calls == ["readiness"]

    set_capabilities(service, training=False)
    authorization_id = authorize_stage(service, started.qualification_run_id, "training")
    disabled = service.advance(started.qualification_run_id, target="training")
    assert disabled.state == "TRAINING_PENDING_APPROVAL"
    assert (
        service.authorization_status(started.qualification_run_id, stage="training")[
            0
        ].authorization_id
        == authorization_id
    )

    set_capabilities(service, training=True)
    completed = service.advance(started.qualification_run_id, target="training")
    assert completed.state == "VALIDATION_PENDING_APPROVAL"
    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert status.status == "CONSUMED"
    assert status.authorization_id == authorization_id
    assert calls == ["readiness", "training"]


def test_authorization_is_deterministic_immutable_and_manifest_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)

    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )
    authorization_bytes = (first.output_dir / "authorization.json").read_bytes()
    second = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )

    assert second.authorization_id == first.authorization_id
    assert second.idempotent is True
    assert (first.output_dir / "manifest.json").is_file()
    assert (first.output_dir / "authorization.json").read_bytes() == authorization_bytes
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None
    assert snapshot.events[-1].details["authorization_id"] == first.authorization_id


@pytest.mark.parametrize(("approved_by", "reason"), [("", "reason"), ("operator", "  ")])
def test_authorization_requires_explicit_nonempty_identity_and_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approved_by: str,
    reason: str,
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    with pytest.raises(DataValidationError, match="non-empty"):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by=approved_by,
            reason=reason,
        )


def test_expired_authorization_cannot_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="short authorization",
        expires_at=(NOW + timedelta(minutes=1)).isoformat(),
    )
    service.now = lambda: NOW + timedelta(minutes=1)
    service.authorization.now = service.now
    set_capabilities(service, training=True)

    result = service.advance(started.qualification_run_id, target="training")

    assert result.state == "TRAINING_PENDING_APPROVAL"
    assert (
        service.authorization_status(started.qualification_run_id, stage="training")[0].status
        == "EXPIRED"
    )
    assert calls == ["readiness"]


def test_revoked_authorization_is_append_only_and_cannot_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorization_id = authorize_stage(service, started.qualification_run_id, "training")
    authorization, _, _ = service.authorization.storage.authorization(
        started.qualification_run_id, authorization_id
    )
    original = authorization.model_dump(mode="json")

    revoked = service.revoke_authorization(
        started.qualification_run_id,
        authorization_id=authorization_id,
        revoked_by="operator-b",
        reason="review withdrawn",
    )
    repeated = service.revoke_authorization(
        started.qualification_run_id,
        authorization_id=authorization_id,
        revoked_by="operator-b",
        reason="review withdrawn",
    )
    set_capabilities(service, training=True)
    result = service.advance(started.qualification_run_id, target="training")

    assert revoked.effective is True
    assert repeated.idempotent is True
    assert result.state == "TRAINING_PENDING_APPROVAL"
    assert (
        service.authorization_status(started.qualification_run_id, stage="training")[0].status
        == "REVOKED"
    )
    current, _, _ = service.authorization.storage.authorization(
        started.qualification_run_id, authorization_id
    )
    assert current.model_dump(mode="json") == original
    assert calls == ["readiness"]


def test_failed_training_consumes_authorization_and_retry_requires_new_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, failure="training"
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorization_id = authorize_stage(service, started.qualification_run_id, "training")

    failed = service.advance(started.qualification_run_id, target="training")
    pending = service.advance(started.qualification_run_id, target="training")

    assert failed.state == "TRAINING_FAILED"
    assert pending.state == "TRAINING_PENDING_APPROVAL"
    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert status.status == "CONSUMED"
    assert authorization_id in status.consumed_authorization_ids
    assert calls == ["readiness", "training"]


def test_shadow_requires_capability_authorization_and_exact_validation_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, real_shadow=False
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    service.advance(started.qualification_run_id, target="training")
    service.advance(started.qualification_run_id, target="validation")

    set_capabilities(service, shadow=True)
    missing = service.advance(started.qualification_run_id, target="shadow")
    assert missing.state == "SHADOW_PENDING_APPROVAL"
    authorization_id = authorize_stage(service, started.qualification_run_id, "shadow")
    set_capabilities(service, shadow=False)
    disabled = service.advance(started.qualification_run_id, target="shadow")
    assert disabled.state == "SHADOW_PENDING_APPROVAL"
    set_capabilities(service, shadow=True)
    completed = service.advance(started.qualification_run_id, target="shadow")

    assert completed.state == "SHADOW_ENROLLED"
    status = service.authorization_status(started.qualification_run_id, stage="shadow")[0]
    assert status.status == "CONSUMED"
    assert status.authorization_id == authorization_id
    assert calls == ["readiness", "training", "validation", "shadow"]


def test_shadow_artifact_change_makes_authorization_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, real_shadow=True
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    service.advance(started.qualification_run_id, target="training")
    service.advance(started.qualification_run_id, target="validation")
    authorize_stage(service, started.qualification_run_id, "shadow")
    artifact_manifest = tmp_path / f"models/challengers/{MODEL_ID}/manifest.json"
    atomic_write_json(artifact_manifest, {"mutated": True})

    result = service.advance(started.qualification_run_id, target="shadow")

    assert result.state == "SHADOW_PENDING_APPROVAL"
    assert (
        service.authorization_status(started.qualification_run_id, stage="shadow")[0].status
        == "STALE"
    )
    assert calls == ["readiness", "training", "validation"]


def test_authorization_expiration_must_be_aware_and_within_policy_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    with pytest.raises(DataValidationError, match="timezone"):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by="operator-a",
            reason="naive expiry",
            expires_at="2026-08-01T12:00:00",
        )
    with pytest.raises(DataValidationError, match="maximum"):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by="operator-a",
            reason="overlong expiry",
            expires_at=(NOW + timedelta(minutes=241)).isoformat(),
        )


def test_expired_authorization_can_be_reissued_with_same_review_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
        expires_at=(NOW + timedelta(minutes=1)).isoformat(),
    )
    first_payload = (first.output_dir / "authorization.json").read_bytes()
    before = service.storage.read(started.qualification_run_id)
    assert before is not None
    service.now = lambda: NOW + timedelta(minutes=2)
    service.authorization.now = service.now

    second = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )

    after = service.storage.read(started.qualification_run_id)
    assert after is not None
    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert second.authorization_id != first.authorization_id
    assert second.idempotent is False
    assert status.status == "ACTIVE"
    assert status.authorization_id == second.authorization_id
    assert first.authorization_id in status.expired_authorization_ids
    assert len(after.events) == len(before.events) + 1
    assert (first.output_dir / "authorization.json").read_bytes() == first_payload


def test_revoked_authorization_can_be_reissued_with_same_review_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )
    first_payload = (first.output_dir / "authorization.json").read_bytes()
    service.revoke_authorization(
        started.qualification_run_id,
        authorization_id=first.authorization_id,
        revoked_by="operator-b",
        reason="replacement requested",
    )

    second = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )

    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert second.authorization_id != first.authorization_id
    assert status.status == "ACTIVE"
    assert status.authorization_id == second.authorization_id
    assert first.authorization_id in status.revoked_authorization_ids
    assert (first.output_dir / "authorization.json").read_bytes() == first_payload


def test_authorization_status_uses_documented_historical_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    expired = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="expiring review",
        expires_at=(NOW + timedelta(minutes=1)).isoformat(),
    )
    service.now = lambda: NOW + timedelta(minutes=2)
    service.authorization.now = service.now
    revoked = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-b",
        reason="withdrawn review",
    )
    service.revoke_authorization(
        started.qualification_run_id,
        authorization_id=revoked.authorization_id,
        revoked_by="operator-b",
        reason="withdraw review",
    )

    status = service.authorization_status(started.qualification_run_id, stage="training")[0]

    assert status.status == "REVOKED"
    assert status.authorization_id == revoked.authorization_id
    assert expired.authorization_id in status.expired_authorization_ids
    assert revoked.authorization_id in status.revoked_authorization_ids


def test_consumed_authorization_can_be_reissued_after_explicit_retry_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, failure="training"
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first_id = authorize_stage(service, started.qualification_run_id, "training")
    assert service.advance(started.qualification_run_id, target="training").state == (
        "TRAINING_FAILED"
    )
    assert service.advance(started.qualification_run_id, target="training").state == (
        "TRAINING_PENDING_APPROVAL"
    )

    second = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="fixture-operator",
        reason="fixture training authorization",
    )

    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert second.authorization_id != first_id
    assert status.status == "ACTIVE"
    assert first_id in status.consumed_authorization_ids
    assert calls == ["readiness", "training"]


def test_consumed_shadow_authorization_can_be_reissued_after_retry_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path,
        monkeypatch,
        real_training=True,
        real_shadow=True,
        failure="shadow",
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorize_stage(service, started.qualification_run_id, "training")
    service.advance(started.qualification_run_id, target="training")
    service.advance(started.qualification_run_id, target="validation")
    first_id = authorize_stage(service, started.qualification_run_id, "shadow")
    assert service.advance(started.qualification_run_id, target="shadow").state == "SHADOW_FAILED"
    assert service.advance(started.qualification_run_id, target="shadow").state == (
        "SHADOW_PENDING_APPROVAL"
    )

    second = service.authorize(
        started.qualification_run_id,
        stage="shadow",
        approved_by="fixture-operator",
        reason="fixture shadow authorization",
    )

    status = service.authorization_status(started.qualification_run_id, stage="shadow")[0]
    assert second.authorization_id != first_id
    assert status.status == "ACTIVE"
    assert first_id in status.consumed_authorization_ids
    assert calls == ["readiness", "training", "validation", "shadow"]


def test_stale_authorization_can_be_reissued_for_current_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None
    snapshot = service._transition(
        snapshot,
        "TRAINING_PENDING_APPROVAL",
        "operator reviewed updated immutable context",
    )
    service._publish(
        snapshot,
        service._frozen_from_summary(snapshot),
        service._manifest_identity(snapshot),
    )
    stale = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert stale.status == "STALE"

    second = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )

    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert second.authorization_id != first.authorization_id
    assert status.status == "ACTIVE"
    assert first.authorization_id in status.stale_authorization_ids


@pytest.mark.parametrize(
    ("approved_by", "reason", "expires_at"),
    [
        ("operator-b", "controlled attempt", None),
        ("operator-a", "changed rationale", None),
        ("operator-a", "controlled attempt", (NOW + timedelta(minutes=30)).isoformat()),
    ],
)
def test_active_authorization_conflict_requires_explicit_revocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    approved_by: str,
    reason: str,
    expires_at: str | None,
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )
    before = service.storage.read(started.qualification_run_id)
    assert before is not None
    authorization_root = first.output_dir.parent

    with pytest.raises(
        QualificationAuthorizationConflictError,
        match=f"Revoke authorization {first.authorization_id}",
    ):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by=approved_by,
            reason=reason,
            expires_at=expires_at,
        )

    after = service.storage.read(started.qualification_run_id)
    assert after is not None
    assert after.events == before.events
    assert [path.name for path in authorization_root.iterdir()] == [first.authorization_id]
    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert status.status == "ACTIVE"
    assert status.authorization_id == first.authorization_id


def test_active_authorization_replacement_requires_revocation_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="initial review",
    )
    with pytest.raises(QualificationAuthorizationConflictError):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by="operator-b",
            reason="replacement review",
        )
    service.revoke_authorization(
        started.qualification_run_id,
        authorization_id=first.authorization_id,
        revoked_by="operator-a",
        reason="transfer review responsibility",
    )

    replacement = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-b",
        reason="replacement review",
    )

    status = service.authorization_status(started.qualification_run_id, stage="training")[0]
    assert replacement.authorization_id != first.authorization_id
    assert status.status == "ACTIVE"
    assert status.authorization_id == replacement.authorization_id
    assert first.authorization_id in status.revoked_authorization_ids


def test_exact_active_explicit_expiry_is_idempotent_without_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    expiry = (NOW + timedelta(minutes=30)).isoformat()
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
        expires_at=expiry,
    )
    original = (first.output_dir / "authorization.json").read_bytes()
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None
    second = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
        expires_at=expiry,
    )

    current = service.storage.read(started.qualification_run_id)
    assert current is not None
    assert second.authorization_id == first.authorization_id
    assert second.idempotent is True
    assert current.events == snapshot.events
    assert (first.output_dir / "authorization.json").read_bytes() == original


def test_corrupt_authorization_storage_blocks_reissuance_without_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    first = service.authorize(
        started.qualification_run_id,
        stage="training",
        approved_by="operator-a",
        reason="controlled attempt",
    )
    manifest_path = first.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_sha256"] = "0" * 64
    atomic_write_json(manifest_path, manifest)
    before = service.storage.read(started.qualification_run_id)
    assert before is not None

    with pytest.raises(DataValidationError, match="hash mismatch"):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by="operator-a",
            reason="controlled attempt",
        )

    after = service.storage.read(started.qualification_run_id)
    assert after is not None
    assert after.events == before.events
    assert len(tuple(first.output_dir.parent.iterdir())) == 1
    recovery = service.recovery(started.qualification_run_id)
    assert recovery.status == "ACTION_REQUIRED"
    assert any("hash mismatch" in issue for issue in recovery.issues)


def test_consumed_authorization_cannot_be_revoked_or_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(tmp_path, monkeypatch, real_training=True)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    authorization_id = authorize_stage(service, started.qualification_run_id, "training")
    first = service.advance(started.qualification_run_id, target="training")
    repeated = service.advance(started.qualification_run_id, target="training")
    revocation = service.revoke_authorization(
        started.qualification_run_id,
        authorization_id=authorization_id,
        revoked_by="operator-b",
        reason="too late",
    )

    assert first.state == "VALIDATION_PENDING_APPROVAL"
    assert repeated.state == "VALIDATION_PENDING_APPROVAL"
    assert revocation.effective is False
    assert calls == ["readiness", "training"]


def test_legacy_qualification_is_readable_but_not_authorizable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = qualification_service(tmp_path, monkeypatch)
    started = service.start(REQUEST_ID, as_of=AS_OF)
    snapshot = service.storage.read(started.qualification_run_id)
    assert snapshot is not None
    legacy = replace(
        snapshot,
        summary=snapshot.summary.model_copy(update={"static_qualification_policy_hash": None}),
    )
    frozen = service._frozen_from_summary(snapshot)
    service._publish(legacy, frozen, service._manifest_identity(snapshot))

    with pytest.raises(DataValidationError, match="LEGACY_AUTHORIZATION_MIGRATION_REQUIRED"):
        service.authorize(
            started.qualification_run_id,
            stage="training",
            approved_by="operator-a",
            reason="legacy attempt",
        )
