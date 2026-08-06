"""Fixture-only tests for controlled operational qualification."""

from __future__ import annotations

import json
from dataclasses import replace
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
from ashare_quant.retraining.qualification.lifecycle import require_qualification_transition
from ashare_quant.retraining.qualification.schemas import QualificationCheck
from ashare_quant.retraining.qualification.service import OperationalQualificationService
from ashare_quant.utils.manifest import atomic_write_json, config_hash
from test_retraining_lifecycle import (
    AS_OF,
    MODEL_ID,
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


def test_full_fixture_qualification_is_explicit_and_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, calls, _ = qualification_service(
        tmp_path, monkeypatch, real_training=True, real_shadow=True
    )
    started = service.start(REQUEST_ID, as_of=AS_OF)
    trained = service.advance(started.qualification_run_id, target="training")
    validated = service.advance(started.qualification_run_id, target="validation")
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
    service.advance(started.qualification_run_id, target="training")
    service.advance(started.qualification_run_id, target="validation")
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
    result = service.advance(started.qualification_run_id, target="training")
    if failure != "training":
        result = service.advance(started.qualification_run_id, target="validation")
    if failure == "shadow":
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
