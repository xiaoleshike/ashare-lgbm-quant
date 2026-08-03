"""Retraining execution readiness and isolation tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, PathSettings
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.promotion.gate_rules import load_promotion_gate_policy
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.configuration import load_retraining_policy
from ashare_quant.retraining.readiness import RetrainingExecutionReadinessValidator
from ashare_quant.retraining.schemas import (
    EvidenceReference,
    RetrainingEvidence,
    TrainingRequest,
    TrainingRequestManifest,
    TrainingTarget,
)
from ashare_quant.retraining.validators import evidence_hash
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20260731"
NOW = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
REQUEST_ID = "training_readiness_test"
RUN_ID = "production_run_1"
SNAPSHOT_ID = "governance_snapshot_1"


def test_ready_bundle_is_published_atomically_and_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = readiness_fixture(tmp_path)
    from ashare_quant.retraining.readiness import service as service_module

    writes: list[str] = []
    original = service_module.atomic_write_json

    def tracked(path: Path, payload: dict[str, Any]) -> None:
        writes.append(path.name)
        original(path, payload)

    monkeypatch.setattr(service_module, "atomic_write_json", tracked)

    first = service.validate(AS_OF, request_id=REQUEST_ID)
    second = service.validate(AS_OF, request_id=REQUEST_ID)

    assert first.report.status == "READY"
    assert set(first.report.checks.values()) == {"PASS"}
    assert second.idempotent is True
    assert (first.output_dir / "readiness.json").is_file()
    assert (first.output_dir / "report.md").is_file()
    assert (first.output_dir / "manifest.json").is_file()
    manifest = read_json(first.output_dir / "manifest.json")
    assert manifest["manifest_written_last"] is True
    assert writes[-1] == "manifest.json"
    assert manifest["request_hash"] == file_sha256(
        tmp_path / f"reports/retraining/requests/{REQUEST_ID}/training_request.json"
    )


@pytest.mark.parametrize("case", ["missing", "stale", "failed"])
def test_scheduler_failures_block_readiness(tmp_path: Path, case: str) -> None:
    service = readiness_fixture(tmp_path)
    invocation = next((tmp_path / "runs/scheduler").glob("*/*.json"))
    if case == "missing":
        invocation.unlink()
    else:
        payload = read_json(invocation)
        if case == "stale":
            payload["completed_time"] = (NOW - timedelta(days=4)).isoformat()
        else:
            payload["status"] = "failed"
        atomic_write_json(invocation, payload)

    result = service.validate(AS_OF, request_id=REQUEST_ID)

    assert result.report.status == "FAILED"
    assert result.report.checks["scheduler"] == "FAIL"
    assert result.report.checks["closed_loop_manifest"] == "NOT_RUN"


@pytest.mark.parametrize("case", ["missing", "incomplete", "dry_run"])
def test_closed_loop_failures_are_rejected(tmp_path: Path, case: str) -> None:
    service = readiness_fixture(tmp_path)
    run = tmp_path / f"runs/20260731/{RUN_ID}/manifest.json"
    if case == "missing":
        run.unlink()
    else:
        payload = read_json(run)
        if case == "incomplete":
            payload["stages"][0]["status"] = "failed"
        else:
            payload["command"] += " --dry-run"
        atomic_write_json(run, payload)

    result = service.validate(AS_OF, request_id=REQUEST_ID)

    assert result.report.status == "FAILED"
    assert result.report.checks["closed_loop_manifest"] == "FAIL"


@pytest.mark.parametrize("case", ["missing", "registry", "recovery"])
def test_governance_failures_are_rejected(tmp_path: Path, case: str) -> None:
    service = readiness_fixture(tmp_path)
    root = tmp_path / f"reports/governance/{AS_OF}"
    if case == "missing":
        (root / "manifest.json").unlink()
    elif case == "registry":
        (tmp_path / "models/registry.json").write_text("{broken", encoding="utf-8")
    else:
        _mutate_governance_report(
            root,
            "recovery.json",
            lambda payload: payload["summary"].update(
                {"interrupted_transactions": ["pending_apply.json"]}
            ),
        )

    result = service.validate(AS_OF, request_id=REQUEST_ID)

    assert result.report.status == "FAILED"
    assert result.report.checks["governance_snapshot"] == "FAIL"


@pytest.mark.parametrize("case", ["hash", "version"])
def test_promotion_policy_drift_has_explicit_failure(tmp_path: Path, case: str) -> None:
    readiness_fixture(tmp_path)
    policy_path = tmp_path / "config/promotion_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if case == "hash":
        policy["promotion"]["minimum_mature_sessions"] = 99
    else:
        policy["promotion"]["policy_version"] = "v3"
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    changed = make_service(tmp_path)

    result = changed.validate(AS_OF, request_id=REQUEST_ID)

    assert result.report.status == "FAILED"
    assert result.report.checks["promotion_policy"] == "FAILED_POLICY_DRIFT"


@pytest.mark.parametrize("case", ["request", "evidence"])
def test_invalid_request_or_evidence_is_rejected(tmp_path: Path, case: str) -> None:
    service = readiness_fixture(tmp_path)
    request_root = tmp_path / f"reports/retraining/requests/{REQUEST_ID}"
    if case == "request":
        payload = read_json(request_root / "training_request.json")
        payload["target_models"][0]["horizon"] = 20
        atomic_write_json(request_root / "training_request.json", payload)
    else:
        request = read_json(request_root / "training_request.json")
        source = tmp_path / "reports" / request["evidence"]["alerts"]["path"]
        source.write_text("{}\n", encoding="utf-8")

    result = service.validate(AS_OF, request_id=REQUEST_ID)

    assert result.report.status == "FAILED"
    assert result.report.checks["training_request"] == "FAIL"


def test_multiple_requests_require_explicit_identity(tmp_path: Path) -> None:
    service = readiness_fixture(tmp_path)
    original = tmp_path / f"reports/retraining/requests/{REQUEST_ID}"
    duplicate = original.parent / "training_second"
    duplicate.mkdir()
    request = read_json(original / "training_request.json")
    request["request_id"] = "training_second"
    atomic_write_json(duplicate / "training_request.json", request)
    manifest = read_json(original / "manifest.json")
    manifest["request_id"] = "training_second"
    manifest["request_file_sha256"] = file_sha256(duplicate / "training_request.json")
    atomic_write_json(duplicate / "manifest.json", manifest)

    result = service.validate(AS_OF)

    assert result.report.status == "FAILED"
    assert "pass --request-id" in result.report.check_details[-1].message


def test_atomic_publish_failure_leaves_no_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = readiness_fixture(tmp_path)

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("forced readiness publication failure")

    monkeypatch.setattr("ashare_quant.retraining.readiness.service.os.replace", fail_replace)

    with pytest.raises(OSError, match="forced readiness publication failure"):
        service.validate(AS_OF, request_id=REQUEST_ID)
    assert not (tmp_path / f"reports/retraining/readiness/{AS_OF}").exists()


def test_cli_exit_codes_and_safety_isolation(tmp_path: Path, monkeypatch, capsys) -> None:
    readiness_fixture(tmp_path)
    from ashare_quant.backtest import BacktestRunner
    from ashare_quant.models.challenger import ChallengerTrainer
    from ashare_quant.models.inference import ProductionInferenceEngine
    from ashare_quant.models.promotion.apply import PromotionApplyService
    from ashare_quant.models.registry import ModelRegistry
    from ashare_quant.paper_trading import PaperTradingService

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden stateful service called")

    monkeypatch.setattr(BacktestRunner, "run", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(PromotionApplyService, "apply", forbidden)
    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)
    registry = tmp_path / "models/registry.json"
    before = registry.read_bytes()

    exit_code = main(
        [
            "--config",
            str(tmp_path / "config/default.yaml"),
            "retraining",
            "readiness",
            "--as-of",
            AS_OF,
            "--request-id",
            REQUEST_ID,
        ]
    )

    assert exit_code == 0
    assert "status=READY" in capsys.readouterr().out
    assert registry.read_bytes() == before


def test_cli_returns_nonzero_when_readiness_fails(tmp_path: Path, capsys) -> None:
    readiness_fixture(tmp_path)
    next((tmp_path / "runs/scheduler").glob("*/*.json")).unlink()

    exit_code = main(
        [
            "--config",
            str(tmp_path / "config/default.yaml"),
            "retraining",
            "readiness",
            "--as-of",
            AS_OF,
            "--request-id",
            REQUEST_ID,
        ]
    )

    assert exit_code == 1
    assert "status=FAILED" in capsys.readouterr().out


def readiness_fixture(tmp_path: Path) -> RetrainingExecutionReadinessValidator:
    settings = make_settings(tmp_path)
    write_configs(tmp_path)
    write_scheduler_and_closed_loop(tmp_path)
    write_registry(tmp_path)
    write_governance(tmp_path)
    write_request(tmp_path)
    return make_service(tmp_path, settings=settings)


def make_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        paths=PathSettings(
            raw_data=tmp_path / "data/raw",
            processed_data=tmp_path / "data/processed",
            parquet_store=tmp_path / "data/parquet",
            duckdb_path=tmp_path / "data/test.duckdb",
            reports=tmp_path / "reports",
            models=tmp_path / "models",
            backtests=tmp_path / "backtests",
            paper_trading=tmp_path / "paper_trading",
            data_quality_logs=tmp_path / "logs",
        )
    )


def make_service(
    tmp_path: Path, *, settings: AppSettings | None = None
) -> RetrainingExecutionReadinessValidator:
    return RetrainingExecutionReadinessValidator(
        settings=settings or make_settings(tmp_path),
        config_path=tmp_path / "config/default.yaml",
        project_root=tmp_path,
        retraining_policy_path=tmp_path / "config/retraining_policy.yaml",
        promotion_policy_path=tmp_path / "config/promotion_policy.yaml",
        now=lambda: NOW,
    )


def write_configs(tmp_path: Path) -> None:
    config = tmp_path / "config/default.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "reports": str(tmp_path / "reports"),
                    "models": str(tmp_path / "models"),
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "config/retraining_policy.yaml").write_text(
        "retraining:\n  policy_version: v1\n  cooldown_days: 30\n",
        encoding="utf-8",
    )
    (tmp_path / "config/promotion_policy.yaml").write_text(
        "promotion:\n  policy_version: v2\n",
        encoding="utf-8",
    )
    deploy = tmp_path / "deploy/systemd"
    deploy.mkdir(parents=True)
    (deploy / "ashare-quant-production.service").write_text("Type=oneshot\n", encoding="utf-8")
    (deploy / "ashare-quant-production.timer").write_text("Persistent=true\n", encoding="utf-8")


def write_scheduler_and_closed_loop(tmp_path: Path) -> None:
    invocation_id = "scheduler_invocation_1"
    invocation = {
        "schema_version": 1,
        "artifact_name": "scheduler_invocation",
        "invocation_id": invocation_id,
        "resolved_as_of": AS_OF,
        "status": "success",
        "skipped": False,
        "pipeline_run_id": RUN_ID,
        "attempts": [{"attempt": 1, "pipeline_run_id": RUN_ID, "status": "success"}],
        "completed_time": (NOW - timedelta(hours=1)).isoformat(),
    }
    atomic_write_json(tmp_path / f"runs/scheduler/{AS_OF}/invocation.json", invocation)
    stage_names = {
        "data_update",
        "data_validate",
        "raw_freshness_gate",
        "universe_build",
        "universe_validate",
        "universe_readiness_gate",
        "features_build",
        "features_validate",
        "features_readiness_gate",
        "model_predict",
        "strategy_candidates",
        "publish_production_summary",
        "shadow_prediction",
        "monitoring",
        "research_agent",
        "governance_snapshot",
        "publish_closed_loop_manifest",
    }
    run = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "status": "success",
        "pipeline_type": "production_daily",
        "resolved_as_of": AS_OF,
        "command": f"ashare-quant pipeline production --as-of {AS_OF}",
        "scheduler_invocation_id": invocation_id,
        "shadow_run_id": "shadow_1",
        "monitor_run_id": "monitor_1",
        "research_run_id": "research_1",
        "governance_snapshot_id": SNAPSHOT_ID,
        "stages": [{"name": name, "status": "success"} for name in sorted(stage_names)],
    }
    atomic_write_json(tmp_path / f"runs/20260731/{RUN_ID}/manifest.json", run)
    closed = {
        "schema_version": 1,
        "artifact_name": "production_closed_loop_manifest",
        "as_of": AS_OF,
        "production_run_id": RUN_ID,
        "shadow_run_id": "shadow_1",
        "monitor_run_id": "monitor_1",
        "research_run_id": "research_1",
        "governance_snapshot_id": SNAPSHOT_ID,
        "stages": [{"name": name, "status": "success"} for name in sorted(stage_names)],
    }
    atomic_write_json(tmp_path / f"reports/{AS_OF}/closed_loop/{RUN_ID}/manifest.json", closed)
    atomic_write_json(tmp_path / f"reports/{AS_OF}/closed_loop_manifest.json", closed)


def write_registry(tmp_path: Path) -> None:
    artifact = tmp_path / "models/artifact"
    artifact.mkdir(parents=True)
    features = ("ret_5d",)
    registry = {
        "schema_version": 1,
        "updated_at": "2026-07-31T12:00:00+00:00",
        "models": [
            {
                "model_id": "champion_h5",
                "experiment_id": "experiment_h5",
                "model_type": "lightgbm_ranker",
                "feature_hash": feature_list_hash(features),
                "feature_count": 1,
                "training_date_range": {"start": "20100101", "end": "20260701"},
                "validation_metrics": {},
                "test_metrics": {},
                "git_commit": "abc",
                "config_hash": "cfg",
                "creation_time": "2026-07-01T00:00:00+00:00",
                "artifact_path": str(artifact),
                "status": "champion",
            }
        ],
    }
    atomic_write_json(tmp_path / "models/registry.json", registry)
    atomic_write_json(
        tmp_path / "models/champion_history/assignment_1.json",
        {
            "champion_assignment_id": "assignment_1",
            "model_id": "champion_h5",
            "registry_version_id": "registry_v1",
            "activated_at": "2026-07-01T00:00:00+00:00",
        },
    )


def write_governance(tmp_path: Path) -> None:
    root = tmp_path / f"reports/governance/{AS_OF}"
    history = root / f"history/{SNAPSHOT_ID}"
    policy = load_promotion_gate_policy(tmp_path / "config/promotion_policy.yaml")
    registry = tmp_path / "models/registry.json"
    assignment = tmp_path / "models/champion_history/assignment_1.json"
    reports = {
        "status.json": {
            "schema_version": 1,
            "artifact_name": "governance_status_report",
            "status": "PASS",
            "summary": {"champion": {"model_id": "champion_h5", "assignment_id": "assignment_1"}},
            "source_hashes": {
                str(registry): file_sha256(registry),
                str(assignment): file_sha256(assignment),
            },
        },
        "validation.json": {
            "schema_version": 1,
            "artifact_name": "governance_validation_report",
            "status": "PASS",
        },
        "recovery.json": {
            "schema_version": 1,
            "artifact_name": "governance_recovery_report",
            "status": "PASS",
            "summary": {"interrupted_transactions": [], "incomplete_publications": []},
        },
        "promotion_status.json": {
            "schema_version": 1,
            "artifact_name": "governance_promotion_status",
            "as_of": AS_OF,
            "production_run_id": RUN_ID,
            "promotion": {"invalid_requests": []},
            "promotion_policy_version": policy.policy_version,
            "promotion_policy_hash": policy.policy_hash,
        },
    }
    for name, payload in reports.items():
        atomic_write_json(root / name, payload)
        atomic_write_json(history / name, payload)
    manifest = {
        "schema_version": 1,
        "artifact_name": "daily_governance_snapshot",
        "snapshot_id": SNAPSHOT_ID,
        "as_of": AS_OF,
        "artifact_hashes": {name: file_sha256(root / name) for name in sorted(reports)},
    }
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(history / "manifest.json", manifest)


def write_request(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    refs: dict[str, EvidenceReference] = {}
    for name, artifact in (
        ("monitor_snapshot", "production_monitor_manifest"),
        ("performance_observation", "performance_monitor"),
        ("alerts", "alert_engine"),
    ):
        path = reports / f"evidence/{name}.json"
        atomic_write_json(path, {"artifact_name": artifact, "as_of": AS_OF})
        refs[name] = EvidenceReference(
            path=path.relative_to(reports).as_posix(),
            sha256=file_sha256(path),
            artifact_name=artifact,
            as_of=AS_OF,
        )
    evidence = RetrainingEvidence(**refs)
    retraining_policy = load_retraining_policy(tmp_path / "config/retraining_policy.yaml")
    promotion_policy = load_promotion_gate_policy(tmp_path / "config/promotion_policy.yaml")
    request = TrainingRequest(
        request_id=REQUEST_ID,
        created_at="2026-07-31T15:00:00+00:00",
        as_of=AS_OF,
        target_models=(TrainingTarget(model_id="champion_h5", model_role="champion", horizon=5),),
        trigger_reason=("alpha_decay",),
        evidence=evidence,
        evidence_hash=evidence_hash(evidence),
        policy_hash=retraining_policy.policy_hash,
        policy_version=retraining_policy.policy_version,
        promotion_policy_hash=promotion_policy.policy_hash,
        promotion_policy_version=promotion_policy.policy_version,
        generation_mode="automatic",
    )
    root = reports / f"retraining/requests/{REQUEST_ID}"
    atomic_write_json(root / "training_request.json", request.model_dump(mode="json"))
    manifest = TrainingRequestManifest(
        request_id=REQUEST_ID,
        model_id="champion_h5",
        model_role="champion",
        horizon=5,
        trigger_reasons=("alpha_decay",),
        evidence_hashes={name: reference.sha256 for name, reference in refs.items()},
        evidence_hash=request.evidence_hash,
        policy_hash=request.policy_hash,
        policy_version=request.policy_version,
        promotion_policy_hash=request.promotion_policy_hash,
        promotion_policy_version=request.promotion_policy_version,
        git_commit="abc",
        git_dirty=False,
        config_hash="cfg",
        generated_at=request.created_at,
        request_file_sha256=file_sha256(root / "training_request.json"),
    )
    atomic_write_json(root / "manifest.json", manifest.model_dump(mode="json"))


def _mutate_governance_report(root: Path, name: str, mutate) -> None:
    payload = read_json(root / name)
    mutate(payload)
    atomic_write_json(root / name, payload)
    history = root / f"history/{SNAPSHOT_ID}"
    atomic_write_json(history / name, payload)
    manifest = read_json(root / "manifest.json")
    manifest["artifact_hashes"][name] = file_sha256(root / name)
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(history / "manifest.json", manifest)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
