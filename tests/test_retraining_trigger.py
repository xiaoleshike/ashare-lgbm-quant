"""Governed retraining trigger and immutable request tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.retraining.configuration import load_retraining_policy
from ashare_quant.retraining.service import RetrainingTriggerService
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20260731"
H5 = "champion-h5"
H10 = "challenger-h10"


def test_alpha_decay_trigger_is_horizon_isolated_and_idempotent(tmp_path: Path) -> None:
    service = retraining_fixture(tmp_path, h10_alpha=0.5)

    first = service.evaluate(AS_OF)
    second = service.evaluate(AS_OF)
    by_model = {item.model_id: item for item in first.decisions}

    assert by_model[H5].status == "NO_ACTION_REQUIRED"
    assert by_model[H10].status == "TRIGGERED"
    assert by_model[H10].reasons == ("alpha_decay",)
    assert by_model[H10].request_id is not None
    assert _decision(second, H10).idempotent is True
    request = by_model[H10].output_dir
    assert request is not None
    assert (request / "training_request.json").is_file()
    assert (request / "manifest.json").is_file()
    payload = read_json(request / "training_request.json")
    assert payload["training_allowed"] is True
    assert payload["promotion_allowed"] is False


def test_ic_decline_feature_drift_and_critical_alert_rules(tmp_path: Path) -> None:
    ic = retraining_fixture(tmp_path / "ic", h10_rolling=-0.01).evaluate(AS_OF)
    assert _decision(ic, H10).reasons == ("ic_decline",)

    drift = retraining_fixture(tmp_path / "drift", maximum_psi=0.3).evaluate(AS_OF)
    assert _decision(drift, H5).reasons == ("feature_drift",)
    assert _decision(drift, H10).status == "NO_ACTION_REQUIRED"

    critical = retraining_fixture(
        tmp_path / "critical",
        alerts=[
            {
                "alert_id": "critical-h10",
                "alert_type": "score_collapse",
                "severity": "CRITICAL",
                "status": "ACTIVE",
                "model_id": H10,
            }
        ],
    ).evaluate(AS_OF)
    assert _decision(critical, H10).reasons == ("critical_alert",)
    assert _decision(critical, H5).status == "NO_ACTION_REQUIRED"


def test_observation_maturity_and_manual_request_gate(tmp_path: Path) -> None:
    service = retraining_fixture(tmp_path, h10_alpha=0.5, h10_sessions=59)

    automatic = service.evaluate(AS_OF)
    assert _decision(automatic, H10).status == "INSUFFICIENT_OBSERVATIONS"
    with pytest.raises(DataValidationError, match="lacks mature observations"):
        service.create_request(model_id=H10, as_of=AS_OF)


def test_hash_manifest_and_policy_changes_are_rejected(tmp_path: Path) -> None:
    service = retraining_fixture(tmp_path / "hash", h10_alpha=0.5)
    result = service.evaluate(AS_OF)
    request_id = _decision(result, H10).request_id
    assert request_id is not None

    metrics = (
        tmp_path / "hash/reports/model_monitor" / AS_OF / "performance/performance_metrics.parquet"
    )
    frame = pd.read_parquet(metrics)
    frame.loc[1, "alpha_decay_ratio"] = 0.1
    frame.to_parquet(metrics, index=False)
    with pytest.raises(DataValidationError, match="hash mismatch"):
        service.evaluate(AS_OF)

    policy_path = tmp_path / "hash/config/retraining_policy.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    policy["retraining"]["triggers"]["alpha_decay"]["threshold"] = 0.6
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    changed_policy = RetrainingTriggerService(
        reports_root=tmp_path / "hash/reports",
        config_path=tmp_path / "hash/config/default.yaml",
        policy_path=policy_path,
    )
    validation = changed_policy.validate(request_id)
    assert validation.valid is False
    assert "policy hash changed" in str(validation.error)

    invalid = retraining_fixture(tmp_path / "invalid")
    manifest = tmp_path / "invalid/reports/model_monitor" / AS_OF / "manifest.json"
    atomic_write_json(manifest, {"artifact_name": "wrong"})
    with pytest.raises(DataValidationError, match="invalid monitor manifest"):
        invalid.evaluate(AS_OF)


def test_changed_evidence_creates_new_request_after_cooldown(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    service = retraining_fixture(
        tmp_path,
        as_of="20260730",
        h10_alpha=0.5,
        cooldown_days=0,
        clock=lambda: now,
    )
    first = _decision(service.evaluate("20260730"), H10)
    build_monitor_fixture(tmp_path, as_of="20260731", h10_alpha=0.4)
    second = _decision(service.evaluate("20260731"), H10)

    assert first.request_id != second.request_id
    assert len(service.status()) == 2


def test_cooldown_blocks_new_evidence_but_not_idempotent_identity(tmp_path: Path) -> None:
    now = datetime(2026, 8, 1, tzinfo=UTC)
    current = [now]
    service = retraining_fixture(
        tmp_path,
        as_of="20260730",
        h10_alpha=0.5,
        cooldown_days=30,
        clock=lambda: current[0],
    )
    first = _decision(service.evaluate("20260730"), H10)
    repeated = _decision(service.evaluate("20260730"), H10)
    build_monitor_fixture(tmp_path, as_of="20260731", h10_alpha=0.4)
    current[0] = now + timedelta(days=1)
    blocked = _decision(service.evaluate("20260731"), H10)

    assert repeated.request_id == first.request_id
    assert repeated.idempotent is True
    assert blocked.status == "NO_ACTION_REQUIRED"
    assert blocked.request_id is None
    assert len(service.status()) == 1


def test_atomic_publish_failure_leaves_no_request_or_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = retraining_fixture(tmp_path, h10_alpha=0.5)

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise OSError("forced transaction failure")

    monkeypatch.setattr(
        "ashare_quant.retraining.storage.replace_targets_atomically",
        fail_publish,
    )
    with pytest.raises(OSError, match="forced transaction failure"):
        service.evaluate(AS_OF)
    assert not list((tmp_path / "reports/retraining/requests").glob("*"))
    assert not (tmp_path / "reports/retraining/history/retraining_requests.parquet").exists()


def test_failed_new_request_preserves_existing_request_and_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = retraining_fixture(
        tmp_path,
        as_of="20260730",
        h10_alpha=0.5,
        cooldown_days=0,
    )
    first = _decision(service.evaluate("20260730"), H10)
    assert first.output_dir is not None
    request_before = (first.output_dir / "training_request.json").read_bytes()
    manifest_before = (first.output_dir / "manifest.json").read_bytes()
    history_path = tmp_path / "reports/retraining/history/retraining_requests.parquet"
    history_before = history_path.read_bytes()
    build_monitor_fixture(tmp_path, as_of="20260731", h10_alpha=0.4)

    def fail_publish(*args: object, **kwargs: object) -> None:
        raise OSError("forced transaction failure")

    monkeypatch.setattr(
        "ashare_quant.retraining.storage.replace_targets_atomically",
        fail_publish,
    )
    with pytest.raises(OSError, match="forced transaction failure"):
        service.evaluate("20260731")

    assert (first.output_dir / "training_request.json").read_bytes() == request_before
    assert (first.output_dir / "manifest.json").read_bytes() == manifest_before
    assert history_path.read_bytes() == history_before
    assert len(list((tmp_path / "reports/retraining/requests").glob("*"))) == 1


def test_request_manifest_identity_tampering_is_rejected(tmp_path: Path) -> None:
    service = retraining_fixture(tmp_path, h10_alpha=0.5)
    request = _decision(service.evaluate(AS_OF), H10)
    assert request.request_id is not None
    assert request.output_dir is not None
    manifest_path = request.output_dir / "manifest.json"
    manifest = read_json(manifest_path)
    manifest["horizon"] = 20
    atomic_write_json(manifest_path, manifest)

    result = service.validate(request.request_id)

    assert result.valid is False
    assert result.error == "retraining request and manifest identities differ"


def test_request_manifest_written_last_and_manual_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    service = retraining_fixture(tmp_path, h10_alpha=0.5)
    writes: list[str] = []
    from ashare_quant.retraining import storage as storage_module

    original = storage_module.atomic_write_json

    def tracked(path: Path, payload: dict[str, Any]) -> None:
        writes.append(path.name)
        original(path, payload)

    monkeypatch.setattr(storage_module, "atomic_write_json", tracked)
    result = service.create_request(model_id=H10, as_of=AS_OF)
    assert writes[-1] == "manifest.json"
    request_id = result.decisions[0].request_id
    assert request_id is not None

    config = tmp_path / "config/default.yaml"
    assert main(["--config", str(config), "retraining", "status"]) == 0
    assert request_id in capsys.readouterr().out
    assert (
        main(
            [
                "--config",
                str(config),
                "retraining",
                "validate",
                "--request-id",
                request_id,
            ]
        )
        == 0
    )


def test_trigger_engine_has_no_stateful_service_dependency(tmp_path: Path, monkeypatch) -> None:
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

    retraining_fixture(tmp_path, h10_alpha=0.5).evaluate(AS_OF)


def retraining_fixture(
    tmp_path: Path,
    *,
    as_of: str = AS_OF,
    h10_alpha: float = 1.0,
    h10_rolling: float = 0.1,
    h10_sessions: int = 70,
    maximum_psi: float = 0.0,
    alerts: list[dict[str, object]] | None = None,
    cooldown_days: int = 30,
    clock=None,
) -> RetrainingTriggerService:
    config = tmp_path / "config/default.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        yaml.safe_dump({"paths": {"reports": str(tmp_path / "reports")}}),
        encoding="utf-8",
    )
    write_policy(tmp_path / "config/retraining_policy.yaml", cooldown_days=cooldown_days)
    build_monitor_fixture(
        tmp_path,
        as_of=as_of,
        h10_alpha=h10_alpha,
        h10_rolling=h10_rolling,
        h10_sessions=h10_sessions,
        maximum_psi=maximum_psi,
        alerts=alerts,
    )
    return RetrainingTriggerService(
        reports_root=tmp_path / "reports",
        config_path=config,
        policy_path=tmp_path / "config/retraining_policy.yaml",
        clock=clock,
    )


def build_monitor_fixture(
    tmp_path: Path,
    *,
    as_of: str,
    h10_alpha: float,
    h10_rolling: float = 0.1,
    h10_sessions: int = 70,
    maximum_psi: float = 0.0,
    alerts: list[dict[str, object]] | None = None,
) -> None:
    reports = tmp_path / "reports"
    monitor = reports / "model_monitor" / as_of
    performance = monitor / "performance"
    alert_root = monitor / "alerts"
    performance.mkdir(parents=True, exist_ok=True)
    alert_root.mkdir(parents=True, exist_ok=True)

    drift_manifest = reports / "model_diagnostics/drift/manifest.json"
    atomic_write_json(drift_manifest, {"schema_version": 1, "artifact_name": "model_drift"})
    health = {
        "as_of": as_of,
        "model_id": H5,
        "drift_reference": {
            "manifest_path": str(drift_manifest),
            "manifest_hash": file_sha256(drift_manifest),
            "metrics": {"maximum_feature_psi": maximum_psi},
        },
    }
    atomic_write_json(monitor / "health.json", health)
    metrics = pd.DataFrame(
        {
            "model_id": [H5, H10],
            "model_role": ["champion", "challenger_h10"],
            "horizon": [5, 10],
            "sessions": [70, h10_sessions],
            "alpha_decay_ratio": [1.0, h10_alpha],
            "rolling_20_ic_mean": [0.1, h10_rolling],
            "rolling_60_ic_mean": [0.1, h10_rolling],
            "rolling_120_ic_mean": [0.1, h10_rolling],
        }
    )
    metrics.to_parquet(performance / "performance_metrics.parquet", index=False)

    observation = reports / "performance_observation" / as_of / "manifest.json"
    atomic_write_json(
        observation,
        {
            "schema_version": 1,
            "artifact_name": "performance_observation",
            "observation_as_of": as_of,
            "access_policy": "prospective_production",
            "contracts": {"labels_used_only_after_maturity": True},
        },
    )
    observation_hash = canonical_payload_hash(
        {
            "manifest": read_json(observation),
            "manifest_file_sha256": file_sha256(observation),
        }
    )
    performance_manifest = {
        "schema_version": 1,
        "artifact_name": "performance_monitor",
        "as_of": as_of,
        "status": "success",
        "metrics_file_sha256": file_sha256(performance / "performance_metrics.parquet"),
        "source_observation_hashes": {as_of: observation_hash},
        "row_counts": {"model_horizon_metrics": 2},
        "models": [
            {"model_id": H5, "model_role": "champion", "horizon": 5},
            {"model_id": H10, "model_role": "challenger_h10", "horizon": 10},
        ],
        "identity_hash": "performance-identity",
    }
    atomic_write_json(performance / "manifest.json", performance_manifest)
    alert_payload = {
        "schema_version": 1,
        "artifact_name": "monitoring_alerts",
        "as_of": as_of,
        "alerts": alerts or [],
    }
    atomic_write_json(alert_root / "alerts.json", alert_payload)
    atomic_write_json(
        alert_root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_name": "alert_engine",
            "as_of": as_of,
            "status": "success",
            "alerts_file_sha256": file_sha256(alert_root / "alerts.json"),
            "identity_hash": "alert-identity",
        },
    )
    atomic_write_json(
        monitor / "manifest.json",
        {
            "schema_version": 1,
            "artifact_name": "production_monitor_manifest",
            "as_of": as_of,
            "model_id": H5,
            "source_artifact_hash": "monitor-identity",
            "monitor_metric_file_hashes": {
                "health": file_sha256(monitor / "health.json"),
                "performance_metrics": file_sha256(performance / "performance_metrics.parquet"),
                "performance_manifest": file_sha256(performance / "manifest.json"),
            },
        },
    )


def write_policy(path: Path, *, cooldown_days: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "retraining": {
            "schema_version": 1,
            "policy_version": "test-v1",
            "enabled": True,
            "cooldown_days": cooldown_days,
            "minimum_observation_sessions": {"h5": 60, "h10": 60, "h20": 90, "h60": 120},
            "triggers": {
                "alpha_decay": {"enabled": True, "threshold": 0.7},
                "ic_decline": {"enabled": True, "rolling_window": 60, "threshold": 0.0},
                "feature_drift": {"enabled": True, "psi_threshold": 0.2},
                "critical_alert": {"enabled": True},
            },
        }
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    assert load_retraining_policy(path).policy_hash


def _decision(result, model_id: str):
    return next(item for item in result.decisions if item.model_id == model_id)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
