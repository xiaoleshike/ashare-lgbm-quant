from __future__ import annotations

import getpass
import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from ashare_quant.backtest import BacktestRunner
from ashare_quant.cli import main
from ashare_quant.config.settings import PromotionReviewSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger import ChallengerTrainer
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.inference import ProductionInferenceEngine
from ashare_quant.models.promotion import (
    HumanReviewService,
    PromotionApplyService,
    PromotionEvidencePaths,
    PromotionEvidenceResolver,
    PromotionGateEngine,
    PromotionGovernanceResult,
    PromotionGovernanceService,
)
from ashare_quant.models.promotion import apply as promotion_apply_module
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.promotion.storage import PromotionStorage
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.paper_trading import PaperTradingService


def _artifact(models_root: Path, model_id: str, *, horizon: int = 5) -> Path:
    path = models_root / model_id
    path.mkdir(parents=True)
    features = ("ret_5d", "amount_mean_20d")
    digest = feature_list_hash(features)
    (path / "model.txt").write_text("tree\n", encoding="utf-8")
    (path / "feature_list.json").write_text(
        json.dumps({"features": list(features), "feature_hash": digest}), encoding="utf-8"
    )
    (path / "metrics.json").write_text(
        json.dumps({"validation": {"rank_ic": 0.03}, "test": {"rank_ic": 0.02}}),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": "challenger_ranker",
                "model_id": model_id,
                "experiment_id": model_id,
                "feature_hash": digest,
                "feature_list_hash": digest,
                "horizon": horizon,
                "holding_period": horizon,
                "execution_rule": "next_open",
                "universe_hash": "universe-v1",
                "train_start": "20100101",
                "train_end": "20201231",
                "completed_at": "2026-07-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    return path


def _setup_registry(models_root: Path) -> tuple[ModelRegistry, str, str]:
    registry = ModelRegistry(models_root)
    champion_id = "champion_model"
    candidate_id = "challenger_h5"
    registry.register_model(_artifact(models_root, champion_id))
    registry.promote_model(champion_id)
    registry.register_model(_artifact(models_root, candidate_id))
    return registry, champion_id, candidate_id


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _evidence(
    reports_root: Path,
    candidate_id: str,
    champion_id: str,
    *,
    date: str = "20260729",
) -> PromotionEvidencePaths:
    performance_dir = reports_root / "performance_observation" / date
    performance_metrics = _write(
        performance_dir / "metrics.json",
        {
            "available_rows": 100,
            "mature_sessions": 20,
            "daily": [
                {"model_id": candidate_id, "signal_date": f"202607{day:02d}"}
                for day in range(1, 21)
            ],
        },
    )
    alerts_dir = reports_root / "model_monitor" / date / "alerts"
    alerts_payload = _write(
        alerts_dir / "alerts.json",
        {
            "artifact_name": "monitoring_alerts",
            "schema_version": 1,
            "as_of": date,
            "alerts": [],
        },
    )
    return PromotionEvidencePaths(
        challenger_evaluation=_write(
            reports_root / "challenger_evaluation" / "run" / "manifest.json",
            {
                "artifact_name": "challenger_evaluation_manifest",
                "schema_version": 1,
                "challenger_model_id": candidate_id,
                "champion_model_id": champion_id,
                "maximum_prediction_date": date,
                "feature_hash": feature_list_hash(("ret_5d", "amount_mean_20d")),
                "universe_hash": "universe-v1",
                "horizon": 5,
                "holding_period": 5,
                "execution_rule": "next_open",
                "promotion_gate": {
                    "criteria": [
                        {"name": "minimum_rank_ic_delta", "passed": True},
                        {"name": "minimum_top10_return_delta", "passed": True},
                    ]
                },
            },
        ),
        executable_validation=_write(
            reports_root / "executable_validation" / "run" / "manifest.json",
            {
                "artifact_name": "executable_oos_portfolio_validation_manifest",
                "schema_version": 2,
                "accounting_schema_version": 2,
                "cost_policy_hash": "c" * 64,
                "execution_cost_policy": {"cost_policy_hash": "c" * 64},
                "challenger_model_id": candidate_id,
                "champion_model_id": champion_id,
                "maximum_signal_date": date,
                "horizon": 5,
                "holding_period": 5,
                "execution_rule": "signal_close_t_next_open_entry_and_horizon_open_exit",
            },
        ),
        shadow_prediction=_write(
            reports_root / "shadow_predictions" / date / "manifest.json",
            {
                "artifact_name": "shadow_prediction_bundle",
                "schema_version": 1,
                "shadow_run_id": "shadow-one",
                "feature_hash": feature_list_hash(("ret_5d", "amount_mean_20d")),
                "universe_hash": "universe-v1",
                "models": [
                    {
                        "model_id": candidate_id,
                        "access_policy": "prospective_production",
                    }
                ],
            },
        ),
        performance_observation=_write(
            performance_dir / "manifest.json",
            {
                "artifact_name": "performance_observation",
                "schema_version": 1,
                "observation_as_of": date,
                "access_policy": "prospective_production",
                "available_rows": 100,
                "model_ids": [candidate_id],
                "metrics_file_sha256": file_sha256(performance_metrics),
            },
        ),
        monitoring_summary=_write(
            reports_root / "model_monitor" / date / "monitor_summary.json",
            {
                "artifact_name": "production_monitor_summary",
                "schema_version": 1,
                "as_of": date,
                "paper_trading_sessions": 20,
            },
        ),
        alerts=_write(
            alerts_dir / "manifest.json",
            {
                "artifact_name": "alert_engine",
                "schema_version": 1,
                "as_of": date,
                "alerts_file_sha256": file_sha256(alerts_payload),
            },
        ),
    )


def _create(
    tmp_path: Path,
) -> tuple[PromotionGovernanceService, PromotionGovernanceResult, Path, Path]:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    registry, champion_id, candidate_id = _setup_registry(models_root)
    service = PromotionGovernanceService(models_root=models_root, reports_root=reports_root)
    result = service.create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=_evidence(reports_root, candidate_id, champion_id),
    )
    assert registry.get_champion().model_id == champion_id  # type: ignore[union-attr]
    return service, result, models_root, reports_root


def test_create_is_immutable_idempotent_and_does_not_change_registry(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    registry, champion_id, candidate_id = _setup_registry(models_root)
    registry_before = registry.registry_path.read_bytes()
    service = PromotionGovernanceService(models_root=models_root, reports_root=reports_root)
    paths = _evidence(reports_root, candidate_id, champion_id)

    first = service.create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=paths,
    )
    second = service.create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=paths,
    )

    assert first.request_id == second.request_id
    assert second.idempotent is True
    assert registry.registry_path.read_bytes() == registry_before
    assert registry.get_champion().model_id == champion_id  # type: ignore[union-attr]
    assert (first.output_dir / "manifest.json").is_file()


def test_legacy_executable_accounting_cannot_become_promotion_evidence(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    paths = _evidence(reports_root, candidate_id, champion_id)
    executable = json.loads(paths.executable_validation.read_text(encoding="utf-8"))
    executable["schema_version"] = 1
    executable.pop("accounting_schema_version")
    paths.executable_validation.write_text(json.dumps(executable), encoding="utf-8")

    with pytest.raises(DataValidationError, match="legacy or unsupported accounting schema"):
        PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
            model_id=candidate_id,
            evidence_cutoff_date="20260729",
            evidence_paths=paths,
        )


def test_evidence_resolver_discovers_and_hash_binds_immutable_sources(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    registry, champion_id, candidate_id = _setup_registry(models_root)
    registry_before = registry.registry_path.read_bytes()
    _evidence(reports_root, candidate_id, champion_id)
    paper = _write(
        reports_root / "paper_trading_daily/20260729/summary.json",
        {
            "schema_version": 1,
            "artifact_name": "paper_trading_daily_report",
            "as_of": "20260729",
            "portfolio_count": 4,
        },
    )

    resolver = PromotionEvidenceResolver(models_root=models_root, reports_root=reports_root)
    first = resolver.prepare(candidate_id)
    second = resolver.prepare(candidate_id)
    payload = json.loads(first.evidence_manifest_path.read_text(encoding="utf-8"))

    assert second.idempotent is True
    assert payload["candidate_model_id"] == candidate_id
    assert {item["evidence_type"] for item in payload["sources"]} == {
        "alerts",
        "challenger_evaluation",
        "executable_validation",
        "monitoring_summary",
        "paper_trading",
        "performance_observation",
        "shadow_prediction",
    }
    assert registry.registry_path.read_bytes() == registry_before

    paper.write_text('{"artifact_name":"paper_trading_daily_report"}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="source changed|identity differs"):
        resolver.prepare(candidate_id)


def test_gate_policy_hash_changes_when_version_or_threshold_changes() -> None:
    first = PromotionGatePolicy(policy_version="v1")
    second = PromotionGatePolicy(policy_version="v2")
    stricter = PromotionGatePolicy(
        policy_version="v1",
        observation={"h5": 60, "h10": 60, "h20": 90, "h60": 120},
    )

    assert first.policy_hash != second.policy_hash
    assert first.policy_hash != stricter.policy_hash


def test_gate_policy_change_invalidates_existing_immutable_gate_result(tmp_path: Path) -> None:
    _, request, models_root, reports_root = _create(tmp_path)
    PromotionGateEngine(
        models_root=models_root,
        reports_root=reports_root,
        policy=PromotionGatePolicy(policy_version="v1"),
    ).evaluate(request.request_id)

    with pytest.raises(DataValidationError, match="different immutable evaluation identity"):
        PromotionGateEngine(
            models_root=models_root,
            reports_root=reports_root,
            policy=PromotionGatePolicy(policy_version="v2"),
        ).evaluate(request.request_id)


def test_evidence_cutoff_rejects_newer_source(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    service = PromotionGovernanceService(models_root=models_root, reports_root=reports_root)

    with pytest.raises(DataValidationError, match="exceeds cutoff"):
        service.create(
            model_id=candidate_id,
            evidence_cutoff_date="20260728",
            evidence_paths=_evidence(reports_root, candidate_id, champion_id),
        )


def test_validation_detects_changed_evidence_hash(tmp_path: Path) -> None:
    service, result, _, reports_root = _create(tmp_path)
    path = reports_root / "model_monitor" / "20260729" / "monitor_summary.json"
    path.write_text('{"artifact_name":"production_monitor_summary"}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="source changed"):
        service.validate(result.request_id)


def test_deployment_contract_rejects_holding_period_mismatch(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    manifest = models_root / candidate_id / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["holding_period"] = 10
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="must equal model horizon"):
        PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
            model_id=candidate_id,
            evidence_cutoff_date="20260729",
            evidence_paths=_evidence(reports_root, candidate_id, champion_id),
        )


def test_incomplete_or_different_identity_cannot_be_overwritten(tmp_path: Path) -> None:
    service, result, models_root, _ = _create(tmp_path)
    storage = PromotionStorage(models_root)
    incomplete = storage.output_dir("promotion_incomplete")
    incomplete.mkdir(parents=True)

    assert storage.read("promotion_incomplete") is None
    assert incomplete.is_dir()
    bundle = storage.read(result.request_id)
    assert bundle is not None
    with pytest.raises(DataValidationError, match="identity differs"):
        storage.publish(
            request=bundle.request,
            evidence=bundle.evidence,
            contract=bundle.contract,
            identity_hash="0" * 64,
        )
    assert service.status(result.request_id).status == "complete"


def test_create_calls_no_promotion_inference_or_trading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden execution API called")

    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)
    result = PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=_evidence(reports_root, candidate_id, champion_id),
    )

    assert result.status == "complete"


def test_cli_create_validate_and_status(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    args = [
        "--config",
        "config/default.yaml",
        "models",
        "--output-root",
        str(models_root),
        "--reports-root",
        str(reports_root),
        "promotion",
        "create",
        "--model-id",
        candidate_id,
        "--evidence-cutoff-date",
        "20260729",
    ]
    for name, path in evidence.items():
        args.extend((f"--{name.replace('_', '-')}", str(path)))

    assert main(args) == 0
    request_id = capsys.readouterr().out.split("request_id=")[1].split()[0]
    common = [
        "--config",
        "config/default.yaml",
        "models",
        "--output-root",
        str(models_root),
        "--reports-root",
        str(reports_root),
        "promotion",
    ]
    assert main([*common, "validate", "--request-id", request_id]) == 0
    assert main([*common, "status", "--request-id", request_id]) == 0


def test_gate_pass_is_idempotent_and_read_only(tmp_path: Path) -> None:
    service, request, models_root, reports_root = _create(tmp_path)
    del service
    registry_before = (models_root / "registry.json").read_bytes()
    engine = PromotionGateEngine(models_root=models_root, reports_root=reports_root)

    first = engine.evaluate(request.request_id)
    second = engine.evaluate(request.request_id)

    assert first.status == "PASS"
    assert second.idempotent is True
    assert first.output_dir == second.output_dir
    assert (first.output_dir / "manifest.json").is_file()
    assert (models_root / "registry.json").read_bytes() == registry_before
    assert ModelRegistry(models_root).get_champion().model_id == "champion_model"  # type: ignore[union-attr]


@pytest.mark.parametrize("mode", ["missing", "invalid_hash"])
def test_gate_missing_or_changed_evidence_fails(tmp_path: Path, mode: str) -> None:
    _, request, models_root, reports_root = _create(tmp_path)
    source = reports_root / "shadow_predictions" / "20260729" / "manifest.json"
    if mode == "missing":
        source.unlink()
    else:
        source.write_text('{"artifact_name":"shadow_prediction_bundle"}', encoding="utf-8")

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )

    assert result.status == "FAIL"


def test_gate_registry_change_and_assignment_change_fail(tmp_path: Path) -> None:
    _, request, models_root, reports_root = _create(tmp_path)
    registry = ModelRegistry(models_root)
    registry.register_model(_artifact(models_root, "unrelated_candidate"))
    before_gate = registry.registry_path.read_bytes()

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )

    assert result.status == "FAIL"
    assert registry.registry_path.read_bytes() == before_gate


def test_gate_critical_candidate_alert_blocks_promotion(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    alerts_path = reports_root / "model_monitor" / "20260729" / "alerts" / "alerts.json"
    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    alerts["alerts"] = [
        {
            "model_id": candidate_id,
            "severity": "CRITICAL",
            "status": "ACTIVE",
        }
    ]
    alerts_path.write_text(json.dumps(alerts), encoding="utf-8")
    manifest = json.loads(evidence.alerts.read_text(encoding="utf-8"))
    manifest["alerts_file_sha256"] = file_sha256(alerts_path)
    evidence.alerts.write_text(json.dumps(manifest), encoding="utf-8")
    request = PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=evidence,
    )

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )

    assert result.status == "FAIL"


def test_gate_insufficient_prospective_observation_fails(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    metrics_path = reports_root / "performance_observation" / "20260729" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["mature_sessions"] = 5
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest = json.loads(evidence.performance_observation.read_text(encoding="utf-8"))
    manifest["metrics_file_sha256"] = file_sha256(metrics_path)
    evidence.performance_observation.write_text(json.dumps(manifest), encoding="utf-8")
    request = PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=evidence,
    )

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )

    assert result.status == "FAIL"


@pytest.mark.parametrize(
    ("field", "value", "check_name"),
    [
        ("horizon", 10, "deployment_horizon"),
        ("feature_hash", "different-feature-hash", "deployment_feature_hash"),
        ("execution_rule", "same_close", "deployment_execution_rule"),
    ],
)
def test_gate_rejects_evidence_contract_mismatch(
    tmp_path: Path, field: str, value: object, check_name: str
) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    payload = json.loads(evidence.challenger_evaluation.read_text(encoding="utf-8"))
    payload[field] = value
    evidence.challenger_evaluation.write_text(json.dumps(payload), encoding="utf-8")
    request = PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=evidence,
    )

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )
    stored = json.loads((result.output_dir / "gate_result.json").read_text(encoding="utf-8"))

    assert result.status == "FAIL"
    assert any(
        check["name"] == check_name and check["status"] == "FAIL" for check in stored["checks"]
    )


def test_frozen_observation_evidence_is_rejected_before_gate(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    payload = json.loads(evidence.performance_observation.read_text(encoding="utf-8"))
    payload["access_policy"] = "frozen_oos_evaluation"
    evidence.performance_observation.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="forbidden frozen OOS evidence"):
        PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
            model_id=candidate_id,
            evidence_cutoff_date="20260729",
            evidence_paths=evidence,
        )


def test_gate_warning_alert_requires_human_review(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    alerts_path = reports_root / "model_monitor" / "20260729" / "alerts" / "alerts.json"
    alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
    alerts["alerts"] = [{"model_id": candidate_id, "severity": "WARNING", "status": "NEW"}]
    alerts_path.write_text(json.dumps(alerts), encoding="utf-8")
    manifest = json.loads(evidence.alerts.read_text(encoding="utf-8"))
    manifest["alerts_file_sha256"] = file_sha256(alerts_path)
    evidence.alerts.write_text(json.dumps(manifest), encoding="utf-8")
    request = PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=evidence,
    )

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )

    assert result.status == "REVIEW_REQUIRED"


def test_gate_calls_no_execution_or_lifecycle_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, request, models_root, reports_root = _create(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden API called")

    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(BacktestRunner, "run", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)

    result = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )

    assert result.status == "PASS"


def _review_ready(
    tmp_path: Path,
    *,
    reviewer: str = "authorized_reviewer",
    allow_same_user: bool = False,
    clock: object | None = None,
) -> tuple[HumanReviewService, PromotionGovernanceResult, Path, Path]:
    _, request, models_root, reports_root = _create(tmp_path)
    gate = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )
    assert gate.status == "PASS"
    service = HumanReviewService(
        models_root=models_root,
        reports_root=reports_root,
        settings=PromotionReviewSettings(
            reviewer_allowlist=(reviewer,),
            allow_requester_as_reviewer=allow_same_user,
            review_expire_hours=72,
        ),
        reviewer_provider=lambda: reviewer,
        clock=clock,  # type: ignore[arg-type]
    )
    return service, request, models_root, reports_root


def test_approval_is_append_only_and_modifies_no_governed_artifact(tmp_path: Path) -> None:
    service, request, models_root, reports_root = _review_ready(tmp_path)
    request_path = request.output_dir / "promotion_request.json"
    gate_path = reports_root / "promotion_gate" / request.request_id / "gate_result.json"
    registry_path = models_root / "registry.json"
    before = (request_path.read_bytes(), gate_path.read_bytes(), registry_path.read_bytes())

    first = service.approve(request.request_id, "Reviewed prospective evidence.")
    second = service.approve(request.request_id, "Reviewed prospective evidence.")

    assert first.status == "APPROVED"
    assert second.idempotent is True
    assert first.event_id == second.event_id
    assert first.event_path is not None and first.event_path.is_file()
    assert first.event_path.with_name(f"{first.event_id}.manifest.json").is_file()
    assert (request_path.read_bytes(), gate_path.read_bytes(), registry_path.read_bytes()) == before


def test_rejection_does_not_modify_registry_and_prevents_conflicting_event(
    tmp_path: Path,
) -> None:
    service, request, models_root, _ = _review_ready(tmp_path)
    registry_path = models_root / "registry.json"
    before = registry_path.read_bytes()

    rejected = service.reject(request.request_id, "Insufficient regime stability.")

    assert rejected.status == "REJECTED"
    assert registry_path.read_bytes() == before
    with pytest.raises(DataValidationError, match="different immutable review decision"):
        service.approve(request.request_id, "Override rejection.")


def test_review_rejects_unallowlisted_reviewer(tmp_path: Path) -> None:
    service, request, models_root, reports_root = _review_ready(tmp_path)
    denied = HumanReviewService(
        models_root=models_root,
        reports_root=reports_root,
        settings=PromotionReviewSettings(reviewer_allowlist=("someone_else",)),
        reviewer_provider=lambda: "not_allowed",
    )

    with pytest.raises(DataValidationError, match="not allowlisted"):
        denied.review(request.request_id)


def test_review_enforces_requester_reviewer_separation(tmp_path: Path) -> None:
    requester = getpass.getuser()
    service, request, models_root, reports_root = _review_ready(tmp_path)
    del service
    conflict = HumanReviewService(
        models_root=models_root,
        reports_root=reports_root,
        settings=PromotionReviewSettings(reviewer_allowlist=(requester,)),
        reviewer_provider=lambda: requester,
    )

    with pytest.raises(DataValidationError, match="separation of duties"):
        conflict.approve(request.request_id, "Self approval should fail.")


def test_approval_expiry_and_request_expiry(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    service, request, models_root, reports_root = _review_ready(tmp_path, clock=lambda: now)
    approved = service.approve(request.request_id, "Approved with finite TTL.")
    expired_service = HumanReviewService(
        models_root=models_root,
        reports_root=reports_root,
        settings=PromotionReviewSettings(
            reviewer_allowlist=("authorized_reviewer",), review_expire_hours=72
        ),
        reviewer_provider=lambda: "authorized_reviewer",
        clock=lambda: now + timedelta(hours=73),
    )

    assert approved.status == "APPROVED"
    assert expired_service.status(request.request_id).status == "APPROVAL_EXPIRED"

    second_root = tmp_path / "second"
    late_service, late_request, _, _ = _review_ready(
        second_root, clock=lambda: now + timedelta(hours=73)
    )
    with pytest.raises(DataValidationError, match="review window has expired"):
        late_service.approve(late_request.request_id, "Too late.")


@pytest.mark.parametrize("target", ["request", "gate"])
def test_review_detects_request_or_gate_hash_change(tmp_path: Path, target: str) -> None:
    service, request, _, reports_root = _review_ready(tmp_path)
    path = (
        request.output_dir / "promotion_request.json"
        if target == "request"
        else reports_root / "promotion_gate" / request.request_id / "gate_result.json"
    )
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(DataValidationError, match="hash|schema|artifact changed"):
        service.approve(request.request_id, "Hashes must bind the decision.")


def test_approval_becomes_invalid_when_registry_changes_after_review(tmp_path: Path) -> None:
    service, request, models_root, _ = _review_ready(tmp_path)
    service.approve(request.request_id, "Valid at review time.")
    ModelRegistry(models_root).register_model(_artifact(models_root, "later_candidate"))

    assert service.status(request.request_id).status == "INVALID"


def test_fail_gate_cannot_be_approved(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    reports_root = tmp_path / "reports"
    _, champion_id, candidate_id = _setup_registry(models_root)
    evidence = _evidence(reports_root, candidate_id, champion_id)
    metrics_path = reports_root / "performance_observation" / "20260729" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["mature_sessions"] = 0
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest = json.loads(evidence.performance_observation.read_text(encoding="utf-8"))
    manifest["metrics_file_sha256"] = file_sha256(metrics_path)
    evidence.performance_observation.write_text(json.dumps(manifest), encoding="utf-8")
    request = PromotionGovernanceService(models_root=models_root, reports_root=reports_root).create(
        model_id=candidate_id,
        evidence_cutoff_date="20260729",
        evidence_paths=evidence,
    )
    gate = PromotionGateEngine(models_root=models_root, reports_root=reports_root).evaluate(
        request.request_id
    )
    assert gate.status == "FAIL"
    service = HumanReviewService(
        models_root=models_root,
        reports_root=reports_root,
        settings=PromotionReviewSettings(reviewer_allowlist=("authorized_reviewer",)),
        reviewer_provider=lambda: "authorized_reviewer",
    )

    with pytest.raises(DataValidationError, match="FAIL promotion gate"):
        service.approve(request.request_id, "Cannot approve a failed gate.")


def test_human_review_calls_no_execution_or_lifecycle_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, _, _ = _review_ready(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden API called")

    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(BacktestRunner, "run", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)

    assert service.approve(request.request_id, "Human-only decision.").status == "APPROVED"


def test_human_review_cli_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, request, models_root, reports_root = _review_ready(tmp_path)
    requester = getpass.getuser()
    config_path = tmp_path / "review.yaml"
    config_path.write_text(
        "promotion:\n"
        f"  reviewer_allowlist: [{requester}]\n"
        "  allow_requester_as_reviewer: true\n"
        "  review_expire_hours: 72\n",
        encoding="utf-8",
    )
    comments_path = tmp_path / "review.txt"
    comments_path.write_text("Approved after reviewing immutable evidence.\n", encoding="utf-8")
    common = [
        "--config",
        str(config_path),
        "models",
        "--output-root",
        str(models_root),
        "--reports-root",
        str(reports_root),
        "promotion",
    ]

    assert main([*common, "review", "--request-id", request.request_id]) == 0
    assert (
        main(
            [
                *common,
                "approve",
                "--request-id",
                request.request_id,
                "--comments-file",
                str(comments_path),
            ]
        )
        == 0
    )
    assert main([*common, "review-status", "--request-id", request.request_id]) == 0
    assert "status=APPROVED" in capsys.readouterr().out


def _approved_ready(
    tmp_path: Path,
    *,
    now: datetime | None = None,
) -> tuple[PromotionGovernanceResult, Path, Path, datetime]:
    reviewed_at = now or datetime.now(UTC)
    review, request, models_root, reports_root = _review_ready(
        tmp_path,
        clock=lambda: reviewed_at,
    )
    result = review.approve(request.request_id, "Approved for controlled registry apply.")
    assert result.status == "APPROVED"
    return request, models_root, reports_root, reviewed_at


def _apply_service(
    tmp_path: Path,
    models_root: Path,
    reports_root: Path,
    *,
    clock: object | None = None,
) -> PromotionApplyService:
    return PromotionApplyService(
        models_root=models_root,
        reports_root=reports_root,
        production_lock_path=tmp_path / "runs" / ".production.lock",
        clock=clock,  # type: ignore[arg-type]
    )


def test_approved_request_applies_with_versions_and_champion_history(tmp_path: Path) -> None:
    request, models_root, reports_root, reviewed_at = _approved_ready(tmp_path)
    registry_path = models_root / "registry.json"
    old_registry = registry_path.read_bytes()
    old_champion_artifact = (models_root / "champion_model" / "model.txt").read_bytes()
    service = _apply_service(
        tmp_path, models_root, reports_root, clock=lambda: reviewed_at + timedelta(minutes=1)
    )

    applied = service.apply(request.request_id)
    repeated = service.apply(request.request_id)

    assert applied.status == "PROMOTED"
    assert repeated.idempotent is True
    assert repeated.apply_id == applied.apply_id
    assert applied.manifest_path is not None and applied.manifest_path.is_file()
    assert registry_path.read_bytes() != old_registry
    assert (models_root / "champion_model" / "model.txt").read_bytes() == old_champion_artifact
    registry = ModelRegistry(models_root)
    assert registry.get_champion().model_id == "challenger_h5"  # type: ignore[union-attr]
    records = {item.model_id: item for item in registry.list_models()}
    assert records["champion_model"].status == "retired"
    assert applied.registry_version_id is not None
    version = models_root / "registry_versions" / f"{applied.registry_version_id}.json"
    assert version.read_bytes() == registry_path.read_bytes()
    assert list((models_root / "registry_versions").glob("registry_source_*.json"))
    assert applied.champion_assignment_id is not None
    history = models_root / "champion_history" / f"{applied.champion_assignment_id}.json"
    assert history.is_file()
    assert json.loads(history.read_text(encoding="utf-8"))["previous_champion_model_id"] == (
        "champion_model"
    )


def test_apply_dry_run_validates_without_any_registry_or_history_mutation(tmp_path: Path) -> None:
    request, models_root, reports_root, reviewed_at = _approved_ready(tmp_path)
    registry_path = models_root / "registry.json"
    before = registry_path.read_bytes()
    service = _apply_service(
        tmp_path,
        models_root,
        reports_root,
        clock=lambda: reviewed_at + timedelta(minutes=1),
    )

    preview = service.dry_run(request.request_id)

    assert preview.current_champion_model_id == "champion_model"
    assert preview.target_champion_model_id == "challenger_h5"
    assert preview.registry_changes == (
        {
            "model_id": "champion_model",
            "from_status": "champion",
            "to_status": "retired",
        },
        {
            "model_id": "challenger_h5",
            "from_status": "candidate",
            "to_status": "champion",
        },
    )
    assert registry_path.read_bytes() == before
    assert not (models_root / "registry_versions").exists()
    assert not (models_root / "champion_history").exists()
    assert not (request.output_dir / "apply").exists()


def test_rejected_or_expired_approval_cannot_apply(tmp_path: Path) -> None:
    review, request, models_root, reports_root = _review_ready(tmp_path / "rejected")
    review.reject(request.request_id, "Rejected by human reviewer.")
    rejected_registry = (models_root / "registry.json").read_bytes()
    with pytest.raises(DataValidationError, match="APPROVED"):
        _apply_service(tmp_path, models_root, reports_root).apply(request.request_id)
    assert (models_root / "registry.json").read_bytes() == rejected_registry

    now = datetime.now(UTC)
    expired_request, expired_models, expired_reports, _ = _approved_ready(
        tmp_path / "expired", now=now
    )
    expired_registry = (expired_models / "registry.json").read_bytes()
    with pytest.raises(DataValidationError, match="expired"):
        _apply_service(
            tmp_path,
            expired_models,
            expired_reports,
            clock=lambda: now + timedelta(hours=73),
        ).apply(expired_request.request_id)
    assert (expired_models / "registry.json").read_bytes() == expired_registry


def test_apply_rejects_registry_or_candidate_artifact_change(tmp_path: Path) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path / "registry-change")
    ModelRegistry(models_root).register_model(_artifact(models_root, "late_candidate"))
    changed_registry = (models_root / "registry.json").read_bytes()
    with pytest.raises(DataValidationError, match="registry changed"):
        _apply_service(tmp_path, models_root, reports_root).apply(request.request_id)
    assert (models_root / "registry.json").read_bytes() == changed_registry

    request, models_root, reports_root, _ = _approved_ready(tmp_path / "artifact-change")
    registry_before = (models_root / "registry.json").read_bytes()
    (models_root / "challenger_h5" / "model.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="artifact changed"):
        _apply_service(tmp_path, models_root, reports_root).apply(request.request_id)
    assert (models_root / "registry.json").read_bytes() == registry_before


def test_apply_failure_restores_registry_and_can_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()
    service = _apply_service(tmp_path, models_root, reports_root)
    original_publish = promotion_apply_module.publish_champion_assignment

    def fail_history(*args: object, **kwargs: object) -> None:
        raise OSError("injected history failure")

    monkeypatch.setattr(promotion_apply_module, "publish_champion_assignment", fail_history)
    with pytest.raises(OSError, match="injected history failure"):
        service.apply(request.request_id)
    assert (models_root / "registry.json").read_bytes() == registry_before
    assert service.status(request.request_id).status == "APPLY_PENDING"

    monkeypatch.setattr(promotion_apply_module, "publish_champion_assignment", original_publish)
    assert service.apply(request.request_id).status == "PROMOTED"


@pytest.mark.parametrize("failure_point", ["version_write", "registry_switch"])
def test_apply_registry_publication_failure_never_changes_champion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()
    service = _apply_service(tmp_path, models_root, reports_root)

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError(f"injected {failure_point} failure")

    target = (
        "publish_registry_versions"
        if failure_point == "version_write"
        else "switch_registry_atomically"
    )
    monkeypatch.setattr(promotion_apply_module, target, fail)

    with pytest.raises(OSError, match=failure_point):
        service.apply(request.request_id)
    assert (models_root / "registry.json").read_bytes() == registry_before
    assert ModelRegistry(models_root).get_champion().model_id == "champion_model"  # type: ignore[union-attr]
    assert not list(request.output_dir.glob("apply/*/manifest.json"))


def test_apply_recovers_process_interruption_after_registry_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()
    service = _apply_service(tmp_path, models_root, reports_root)
    original_publish = promotion_apply_module.publish_champion_assignment

    def interrupt_after_switch(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        promotion_apply_module, "publish_champion_assignment", interrupt_after_switch
    )
    with pytest.raises(KeyboardInterrupt):
        service.apply(request.request_id)
    assert (models_root / "registry.json").read_bytes() != registry_before

    monkeypatch.setattr(promotion_apply_module, "publish_champion_assignment", original_publish)
    assert service.apply(request.request_id).status == "PROMOTED"
    assert ModelRegistry(models_root).get_champion().model_id == "challenger_h5"  # type: ignore[union-attr]


def test_apply_manifest_failure_cleans_uncommitted_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()
    service = _apply_service(tmp_path, models_root, reports_root)
    original_write = promotion_apply_module.atomic_write_json

    def fail_commit_marker(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "manifest.json" and path.parent.parent.name == "apply":
            raise OSError("injected apply manifest failure")
        original_write(path, payload)

    monkeypatch.setattr(promotion_apply_module, "atomic_write_json", fail_commit_marker)
    with pytest.raises(OSError, match="apply manifest failure"):
        service.apply(request.request_id)

    assert (models_root / "registry.json").read_bytes() == registry_before
    apply_root = request.output_dir / "apply"
    assert not list(apply_root.glob("*/manifest.json"))
    assert not list(apply_root.glob("*/promoted.json"))
    assert not list((models_root / "champion_history").glob("*.json"))

    monkeypatch.setattr(promotion_apply_module, "atomic_write_json", original_write)
    assert service.apply(request.request_id).status == "PROMOTED"


def test_apply_acquires_production_then_registry_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)
    production_path = tmp_path / "runs" / ".production.lock"
    observed: list[Path] = []

    @contextmanager
    def recording_lock(path: Path, *, command: str | None = None) -> Iterator[object]:
        del command
        observed.append(path)
        yield object()

    monkeypatch.setattr(promotion_apply_module, "production_lock", recording_lock)
    result = PromotionApplyService(
        models_root=models_root,
        reports_root=reports_root,
        production_lock_path=production_path,
    ).apply(request.request_id)

    assert result.status == "PROMOTED"
    assert observed == [production_path, models_root / ".registry.lock"]


def test_apply_calls_no_inference_training_backtest_or_trading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden API called")

    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(BacktestRunner, "run", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)

    assert _apply_service(tmp_path, models_root, reports_root).apply(request.request_id).status == (
        "PROMOTED"
    )


def test_promotion_apply_cli_and_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, models_root, reports_root, _ = _approved_ready(tmp_path)
    monkeypatch.chdir(tmp_path)
    common = [
        "--config",
        str(Path(__file__).parents[1] / "config" / "default.yaml"),
        "models",
        "--output-root",
        str(models_root),
        "--reports-root",
        str(reports_root),
        "promotion",
    ]

    assert main([*common, "apply", "--request-id", request.request_id]) == 0
    assert main([*common, "apply-status", "--request-id", request.request_id]) == 0
    output = capsys.readouterr().out
    assert "status=PROMOTED" in output
