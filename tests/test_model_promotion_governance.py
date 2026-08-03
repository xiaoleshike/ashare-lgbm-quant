from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.challenger import ChallengerTrainer
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.inference import ProductionInferenceEngine
from ashare_quant.models.promotion import (
    PromotionEvidencePaths,
    PromotionGovernanceResult,
    PromotionGovernanceService,
)
from ashare_quant.models.promotion.storage import PromotionStorage
from ashare_quant.models.registry import ModelRegistry
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
    return PromotionEvidencePaths(
        challenger_evaluation=_write(
            reports_root / "challenger_evaluation" / "run" / "manifest.json",
            {
                "artifact_name": "challenger_evaluation_manifest",
                "schema_version": 1,
                "challenger_model_id": candidate_id,
                "champion_model_id": champion_id,
                "maximum_prediction_date": date,
            },
        ),
        executable_validation=_write(
            reports_root / "executable_validation" / "run" / "manifest.json",
            {
                "artifact_name": "executable_oos_portfolio_validation_manifest",
                "schema_version": 1,
                "challenger_model_id": candidate_id,
                "champion_model_id": champion_id,
                "maximum_signal_date": date,
            },
        ),
        shadow_prediction=_write(
            reports_root / "shadow_predictions" / date / "manifest.json",
            {
                "artifact_name": "shadow_prediction_bundle",
                "schema_version": 1,
                "shadow_run_id": "shadow-one",
                "models": [
                    {
                        "model_id": candidate_id,
                        "access_policy": "prospective_production",
                    }
                ],
            },
        ),
        performance_observation=_write(
            reports_root / "performance_observation" / date / "manifest.json",
            {
                "artifact_name": "performance_observation",
                "schema_version": 1,
                "observation_as_of": date,
            },
        ),
        monitoring_summary=_write(
            reports_root / "model_monitor" / date / "monitor_summary.json",
            {
                "artifact_name": "production_monitor_summary",
                "schema_version": 1,
                "as_of": date,
            },
        ),
        alerts=_write(
            reports_root / "model_monitor" / date / "alerts" / "manifest.json",
            {"artifact_name": "alert_engine", "schema_version": 1, "as_of": date},
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
