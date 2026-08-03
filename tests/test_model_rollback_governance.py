from __future__ import annotations

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
from ashare_quant.models.promotion import RollbackService
from ashare_quant.models.promotion import rollback as rollback_module
from ashare_quant.models.promotion.champion_history import (
    build_champion_assignment,
    publish_champion_assignment,
)
from ashare_quant.models.promotion.rollback_schema import RollbackReason
from ashare_quant.models.promotion.rollback_storage import RollbackStorage
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.paper_trading import PaperTradingService
from ashare_quant.strategy import CandidateSelector


def _artifact(models_root: Path, model_id: str, *, horizon: int = 5) -> Path:
    path = models_root / model_id
    path.mkdir(parents=True)
    features = ("ret_5d", "amount_mean_20d")
    digest = feature_list_hash(features)
    (path / "model.txt").write_text("tree\n", encoding="utf-8")
    (path / "feature_list.json").write_text(
        json.dumps({"features": list(features), "feature_hash": digest}),
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"validation": {"rank_ic": 0.02}, "test": {"rank_ic": 0.01}}),
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


def _setup_history(models_root: Path) -> tuple[str, str]:
    target_id = "historical_champion"
    current_id = "current_champion"
    registry = ModelRegistry(models_root)
    registry.register_model(_artifact(models_root, target_id))
    registry.promote_model(target_id)
    registry.register_model(_artifact(models_root, current_id))
    registry.promote_model(current_id)
    payload = json.loads(registry.registry_path.read_text(encoding="utf-8"))
    for item in payload["models"]:
        if item["model_id"] == target_id:
            item["status"] = "retired"
    registry.registry_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    assignment = build_champion_assignment(
        deployment_slot="daily_stock_ranker",
        model_id=current_id,
        previous_champion_model_id=target_id,
        promotion_request_id="promotion_fixture",
        approval_event_id="approval_fixture",
        registry_version_id="registry_fixture",
        activated_at="2026-07-30T00:00:00+00:00",
    )
    publish_champion_assignment(models_root, assignment)
    return target_id, current_id


def _service(
    tmp_path: Path,
    *,
    now: datetime | None = None,
    requester: str = "requester",
    reviewer: str = "reviewer",
) -> tuple[RollbackService, Path, str, str]:
    models_root = tmp_path / "models"
    target_id, current_id = _setup_history(models_root)
    service = RollbackService(
        models_root=models_root,
        settings=PromotionReviewSettings(
            reviewer_allowlist=(reviewer,),
            allow_requester_as_reviewer=False,
            review_expire_hours=72,
        ),
        production_lock_path=tmp_path / "runs" / ".production.lock",
        requester_provider=lambda: requester,
        reviewer_provider=lambda: reviewer,
        clock=(lambda: now) if now is not None else None,
    )
    return service, models_root, target_id, current_id


def _approved(
    tmp_path: Path, *, now: datetime | None = None
) -> tuple[RollbackService, Path, str, str, str]:
    service, models_root, target_id, current_id = _service(tmp_path, now=now)
    created = service.create(
        model_id=target_id,
        reason_type="model_degradation",
        reason_description="Prospective performance degraded under human review.",
    )
    assert service.validate(created.request_id).status == "VALIDATED"
    assert service.approve(created.request_id, "Approve controlled rollback.").status == (
        "APPROVED"
    )
    return service, models_root, target_id, current_id, created.request_id


def test_rollback_requires_historical_retired_champion(tmp_path: Path) -> None:
    service, models_root, _, _ = _service(tmp_path)
    arbitrary = "arbitrary_candidate"
    ModelRegistry(models_root).register_model(_artifact(models_root, arbitrary))

    with pytest.raises(DataValidationError, match="retired historical Champion"):
        service.create(
            model_id=arbitrary,
            reason_type="manual",
            reason_description="Not a historical Champion.",
        )


def test_rollback_rejects_wrong_slot_deleted_or_changed_artifact(tmp_path: Path) -> None:
    service, models_root, target_id, _ = _service(tmp_path)
    with pytest.raises(DataValidationError, match="no Champion history"):
        service.create(
            model_id=target_id,
            reason_type="manual",
            reason_description="Wrong deployment slot.",
            deployment_slot="other_slot",
        )

    (models_root / target_id / "model.txt").unlink()
    with pytest.raises(DataValidationError, match="incomplete"):
        service.create(
            model_id=target_id,
            reason_type="manual",
            reason_description="Deleted artifact.",
        )

    service, models_root, target_id, _ = _service(tmp_path / "changed")
    request = service.create(
        model_id=target_id,
        reason_type="manual",
        reason_description="Freeze artifact before validation.",
    )
    (models_root / target_id / "model.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="artifact or deployment contract changed"):
        service.validate(request.request_id)


def test_rollback_supports_frozen_legacy_label_horizon_contract(tmp_path: Path) -> None:
    service, models_root, target_id, _ = _service(tmp_path)
    manifest_path = models_root / target_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["label_horizon"] = manifest.pop("horizon")
    manifest.pop("holding_period")
    manifest.pop("execution_rule")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    request = service.create(
        model_id=target_id,
        reason_type="manual",
        reason_description="Restore a legacy-schema historical Champion.",
    )

    assert request.status == "REQUEST_CREATED"


def test_approved_rollback_applies_and_preserves_old_champion(tmp_path: Path) -> None:
    service, models_root, target_id, current_id, request_id = _approved(tmp_path)
    current_artifact = (models_root / current_id / "model.txt").read_bytes()
    first = service.apply(request_id)
    second = service.apply(request_id)

    assert first.status == "APPLIED"
    assert second.status == "APPLIED"
    assert second.idempotent is True
    registry = ModelRegistry(models_root)
    assert registry.get_champion().model_id == target_id  # type: ignore[union-attr]
    records = {item.model_id: item for item in registry.list_models()}
    assert records[current_id].status == "retired"
    assert (models_root / current_id / "model.txt").read_bytes() == current_artifact
    assert first.registry_version_id is not None
    assert (models_root / "registry_versions" / f"{first.registry_version_id}.json").is_file()
    assert first.champion_assignment_id is not None
    history = models_root / "champion_history" / f"{first.champion_assignment_id}.json"
    payload = json.loads(history.read_text(encoding="utf-8"))
    assert payload["artifact_name"] == "rollback_champion_assignment"
    assert payload["rollback_request_id"] == request_id
    assert (
        models_root / "rollback_requests" / request_id / "rollback_apply_manifest.json"
    ).is_file()


def test_rejected_or_expired_rollback_cannot_apply(tmp_path: Path) -> None:
    service, _, target_id, _ = _service(tmp_path / "rejected")
    request = service.create(
        model_id=target_id,
        reason_type="manual",
        reason_description="Request subject to rejection.",
    )
    service.validate(request.request_id)
    service.reject(request.request_id, "Rollback is not justified.")
    with pytest.raises(DataValidationError, match="APPROVED"):
        service.apply(request.request_id)

    now = datetime.now(UTC)
    service, models_root, _, _, request_id = _approved(tmp_path / "expired", now=now)
    expired = RollbackService(
        models_root=models_root,
        settings=PromotionReviewSettings(reviewer_allowlist=("reviewer",), review_expire_hours=72),
        production_lock_path=tmp_path / "expired" / "runs" / ".production.lock",
        requester_provider=lambda: "requester",
        reviewer_provider=lambda: "reviewer",
        clock=lambda: now + timedelta(hours=73),
    )
    with pytest.raises(DataValidationError, match="expired"):
        expired.apply(request_id)


def test_registry_or_target_change_after_approval_invalidates_rollback(tmp_path: Path) -> None:
    service, models_root, target_id, _, request_id = _approved(tmp_path / "registry")
    ModelRegistry(models_root).register_model(_artifact(models_root, "later_candidate"))
    with pytest.raises(DataValidationError, match="registry changed"):
        service.apply(request_id)

    service, models_root, target_id, _, request_id = _approved(tmp_path / "artifact")
    (models_root / target_id / "model.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="artifact or deployment contract changed"):
        service.apply(request_id)


@pytest.mark.parametrize(
    "failure_target",
    ["publish_registry_versions", "switch_registry_atomically"],
)
def test_rollback_registry_failure_keeps_current_champion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_target: str,
) -> None:
    service, models_root, _, current_id, request_id = _approved(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()

    def fail(*args: object, **kwargs: object) -> None:
        raise OSError(f"injected {failure_target}")

    monkeypatch.setattr(rollback_module, failure_target, fail)
    with pytest.raises(OSError, match=failure_target):
        service.apply(request_id)
    assert (models_root / "registry.json").read_bytes() == registry_before
    assert ModelRegistry(models_root).get_champion().model_id == current_id  # type: ignore[union-attr]


def test_rollback_manifest_failure_restores_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, models_root, _, current_id, request_id = _approved(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()
    history_before = set((models_root / "champion_history").glob("*.json"))
    original_write = rollback_module.atomic_write_json

    def fail_manifest(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "rollback_apply_manifest.json":
            raise OSError("injected rollback manifest failure")
        original_write(path, payload)

    monkeypatch.setattr(rollback_module, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="manifest failure"):
        service.apply(request_id)
    assert (models_root / "registry.json").read_bytes() == registry_before
    assert set((models_root / "champion_history").glob("*.json")) == history_before
    assert ModelRegistry(models_root).get_champion().model_id == current_id  # type: ignore[union-attr]

    monkeypatch.setattr(rollback_module, "atomic_write_json", original_write)
    assert service.apply(request_id).status == "APPLIED"


def test_rollback_recovers_interrupted_registry_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, models_root, target_id, _, request_id = _approved(tmp_path)
    registry_before = (models_root / "registry.json").read_bytes()
    original_publish = rollback_module._publish_immutable

    def interrupt_history(path: Path, payload: dict[str, Any]) -> None:
        if path.parent.name == "champion_history":
            raise KeyboardInterrupt
        original_publish(path, payload)

    monkeypatch.setattr(rollback_module, "_publish_immutable", interrupt_history)
    with pytest.raises(KeyboardInterrupt):
        service.apply(request_id)
    assert (models_root / "registry.json").read_bytes() != registry_before

    monkeypatch.setattr(rollback_module, "_publish_immutable", original_publish)
    assert service.apply(request_id).status == "APPLIED"
    assert ModelRegistry(models_root).get_champion().model_id == target_id  # type: ignore[union-attr]


def test_rollback_lock_order_is_production_then_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, models_root, _, _, request_id = _approved(tmp_path)
    observed: list[Path] = []

    @contextmanager
    def recording_lock(path: Path, *, command: str | None = None) -> Iterator[object]:
        del command
        observed.append(path)
        yield object()

    monkeypatch.setattr(rollback_module, "production_lock", recording_lock)
    assert service.apply(request_id).status == "APPLIED"
    assert observed == [
        tmp_path / "runs" / ".production.lock",
        models_root / ".registry.lock",
    ]


def test_rollback_request_is_idempotent_and_different_reason_is_distinct(tmp_path: Path) -> None:
    service, models_root, target_id, _ = _service(tmp_path)
    first = service.create(
        model_id=target_id,
        reason_type="manual",
        reason_description="First governed reason.",
    )
    second = service.create(
        model_id=target_id,
        reason_type="manual",
        reason_description="First governed reason.",
    )
    different = service.create(
        model_id=target_id,
        reason_type="manual",
        reason_description="Different governed reason.",
    )

    assert first.request_id == second.request_id
    assert second.idempotent is True
    assert different.request_id != first.request_id
    storage = RollbackStorage(models_root)
    bundle = storage.read(first.request_id)
    assert bundle is not None
    altered = bundle.request.model_copy(
        update={
            "reason": RollbackReason(type="manual", description="Attempted immutable overwrite.")
        }
    )
    with pytest.raises(DataValidationError, match="identity differs"):
        storage.publish_request(altered, "0" * 64)


def test_rollback_calls_no_model_or_execution_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _, _, request_id = _approved(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("forbidden API called")

    monkeypatch.setattr(ModelRegistry, "promote_model", forbidden)
    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(BacktestRunner, "run", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)
    monkeypatch.setattr(CandidateSelector, "select", forbidden)

    assert service.apply(request_id).status == "APPLIED"


def test_rollback_cli_create_validate_approve_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models_root = tmp_path / "models"
    target_id, _ = _setup_history(models_root)
    config = tmp_path / "review.yaml"
    config.write_text(
        "promotion:\n"
        "  reviewer_allowlist: [cli_user]\n"
        "  allow_requester_as_reviewer: true\n"
        "  review_expire_hours: 72\n",
        encoding="utf-8",
    )
    reason = tmp_path / "reason.txt"
    reason.write_text("Manual recovery after reviewed degradation.\n", encoding="utf-8")
    comments = tmp_path / "comments.txt"
    comments.write_text("Approve rollback after evidence review.\n", encoding="utf-8")
    monkeypatch.setattr("getpass.getuser", lambda: "cli_user")
    monkeypatch.chdir(tmp_path)
    common = [
        "--config",
        str(config),
        "models",
        "--output-root",
        str(models_root),
        "promotion",
    ]
    assert (
        main(
            [
                *common,
                "rollback-create",
                "--model-id",
                target_id,
                "--reason-file",
                str(reason),
            ]
        )
        == 0
    )
    request_id = capsys.readouterr().out.split("request_id=")[1].split()[0]
    assert main([*common, "rollback-validate", "--request-id", request_id]) == 0
    assert (
        main(
            [
                *common,
                "approve",
                "--request-id",
                request_id,
                "--comments-file",
                str(comments),
            ]
        )
        == 0
    )
    assert main([*common, "rollback-apply", "--request-id", request_id]) == 0
    assert "status=APPLIED" in capsys.readouterr().out
