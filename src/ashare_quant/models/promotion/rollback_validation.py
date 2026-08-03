"""Historical Champion and deployment validation for governed rollback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.promotion.review_policy import ReviewPolicy, parse_timestamp
from ashare_quant.models.promotion.rollback_schema import (
    RollbackApprovalEvent,
    RollbackRequest,
    RollbackTargetContract,
)
from ashare_quant.models.promotion.rollback_storage import RollbackBundle, RollbackStorage
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256

REQUIRED_MODEL_ARTIFACTS = ("model.txt", "feature_list.json", "manifest.json", "metrics.json")


@dataclass(frozen=True, slots=True)
class RollbackState:
    target: RegisteredModel
    current_champion: RegisteredModel
    target_contract: RollbackTargetContract
    registry_hash: str


@dataclass(frozen=True, slots=True)
class RollbackApplyContext:
    bundle: RollbackBundle
    state: RollbackState
    approval_event: RollbackApprovalEvent
    approval_event_hash: str


def build_rollback_state(
    *, models_root: Path, target_model_id: str, deployment_slot: str
) -> RollbackState:
    """Validate and freeze a historical Champion target without changing state."""

    registry_path = models_root / "registry.json"
    registry_hash = file_sha256(registry_path)
    registry = ModelRegistry(models_root)
    models = registry.list_models()
    target = next((item for item in models if item.model_id == target_model_id), None)
    if target is None:
        raise DataValidationError(f"rollback target is not registered: {target_model_id}")
    if target.status != "retired":
        raise DataValidationError("rollback target must be a retired historical Champion")
    current = registry.get_champion(target.model_type)
    if current is None:
        raise DataValidationError("rollback requires a current Champion")
    if current.model_id == target.model_id:
        raise DataValidationError("rollback target is already the current Champion")
    assignment_id = _historical_assignment(models_root, target.model_id, deployment_slot)
    target_contract = _target_contract(target, current, assignment_id)
    return RollbackState(target, current, target_contract, registry_hash)


def validate_rollback_request(request: RollbackRequest, models_root: Path) -> RollbackState:
    """Revalidate every request-bound registry, history, and artifact identity."""

    state = build_rollback_state(
        models_root=models_root,
        target_model_id=request.target_model_id,
        deployment_slot=request.deployment_slot,
    )
    if state.registry_hash != request.registry_hash:
        raise DataValidationError("registry changed after rollback request")
    if state.current_champion.model_id != request.current_champion_model_id:
        raise DataValidationError("current Champion changed after rollback request")
    if state.target_contract != request.target_contract:
        raise DataValidationError("rollback target artifact or deployment contract changed")
    return state


def current_artifact_set_hash(request: RollbackRequest, models_root: Path) -> str:
    """Hash the current target files using the request's frozen file set."""

    record = next(
        (
            item
            for item in ModelRegistry(models_root).list_models()
            if item.model_id == request.target_model_id
        ),
        None,
    )
    if record is None:
        raise DataValidationError("rollback target disappeared from registry")
    hashes = {
        name: file_sha256(Path(record.artifact_path) / name)
        for name in sorted(request.target_contract.artifact_hashes)
    }
    return canonical_payload_hash(hashes)


def validate_rollback_approval(
    *,
    request_id: str,
    models_root: Path,
    policy: ReviewPolicy,
    now: datetime,
) -> RollbackApplyContext:
    """Validate a bound, approved, unexpired rollback under current state."""

    storage = RollbackStorage(models_root)
    bundle = storage.read(request_id)
    if bundle is None:
        raise DataValidationError(f"rollback request does not exist: {request_id}")
    state = validate_rollback_request(bundle.request, models_root)
    validation = storage.read_validation(request_id)
    if validation is None:
        raise DataValidationError("rollback request has not been validated")
    result, _ = validation
    request_hash = file_sha256(bundle.output_dir / "request.json")
    validation_hash = file_sha256(bundle.output_dir / "validation" / "validation_result.json")
    if result.request_hash != request_hash or result.registry_hash != state.registry_hash:
        raise DataValidationError("rollback validation no longer matches request state")
    approvals = storage.list_approvals(request_id)
    if len(approvals) != 1:
        raise DataValidationError("exactly one rollback approval event is required")
    stored = approvals[0]
    event = stored.event
    if event.event_type != "APPROVED":
        raise DataValidationError("rollback request does not have an APPROVED event")
    if now.astimezone(UTC) > parse_timestamp(event.expires_at):
        raise DataValidationError("rollback approval has expired")
    if stored.manifest.policy_hash != policy.policy_hash:
        raise DataValidationError("rollback review policy changed after approval")
    if event.request_hash != request_hash:
        raise DataValidationError("rollback approval request hash changed")
    if event.validation_result_hash != validation_hash:
        raise DataValidationError("rollback approval validation hash changed")
    if event.registry_hash_at_review != state.registry_hash:
        raise DataValidationError("registry changed after rollback approval")
    artifact_hash = current_artifact_set_hash(bundle.request, models_root)
    if event.target_artifact_hash_at_review != artifact_hash:
        raise DataValidationError("rollback target artifact changed after approval")
    return RollbackApplyContext(
        bundle=bundle,
        state=state,
        approval_event=event,
        approval_event_hash=file_sha256(stored.event_path),
    )


def _target_contract(
    target: RegisteredModel,
    current: RegisteredModel,
    assignment_id: str,
) -> RollbackTargetContract:
    target_artifact = Path(target.artifact_path)
    missing = [name for name in REQUIRED_MODEL_ARTIFACTS if not (target_artifact / name).is_file()]
    if missing:
        raise DataValidationError("rollback target artifact is incomplete: " + ", ".join(missing))
    target_manifest = _load_json(target_artifact / "manifest.json")
    current_manifest = _load_json(Path(current.artifact_path) / "manifest.json")
    features = _feature_names(target_artifact / "feature_list.json")
    computed_feature_hash = feature_list_hash(features)
    if computed_feature_hash != target.feature_hash:
        raise DataValidationError("rollback target feature hash differs from registry")
    horizon = _positive_int(
        target_manifest.get("horizon", target_manifest.get("label_horizon")),
        "target horizon",
    )
    holding = _positive_int(
        target_manifest.get("holding_period", target_manifest.get("holding_days", horizon)),
        "target holding period",
    )
    current_horizon = _positive_int(
        current_manifest.get("horizon", current_manifest.get("label_horizon")),
        "Champion horizon",
    )
    current_holding = _positive_int(
        current_manifest.get(
            "holding_period", current_manifest.get("holding_days", current_horizon)
        ),
        "Champion holding period",
    )
    target_execution = str(target_manifest.get("execution_rule") or "")
    current_execution = str(current_manifest.get("execution_rule") or "")
    execution = target_execution or current_execution
    current_execution = current_execution or target_execution
    if horizon != holding or not execution:
        raise DataValidationError("rollback target deployment contract is invalid")
    if current_horizon != current_holding or not current_execution:
        raise DataValidationError("current Champion deployment contract is invalid")
    if (horizon, holding, execution) != (
        current_horizon,
        current_holding,
        current_execution,
    ):
        raise DataValidationError("rollback target execution contract is incompatible")
    hashes = {name: file_sha256(target_artifact / name) for name in REQUIRED_MODEL_ARTIFACTS}
    return RollbackTargetContract(
        model_type=target.model_type,
        feature_hash=computed_feature_hash,
        horizon=horizon,
        holding_period=holding,
        execution_rule=execution,
        historical_assignment_id=assignment_id,
        artifact_hashes=hashes,
        artifact_set_hash=canonical_payload_hash(hashes),
    )


def _historical_assignment(models_root: Path, model_id: str, deployment_slot: str) -> str:
    root = models_root / "champion_history"
    matches: list[tuple[str, str]] = []
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        payload = _load_json(path)
        if payload.get("deployment_slot") != deployment_slot:
            continue
        if model_id not in {
            payload.get("model_id"),
            payload.get("previous_champion_model_id"),
        }:
            continue
        matches.append((str(payload.get("activated_at") or ""), path.stem))
    if not matches:
        raise DataValidationError(
            "rollback target has no Champion history for deployment slot " + deployment_slot
        )
    return max(matches)[1]


def _feature_names(path: Path) -> tuple[str, ...]:
    payload = _load_json(path)
    values = payload.get("features")
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(item, str) and item for item in values)
    ):
        raise DataValidationError("rollback target feature list is invalid")
    return tuple(cast(str, item) for item in values)


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise DataValidationError(f"{name} is invalid")
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise DataValidationError(f"{name} is invalid") from error
    if result <= 0:
        raise DataValidationError(f"{name} must be positive")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"rollback source artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid rollback source JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"rollback source must contain an object: {path}")
    return payload
