"""Frozen training-request validation for lifecycle orchestration."""

from __future__ import annotations

from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.configuration import RetrainingPolicy
from ashare_quant.retraining.orchestration.schemas import LifecycleInput
from ashare_quant.retraining.storage import RetrainingRequestStorage
from ashare_quant.retraining.validators import evidence_hash, validate_recorded_evidence


def validate_lifecycle_input(
    *,
    request_id: str,
    reports_root: Path,
    storage: RetrainingRequestStorage,
    retraining_policy: RetrainingPolicy,
    promotion_policy: PromotionGatePolicy,
) -> LifecycleInput:
    """Require a complete immutable request without consuming or changing it."""

    stored = storage.read(request_id)
    if stored is None:
        raise DataValidationError(f"lifecycle training request does not exist: {request_id}")
    request, manifest = stored
    if request.status not in {"CREATED", "VALIDATED"}:
        raise DataValidationError("lifecycle request is cancelled, consumed, or invalid")
    if len(request.target_models) != 1:
        raise DataValidationError("lifecycle requires exactly one horizon-isolated target")
    request_path = storage.requests_root / request_id / "training_request.json"
    request_hash = file_sha256(request_path)
    if manifest.request_file_sha256 != request_hash:
        raise DataValidationError("lifecycle request manifest hash mismatch")
    validate_recorded_evidence(reports_root, request.evidence)
    if evidence_hash(request.evidence) != request.evidence_hash:
        raise DataValidationError("lifecycle request evidence hash mismatch")
    if request.policy_hash != retraining_policy.policy_hash:
        raise DataValidationError("lifecycle retraining policy differs from frozen request")
    if request.promotion_policy_hash != promotion_policy.policy_hash:
        raise DataValidationError("lifecycle promotion policy differs from frozen request")
    if not retraining_policy.lifecycle.enabled:
        raise DataValidationError("retraining lifecycle orchestration is disabled")
    return LifecycleInput(
        request=request,
        training_request_hash=request_hash,
        retraining_policy_hash=retraining_policy.policy_hash,
        lifecycle_policy_hash=retraining_policy.lifecycle_policy_hash,
        promotion_policy_hash=promotion_policy.policy_hash,
    )
