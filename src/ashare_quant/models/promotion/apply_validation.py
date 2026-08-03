"""Strict precondition validation for an approved promotion apply."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.approval_schema import ApprovalEvent
from ashare_quant.models.promotion.approval_storage import ApprovalEventStorage
from ashare_quant.models.promotion.contracts import validate_deployment_contract
from ashare_quant.models.promotion.gate_report import read_gate_result
from ashare_quant.models.promotion.gate_schemas import GateResult
from ashare_quant.models.promotion.review_policy import parse_timestamp
from ashare_quant.models.promotion.storage import PromotionBundle, PromotionStorage
from ashare_quant.models.promotion.validation import validate_bundle
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.shadow.storage import file_sha256


@dataclass(frozen=True, slots=True)
class ApplyValidationContext:
    """All immutable identities required by the registry transition."""

    bundle: PromotionBundle
    gate_result: GateResult
    approval_event: ApprovalEvent
    approval_event_hash: str
    candidate: RegisteredModel
    champion: RegisteredModel
    registry_hash: str


def validate_apply_preconditions(
    *,
    request_id: str,
    models_root: Path,
    reports_root: Path,
    now: datetime,
) -> ApplyValidationContext:
    """Validate approval, gate, registry, artifact, and contract bindings."""

    registry_path = models_root / "registry.json"
    bundle = PromotionStorage(models_root).read(request_id)
    if bundle is None:
        raise DataValidationError(f"complete promotion request does not exist: {request_id}")
    validate_bundle(bundle, reports_root, registry_path)
    gate_dir = reports_root / "promotion_gate" / request_id
    stored_gate = read_gate_result(gate_dir)
    if stored_gate is None:
        raise DataValidationError("promotion gate result is missing")
    gate_result, gate_manifest = stored_gate
    if gate_result.status not in {"PASS", "REVIEW_REQUIRED"}:
        raise DataValidationError(f"promotion gate status cannot be applied: {gate_result.status}")
    if gate_manifest.source_request_manifest_hash != file_sha256(
        bundle.output_dir / "manifest.json"
    ):
        raise DataValidationError("gate result is not bound to current promotion request")
    events = ApprovalEventStorage(models_root).list_events(request_id)
    if len(events) != 1:
        raise DataValidationError("exactly one complete approval event is required")
    stored_event = events[0]
    event = stored_event.event
    if event.event_type != "APPROVED":
        raise DataValidationError("promotion request does not have an APPROVED event")
    if now.astimezone(UTC) > parse_timestamp(event.expires_at):
        raise DataValidationError("promotion approval has expired")
    request_hash = file_sha256(bundle.output_dir / "promotion_request.json")
    gate_hash = file_sha256(gate_dir / "gate_result.json")
    registry_hash = file_sha256(registry_path)
    if event.request_hash != request_hash:
        raise DataValidationError("approval request hash no longer matches")
    if event.gate_result_hash != gate_hash:
        raise DataValidationError("approval gate result hash no longer matches")
    if event.registry_hash_at_review != registry_hash:
        raise DataValidationError("registry changed since human approval")
    if bundle.request.registry_hash != registry_hash:
        raise DataValidationError("registry changed since promotion request")
    validate_deployment_contract(bundle.contract)
    required_hashes = bundle.contract.artifact_hashes
    required_files = bundle.contract.inference_compatibility.required_artifacts
    if set(required_hashes) != set(required_files):
        raise DataValidationError("deployment contract lacks complete frozen model artifact hashes")
    registry = ModelRegistry(models_root)
    candidate = next(
        (
            item
            for item in registry.list_models()
            if item.model_id == bundle.request.candidate.model_id
        ),
        None,
    )
    champion = registry.get_champion(bundle.request.current_champion.model_type)
    if candidate is None or candidate.status != "candidate":
        raise DataValidationError("approved model is no longer a registered candidate")
    if champion is None or champion.model_id != bundle.request.current_champion.model_id:
        raise DataValidationError("current Champion assignment changed after approval")
    artifact_root = Path(candidate.artifact_path)
    for filename, expected_hash in required_hashes.items():
        path = artifact_root / filename
        if file_sha256(path) != expected_hash:
            raise DataValidationError(f"candidate model artifact changed: {filename}")
    if candidate.feature_hash != bundle.contract.feature_hash:
        raise DataValidationError("candidate feature hash differs from deployment contract")
    return ApplyValidationContext(
        bundle=bundle,
        gate_result=gate_result,
        approval_event=event,
        approval_event_hash=file_sha256(stored_event.event_path),
        candidate=candidate,
        champion=champion,
        registry_hash=registry_hash,
    )
