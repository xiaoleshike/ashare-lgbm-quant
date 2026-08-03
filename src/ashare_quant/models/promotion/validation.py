"""Cross-artifact validation for immutable promotion requests."""

from __future__ import annotations

from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.contracts import validate_deployment_contract
from ashare_quant.models.promotion.evidence import verify_evidence_snapshot
from ashare_quant.models.promotion.storage import PromotionBundle
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256


def validate_bundle(bundle: PromotionBundle, reports_root: Path, registry_path: Path) -> None:
    """Validate all logical hashes and current immutable source bytes."""

    request = bundle.request
    if request.evidence_snapshot_hash != bundle.evidence.evidence_snapshot_hash:
        raise DataValidationError("request evidence hash differs from evidence snapshot")
    if request.deployment_contract_hash != bundle.contract.deployment_contract_hash:
        raise DataValidationError("request deployment hash differs from deployment contract")
    if request.candidate.model_id != bundle.contract.model_id:
        raise DataValidationError("deployment contract model differs from request candidate")
    if request.candidate.feature_hash != bundle.contract.feature_hash:
        raise DataValidationError("deployment contract feature hash differs from candidate")
    if request.registry_hash != file_sha256(registry_path):
        raise DataValidationError("model registry changed after promotion request creation")
    verify_evidence_snapshot(bundle.evidence, reports_root)
    validate_deployment_contract(bundle.contract)
    identity = request_identity_payload(request)
    if canonical_payload_hash(identity) != bundle.manifest.identity_hash:
        raise DataValidationError("promotion request identity hash is invalid")
    expected_request_id = f"promotion_{bundle.manifest.identity_hash[:24]}"
    if request.request_id != expected_request_id:
        raise DataValidationError("promotion request_id is not derived from its identity")


def request_identity_payload(request: object) -> dict[str, object]:
    """Return logical request identity excluding time and self-derived request ID."""

    from ashare_quant.models.promotion.schemas import PromotionRequest

    if not isinstance(request, PromotionRequest):
        raise TypeError("request must be PromotionRequest")
    return {
        "candidate": request.candidate.model_dump(mode="json"),
        "current_champion": request.current_champion.model_dump(mode="json"),
        "current_champion_assignment": request.current_champion_assignment.model_dump(mode="json"),
        "evidence_cutoff_date": request.evidence_cutoff_date,
        "evidence_snapshot_hash": request.evidence_snapshot_hash,
        "deployment_contract_hash": request.deployment_contract_hash,
        "registry_hash": request.registry_hash,
    }
