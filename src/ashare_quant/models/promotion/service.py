"""Read-only registry service for immutable model-promotion requests."""

from __future__ import annotations

import getpass
from dataclasses import dataclass
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.contracts import build_deployment_contract
from ashare_quant.models.promotion.evidence import (
    PromotionEvidencePaths,
    build_evidence_snapshot,
)
from ashare_quant.models.promotion.schemas import (
    ChampionAssignment,
    ModelIdentity,
    PromotionRequest,
)
from ashare_quant.models.promotion.storage import PromotionStorage
from ashare_quant.models.promotion.validation import request_identity_payload, validate_bundle
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import utc_now_iso


@dataclass(frozen=True, slots=True)
class PromotionGovernanceResult:
    """Public result for create, validate, and status operations."""

    request_id: str
    status: str
    output_dir: Path
    candidate_model_id: str
    champion_model_id: str
    evidence_cutoff_date: str
    idempotent: bool = False


class PromotionGovernanceService:
    """Create governance requests without any lifecycle state mutation."""

    def __init__(self, *, models_root: Path, reports_root: Path) -> None:
        self.models_root = models_root
        self.reports_root = reports_root
        self.registry = ModelRegistry(models_root)
        self.storage = PromotionStorage(models_root)

    def create(
        self,
        *,
        model_id: str,
        evidence_cutoff_date: str,
        evidence_paths: PromotionEvidencePaths,
        deployment_slot: str = "daily_stock_ranker",
    ) -> PromotionGovernanceResult:
        """Freeze one candidate review request while keeping the registry byte-identical."""

        registry_path = self.registry.registry_path
        registry_hash = file_sha256(registry_path)
        models = self.registry.list_models()
        candidate = next((item for item in models if item.model_id == model_id), None)
        if candidate is None:
            raise DataValidationError(f"candidate model is not registered: {model_id}")
        if candidate.status != "candidate":
            raise DataValidationError(f"promotion request requires candidate status: {model_id}")
        champion = self.registry.get_champion(candidate.model_type)
        if champion is None:
            raise DataValidationError(f"no current champion for model type: {candidate.model_type}")
        contract = build_deployment_contract(candidate)
        evidence = build_evidence_snapshot(
            paths=evidence_paths,
            reports_root=self.reports_root,
            candidate_model_id=candidate.model_id,
            champion_model_id=champion.model_id,
            cutoff_date=evidence_cutoff_date,
        )
        assignment = _champion_assignment(champion, deployment_slot)
        request_base = PromotionRequest(
            request_id="pending",
            candidate=_model_identity(candidate),
            current_champion=_model_identity(champion),
            current_champion_assignment=assignment,
            evidence_cutoff_date=evidence_cutoff_date,
            evidence_snapshot_hash=evidence.evidence_snapshot_hash,
            deployment_contract_hash=contract.deployment_contract_hash,
            registry_hash=registry_hash,
            requester=getpass.getuser(),
            created_time=utc_now_iso(),
        )
        identity_hash = canonical_payload_hash(request_identity_payload(request_base))
        request = request_base.model_copy(update={"request_id": f"promotion_{identity_hash[:24]}"})
        existed = self.storage.read(request.request_id) is not None
        bundle = self.storage.publish(
            request=request,
            evidence=evidence,
            contract=contract,
            identity_hash=identity_hash,
        )
        if file_sha256(registry_path) != registry_hash:
            raise DataValidationError(
                "registry changed during read-only promotion request creation"
            )
        validate_bundle(bundle, self.reports_root, registry_path)
        return _result(bundle, idempotent=existed)

    def validate(self, request_id: str) -> PromotionGovernanceResult:
        """Validate one complete request and all frozen source evidence."""

        bundle = self.storage.read(request_id)
        if bundle is None:
            raise DataValidationError(f"complete promotion request does not exist: {request_id}")
        validate_bundle(bundle, self.reports_root, self.registry.registry_path)
        return _result(bundle)

    def status(self, request_id: str) -> PromotionGovernanceResult:
        """Inspect physical publication status without applying lifecycle changes."""

        bundle = self.storage.read(request_id)
        if bundle is None:
            return PromotionGovernanceResult(
                request_id=request_id,
                status="missing",
                output_dir=self.storage.output_dir(request_id),
                candidate_model_id="",
                champion_model_id="",
                evidence_cutoff_date="",
            )
        return _result(bundle)


def _model_identity(model: RegisteredModel) -> ModelIdentity:
    manifest_path = Path(model.artifact_path) / "manifest.json"
    return ModelIdentity(
        model_id=model.model_id,
        experiment_id=model.experiment_id,
        model_type=model.model_type,
        feature_hash=model.feature_hash,
        artifact_manifest_sha256=file_sha256(manifest_path),
        status=model.status,
    )


def _champion_assignment(champion: RegisteredModel, deployment_slot: str) -> ChampionAssignment:
    core = {
        "model_id": champion.model_id,
        "deployment_slot": deployment_slot,
        "activated_at": champion.creation_time,
        "previous_assignment_id": None,
    }
    assignment_hash = canonical_payload_hash(core)
    return ChampionAssignment(
        champion_assignment_id=f"champion_assignment_{assignment_hash[:24]}",
        model_id=champion.model_id,
        deployment_slot=deployment_slot,
        activated_at=champion.creation_time,
        previous_assignment_id=None,
    )


def _result(bundle: object, idempotent: bool = False) -> PromotionGovernanceResult:
    from ashare_quant.models.promotion.storage import PromotionBundle

    if not isinstance(bundle, PromotionBundle):
        raise TypeError("bundle must be PromotionBundle")
    return PromotionGovernanceResult(
        request_id=bundle.request.request_id,
        status="complete",
        output_dir=bundle.output_dir,
        candidate_model_id=bundle.request.candidate.model_id,
        champion_model_id=bundle.request.current_champion.model_id,
        evidence_cutoff_date=bundle.request.evidence_cutoff_date,
        idempotent=idempotent,
    )
