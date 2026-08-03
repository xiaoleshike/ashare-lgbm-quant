"""Pre-training request, readiness, policy, and data identity validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.registry import ModelRegistry, RegisteredModel
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.configuration import RetrainingPolicy
from ashare_quant.retraining.execution.schemas import PreparedTrainingData
from ashare_quant.retraining.readiness.schemas import (
    RetrainingReadinessManifest,
    RetrainingReadinessReport,
)
from ashare_quant.retraining.schemas import TrainingRequest
from ashare_quant.retraining.storage import RetrainingRequestStorage
from ashare_quant.retraining.validators import evidence_hash, validate_recorded_evidence
from ashare_quant.utils.manifest import config_hash


class StaleTrainingRequestError(DataValidationError):
    """Frozen readiness or request lineage changed before execution."""


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    request: TrainingRequest
    request_hash: str
    readiness: RetrainingReadinessReport
    readiness_manifest: RetrainingReadinessManifest
    source_model: RegisteredModel
    registry_hash: str


def validate_execution_inputs(
    *,
    request_id: str,
    reports_root: Path,
    processed_root: Path,
    models_root: Path,
    request_storage: RetrainingRequestStorage,
    retraining_policy: RetrainingPolicy,
    promotion_policy: PromotionGatePolicy,
    config_path: Path,
) -> ExecutionContext:
    stored = request_storage.read(request_id)
    if stored is None:
        raise DataValidationError(f"retraining request does not exist: {request_id}")
    request, request_manifest = stored
    if len(request.target_models) != 1:
        raise DataValidationError("challenger_refresh requires exactly one target model")
    request_path = request_storage.requests_root / request_id / "training_request.json"
    request_hash = file_sha256(request_path)
    if (
        request.policy_hash != retraining_policy.policy_hash
        or request.promotion_policy_hash != promotion_policy.policy_hash
    ):
        raise StaleTrainingRequestError("FAILED_STALE_REQUEST: policy hash mismatch")
    validate_recorded_evidence(reports_root, request.evidence)
    if evidence_hash(request.evidence) != request.evidence_hash:
        raise StaleTrainingRequestError("FAILED_STALE_REQUEST: evidence hash mismatch")
    readiness, readiness_manifest = _readiness(reports_root, request, request_hash)
    current_config_hash = config_hash(config_path)
    if (
        current_config_hash is None
        or request_manifest.config_hash != current_config_hash
        or readiness_manifest.config_hash != current_config_hash
    ):
        raise StaleTrainingRequestError("FAILED_STALE_REQUEST: configuration hash mismatch")
    current = {
        "feature_hash": file_sha256(processed_root / "features_daily" / "_manifest.json"),
        "universe_hash": file_sha256(processed_root / "universe_daily" / "_manifest.json"),
        "label_hash": file_sha256(processed_root / "labels_forward" / "_manifest.json"),
    }
    for field, digest in current.items():
        if getattr(readiness, field) != digest or getattr(readiness_manifest, field) != digest:
            raise StaleTrainingRequestError(f"FAILED_STALE_REQUEST: {field} changed")
    registry = models_root / "registry.json"
    registry_hash = file_sha256(registry)
    target = request.target_models[0]
    matches = [
        item
        for item in ModelRegistry(models_root).list_models()
        if item.model_id == target.model_id
    ]
    if len(matches) != 1:
        raise DataValidationError("training request target model is not registered")
    return ExecutionContext(
        request=request,
        request_hash=request_hash,
        readiness=readiness,
        readiness_manifest=readiness_manifest,
        source_model=matches[0],
        registry_hash=registry_hash,
    )


def validate_prepared_training_data(
    prepared: PreparedTrainingData,
    context: ExecutionContext,
) -> None:
    """Ensure dataset preparation preserved the frozen execution identity."""

    target = context.request.target_models[0]
    dataset = prepared.dataset_manifest
    expected = {
        "feature_hash": context.source_model.feature_hash,
        "feature_manifest_hash": context.readiness.feature_hash,
        "universe_hash": context.readiness.universe_hash,
        "label_hash": context.readiness.label_hash,
    }
    for field, value in expected.items():
        if value is None or getattr(dataset, field) != value:
            raise StaleTrainingRequestError(
                f"FAILED_STALE_REQUEST: prepared dataset {field} mismatch"
            )
    if (
        dataset.horizon != target.horizon
        or dataset.label_name != f"future_excess_ret_{target.horizon}d"
        or prepared.holding_period != target.horizon
        or prepared.execution_rule != "next_open"
    ):
        raise DataValidationError("prepared dataset horizon or execution contract mismatch")
    if not prepared.features or dataset.feature_hash != context.source_model.feature_hash:
        raise DataValidationError("prepared feature list differs from frozen source model")
    if len(prepared.train.frame) == 0 or len(prepared.validation.frame) == 0:
        raise DataValidationError("retraining dataset must contain train and validation rows")
    train_start, train_end = _date_range(dataset.train_dates, "train_dates")
    validation_start, validation_end = _date_range(dataset.validation_dates, "validation_dates")
    if not train_start <= train_end < validation_start <= validation_end:
        raise DataValidationError("prepared train/validation chronology is invalid")


def _date_range(value: dict[str, str], name: str) -> tuple[str, str]:
    if set(value) != {"start", "end"}:
        raise DataValidationError(f"prepared {name} must contain start and end")
    start, end = value["start"], value["end"]
    if any(len(item) != 8 or not item.isdigit() for item in (start, end)):
        raise DataValidationError(f"prepared {name} contains an invalid date")
    return start, end


def _readiness(
    reports_root: Path, request: TrainingRequest, request_hash: str
) -> tuple[RetrainingReadinessReport, RetrainingReadinessManifest]:
    root = reports_root / "retraining" / "readiness" / request.as_of
    try:
        report = RetrainingReadinessReport.model_validate(_json(root / "readiness.json"))
        manifest = RetrainingReadinessManifest.model_validate(_json(root / "manifest.json"))
    except ValueError as error:
        raise DataValidationError(f"invalid retraining readiness artifact: {error}") from error
    if (
        report.status != "READY"
        or manifest.status != "READY"
        or report.request_id != request.request_id
        or manifest.request_id != request.request_id
        or report.request_hash != request_hash
        or manifest.request_hash != request_hash
        or manifest.report_sha256 != file_sha256(root / "readiness.json")
        or manifest.markdown_sha256 != file_sha256(root / "report.md")
    ):
        raise StaleTrainingRequestError("FAILED_STALE_REQUEST: readiness is not valid and READY")
    return report, manifest


def _json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DataValidationError(f"required execution artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError(f"execution artifact must contain an object: {path}")
    return payload
