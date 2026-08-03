"""Static candidate deployment-contract construction and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.promotion.schemas import DeploymentContract, InferenceCompatibility
from ashare_quant.models.registry import RegisteredModel
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256


def build_deployment_contract(candidate: RegisteredModel) -> DeploymentContract:
    """Build a static inference contract without loading the trained model."""

    if candidate.status != "candidate":
        raise DataValidationError("promotion request model must have candidate status")
    artifact = Path(candidate.artifact_path)
    required = ("model.txt", "feature_list.json", "manifest.json", "metrics.json")
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        raise DataValidationError(
            "candidate deployment artifacts are missing: " + ", ".join(missing)
        )
    manifest = _load_json(artifact / "manifest.json")
    feature_payload = _load_json(artifact / "feature_list.json")
    features = feature_payload.get("features")
    if (
        not isinstance(features, list)
        or not features
        or not all(isinstance(item, str) and item for item in features)
    ):
        raise DataValidationError("candidate feature_list.json has no valid features")
    feature_names = tuple(cast(str, item) for item in features)
    computed_feature_hash = feature_list_hash(feature_names)
    if computed_feature_hash != candidate.feature_hash:
        raise DataValidationError("candidate feature hash differs from registry")
    declared_hashes = (
        feature_payload.get("feature_hash"),
        manifest.get("feature_hash"),
        manifest.get("feature_list_hash"),
    )
    if any(value is not None and value != computed_feature_hash for value in declared_hashes):
        raise DataValidationError("candidate artifact contains inconsistent feature hash")
    manifest_model_id = manifest.get("model_id")
    if manifest_model_id is not None and manifest_model_id != candidate.model_id:
        raise DataValidationError("candidate manifest model_id differs from registry")
    horizon = _positive_int(manifest.get("horizon"), "horizon")
    holding_period = _positive_int(
        manifest.get("holding_period", manifest.get("holding_days")), "holding_period"
    )
    if horizon != holding_period:
        raise DataValidationError("deployment holding_period must equal model horizon")
    execution_rule = str(manifest.get("execution_rule") or "")
    if not execution_rule:
        raise DataValidationError("candidate manifest lacks execution_rule")
    compatibility = InferenceCompatibility(
        compatible=True,
        model_type=candidate.model_type,
        required_artifacts=required,
        feature_count=len(features),
    )
    artifact_hashes = {name: file_sha256(artifact / name) for name in required}
    core = {
        "schema_version": 1,
        "artifact_name": "promotion_deployment_contract",
        "model_id": candidate.model_id,
        "feature_hash": computed_feature_hash,
        "horizon": horizon,
        "holding_period": holding_period,
        "execution_rule": execution_rule,
        "inference_compatibility": compatibility.model_dump(mode="json"),
        "artifact_hashes": artifact_hashes,
    }
    return DeploymentContract(
        model_id=candidate.model_id,
        feature_hash=computed_feature_hash,
        horizon=horizon,
        holding_period=holding_period,
        execution_rule=execution_rule,
        inference_compatibility=compatibility,
        artifact_hashes=artifact_hashes,
        deployment_contract_hash=canonical_payload_hash(core),
    )


def validate_deployment_contract(contract: DeploymentContract) -> None:
    """Validate self hash and immutable horizon/execution semantics."""

    if contract.horizon != contract.holding_period:
        raise DataValidationError("deployment holding_period must equal model horizon")
    core = contract.model_dump(mode="json", exclude={"deployment_contract_hash"})
    if not contract.artifact_hashes:
        core.pop("artifact_hashes", None)
    if canonical_payload_hash(core) != contract.deployment_contract_hash:
        raise DataValidationError("deployment contract hash is invalid")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid candidate artifact JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"candidate artifact must contain an object: {path}")
    return payload


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise DataValidationError(f"candidate manifest has invalid {name}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError as error:
            raise DataValidationError(f"candidate manifest lacks valid {name}") from error
    else:
        raise DataValidationError(f"candidate manifest lacks valid {name}")
    if result <= 0:
        raise DataValidationError(f"candidate manifest {name} must be positive")
    return result
