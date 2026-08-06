"""Read-only eligibility validation for retrained Challenger shadow scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256, read_complete_manifest
from ashare_quant.orchestration.publication import validate_production_publication
from ashare_quant.retraining.execution.schemas import QualificationExecutionContext
from ashare_quant.retraining.schemas import TrainingRequest
from ashare_quant.retraining.shadow.schemas import (
    RetrainedModelLineage,
    RetrainedShadowContext,
)
from ashare_quant.retraining.validation.artifact_validation import validate_candidate_artifact
from ashare_quant.retraining.validation.schemas import RetrainingValidationManifest
from ashare_quant.utils.manifest import config_hash


def validate_retrained_shadow_eligibility(
    *,
    model_id: str,
    as_of: str,
    settings: AppSettings,
    config_path: Path,
    runs_root: Path,
    qualification: QualificationExecutionContext | None = None,
) -> tuple[RetrainedShadowContext, pd.DataFrame]:
    """Validate immutable training/validation lineage and current production inputs."""

    production_dir = settings.paths.reports / "shadow_predictions" / as_of
    production_manifest = read_complete_manifest(production_dir)
    if production_manifest is None:
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: production shadow bundle is missing")
    production_summary = validate_production_publication(
        reports_root=settings.paths.reports,
        runs_root=runs_root,
        as_of=as_of,
    )
    if str(production_summary.get("run_id")) != str(production_manifest.get("production_run_id")):
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: production run lineage mismatch")
    if production_manifest.get("access_policy", "prospective_production") not in {
        "prospective_production",
        None,
    }:
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: production source is not prospective")
    champion = pd.read_parquet(production_dir / "predictions.parquet")
    champion = champion.loc[champion["model_role"].astype(str).eq("champion")].copy()
    if champion.empty or set(champion["trade_date"].astype(str)) != {as_of}:
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: Champion reference is invalid")

    candidate = validate_candidate_artifact(
        model_id=model_id,
        models_root=settings.paths.models,
        reports_root=settings.paths.reports,
        processed_root=settings.paths.processed_data,
        config_path=config_path,
        require_current_processed_hashes=False,
    )
    if qualification is None and candidate.artifact.qualification_only:
        raise DataValidationError(
            "SHADOW_NOT_ELIGIBLE: qualification-only candidate requires qualification context"
        )
    if qualification is not None and (
        not candidate.artifact.qualification_only
        or candidate.artifact.qualification_run_id != qualification.qualification_run_id
        or candidate.registration.qualification_run_id != qualification.qualification_run_id
    ):
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: qualification lineage mismatch")
    validation_dir, validation = _validation_manifest(settings.paths.reports, model_id)
    if (
        validation.training_run_id != candidate.artifact.training_run_id
        or validation.artifact_hash != candidate.artifact.artifact_hash
        or validation.feature_hash != candidate.artifact.feature_hash
        or validation.universe_hash != candidate.artifact.universe_hash
        or validation.qualification_only != (qualification is not None)
        or validation.qualification_run_id
        != (qualification.qualification_run_id if qualification else None)
    ):
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: validation lineage mismatch")
    eligibility_path = validation_dir / "shadow" / "eligibility.json"
    eligibility = _json(eligibility_path)
    if (
        eligibility.get("model_id") != model_id
        or eligibility.get("shadow_eligible") is not True
        or eligibility.get("feature_hash_compatible") is not True
        or eligibility.get("universe_compatible") is not True
        or eligibility.get("deployment_contract_compatible") is not True
        or eligibility.get("inference_adapter_available") is not True
        or file_sha256(eligibility_path) != validation.shadow_eligibility_hash
    ):
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: validation eligibility failed")
    artifact = candidate.artifact
    if artifact.holding_period != artifact.horizon or artifact.execution_rule != "next_open":
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: deployment contract mismatch")

    execution_path = (
        settings.paths.reports
        / "retraining"
        / "executions"
        / artifact.training_run_id
        / "execution.json"
    )
    execution = _json(execution_path)
    request_id = str(execution.get("request_id", ""))
    request_path = (
        settings.paths.reports / "retraining" / "requests" / request_id / "training_request.json"
    )
    request = TrainingRequest.model_validate(_json(request_path))
    if (
        not request_id
        or request.request_id != request_id
        or request.target_models[0].horizon != artifact.horizon
        or file_sha256(request_path) != artifact.training_request_hash
    ):
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: training request lineage mismatch")
    current_config = config_hash(config_path)
    if current_config is None or (artifact.config_hash != current_config and qualification is None):
        raise DataValidationError("SHADOW_NOT_ELIGIBLE: configuration hash mismatch")
    lineage = RetrainedModelLineage(
        model_id=model_id,
        parent_model_id=request.target_models[0].model_id,
        training_request_id=request_id,
        training_run_id=artifact.training_run_id,
        validation_run_id=validation.run_id,
    )
    return (
        RetrainedShadowContext(
            as_of=as_of,
            production_run_id=str(production_manifest.get("production_run_id", "")),
            production_shadow_run_id=str(production_manifest.get("shadow_run_id", "")),
            current_universe_hash=str(production_manifest.get("universe_hash", "")),
            generated_at=str(production_manifest.get("generated_at", "")),
            model=candidate.model,
            horizon=artifact.horizon,
            artifact_hash=artifact.artifact_hash,
            feature_hash=artifact.feature_hash,
            training_universe_hash=artifact.universe_hash,
            validation_manifest_hash=file_sha256(validation_dir / "manifest.json"),
            lineage=lineage,
        ),
        champion.loc[:, ["trade_date", "ts_code"]],
    )


def _validation_manifest(
    reports_root: Path, model_id: str
) -> tuple[Path, RetrainingValidationManifest]:
    matches: list[tuple[Path, RetrainingValidationManifest]] = []
    for path in sorted((reports_root / "retraining_validation").glob("*/manifest.json")):
        try:
            manifest = RetrainingValidationManifest.model_validate(_json(path))
        except ValueError as error:
            raise DataValidationError(
                f"SHADOW_NOT_ELIGIBLE: invalid validation: {error}"
            ) from error
        if manifest.model_id == model_id:
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise DataValidationError(
            "SHADOW_NOT_ELIGIBLE: exactly one completed validation artifact is required"
        )
    return matches[0]


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"SHADOW_NOT_ELIGIBLE: required artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"SHADOW_NOT_ELIGIBLE: invalid JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"SHADOW_NOT_ELIGIBLE: JSON must be an object: {path}")
    return payload
