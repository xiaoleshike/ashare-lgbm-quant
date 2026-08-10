"""Versioned feature-set provenance without retroactive historical claims."""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.utils.manifest import atomic_write_json


class FeatureSetProvenance(BaseModel):
    """Immutable identity and evidence behind an ordered feature set."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    artifact_name: Literal["feature_set_provenance"]
    feature_set_name: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    provenance_status: Literal["GOVERNED", "LEGACY_PROVENANCE_INCOMPLETE"]
    features: tuple[str, ...]
    feature_list_hash: str
    selection_policy: str = Field(min_length=1)
    selection_policy_version: str | None = None
    selection_start: str | None = None
    selection_end: str | None = None
    source_diagnostics_run_id: str | None = None
    source_diagnostics_manifest_path: str | None = None
    source_diagnostics_manifest_hash: str | None = None
    source_feature_universe_hash: str | None = None
    created_at: str | None = None
    created_by: str | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> FeatureSetProvenance:
        if not self.features or len(self.features) != len(set(self.features)):
            raise ValueError("feature-set features must be non-empty and unique")
        if feature_list_hash(self.features) != self.feature_list_hash:
            raise ValueError("feature_list_hash does not match ordered features")
        governed = (
            self.selection_policy_version,
            self.selection_start,
            self.selection_end,
            self.source_diagnostics_run_id,
            self.source_diagnostics_manifest_path,
            self.source_diagnostics_manifest_hash,
            self.source_feature_universe_hash,
            self.created_at,
            self.created_by,
        )
        if self.provenance_status == "GOVERNED" and any(value is None for value in governed):
            raise ValueError("governed feature set requires complete selection provenance")
        if (
            self.selection_start
            and self.selection_end
            and self.selection_start > self.selection_end
        ):
            raise ValueError("feature selection window is reversed")
        return self

    @property
    def feature_set_id(self) -> str:
        payload = self.model_dump(mode="json", exclude={"created_at"})
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return f"feature_set_{hashlib.sha256(encoded.encode()).hexdigest()[:16]}"


def load_feature_set_provenance(path: Path) -> FeatureSetProvenance:
    """Load strict provenance; legacy feature lists are never silently upgraded."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FeatureSetProvenance.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DataValidationError(f"invalid feature-set provenance {path}: {error}") from error


def validate_governed_feature_set(path: Path) -> FeatureSetProvenance:
    """Require complete provenance for new governed research evidence."""

    provenance = load_feature_set_provenance(path)
    if provenance.provenance_status != "GOVERNED":
        raise DataValidationError("LEGACY_PROVENANCE_INCOMPLETE: governed feature set required")
    manifest_path = Path(str(provenance.source_diagnostics_manifest_path))
    if not manifest_path.is_file():
        raise DataValidationError("source diagnostics manifest is missing")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if digest != provenance.source_diagnostics_manifest_hash:
        raise DataValidationError("source diagnostics manifest hash changed")
    return provenance


def create_governed_feature_set(
    *,
    diagnostics_dir: Path,
    output_root: Path,
    feature_set_name: str,
    feature_set_version: str,
    created_by: str,
    research_policy_path: Path = Path("config/research_policy.yaml"),
) -> Path:
    """Publish provenance for an existing immutable diagnostics recommendation."""

    if not created_by.strip():
        raise DataValidationError("created_by must be non-empty")
    manifest_path = diagnostics_dir / "manifest.json"
    recommendation_path = diagnostics_dir / "recommended_features.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot load diagnostics provenance: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("artifact_name") != "feature_diagnostics":
        raise DataValidationError("source diagnostics manifest is invalid")
    if not isinstance(recommendation, dict):
        raise DataValidationError("diagnostics recommendation is invalid")
    raw_features = recommendation.get("recommended_features")
    if not isinstance(raw_features, list) or not all(
        isinstance(item, str) for item in raw_features
    ):
        raise DataValidationError("diagnostics recommendation has no feature list")
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise DataValidationError("diagnostics manifest has no chronological split")
    selection_start = str(split.get("train_start", ""))
    selection_end = str(split.get("validation_end", ""))
    policy = load_research_policy(research_policy_path)
    enforce_research_window(
        policy,
        consumer="feature_selection",
        start_date=selection_start,
        end_date=selection_end,
    )
    features = tuple(raw_features)
    source_feature_universe_hash = _source_feature_universe_hash(manifest)
    provenance = FeatureSetProvenance(
        schema_version=1,
        artifact_name="feature_set_provenance",
        feature_set_name=feature_set_name,
        feature_set_version=feature_set_version,
        provenance_status="GOVERNED",
        features=features,
        feature_list_hash=feature_list_hash(features),
        selection_policy="diagnostics_train_validation_selection",
        selection_policy_version="1",
        selection_start=selection_start,
        selection_end=selection_end,
        source_diagnostics_run_id=diagnostics_dir.name,
        source_diagnostics_manifest_path=str(manifest_path.resolve()),
        source_diagnostics_manifest_hash=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_feature_universe_hash=source_feature_universe_hash,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        created_by=created_by.strip(),
    )
    output_dir = output_root / provenance.feature_set_id
    if output_dir.exists():
        existing = load_feature_set_provenance(output_dir / "feature_set.json")
        if existing.model_dump(mode="json", exclude={"created_at"}) != provenance.model_dump(
            mode="json", exclude={"created_at"}
        ):
            raise DataValidationError(f"feature-set identity conflict: {output_dir}")
        return output_dir / "feature_set.json"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root, prefix=".feature-set-") as temporary:
        staging = Path(temporary)
        atomic_write_json(staging / "feature_set.json", provenance.model_dump(mode="json"))
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "artifact_name": "governed_feature_set",
                "feature_set_id": provenance.feature_set_id,
                "feature_set_sha256": hashlib.sha256(
                    (staging / "feature_set.json").read_bytes()
                ).hexdigest(),
                "research_policy_hash": policy.policy_hash,
            },
        )
        staging.rename(output_dir)
    return output_dir / "feature_set.json"


def _source_feature_universe_hash(manifest: dict[str, object]) -> str:
    sources = manifest.get("source_manifests")
    if not isinstance(sources, dict) or not isinstance(sources.get("features_daily"), dict):
        raise DataValidationError("diagnostics manifest lacks feature-universe provenance")
    encoded = json.dumps(
        sources["features_daily"], sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()
