"""Read-only discovery and freezing of promotion evidence artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.evidence import PromotionEvidencePaths
from ashare_quant.models.promotion.service import PromotionGovernanceService
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json, utc_now_iso


@dataclass(frozen=True, slots=True)
class EvidencePreparationResult:
    request_id: str
    candidate_model_id: str
    evidence_cutoff_date: str
    evidence_manifest_path: Path
    idempotent: bool


class PromotionEvidenceResolver:
    """Discover only immutable, lineage-matching evidence for one candidate."""

    def __init__(self, *, models_root: Path, reports_root: Path) -> None:
        self.models_root = models_root
        self.reports_root = reports_root
        self.registry = ModelRegistry(models_root)

    def prepare(self, model_id: str) -> EvidencePreparationResult:
        """Discover evidence, create the immutable request, and bind discovery provenance."""

        candidate = next(
            (item for item in self.registry.list_models() if item.model_id == model_id), None
        )
        if candidate is None or candidate.status != "candidate":
            raise DataValidationError("promotion prepare requires a registered candidate")
        champion = self.registry.get_champion(candidate.model_type)
        if champion is None:
            raise DataValidationError("promotion prepare requires a current Champion")
        discovered = {
            "challenger_evaluation": self._latest(
                "challenger_evaluation/*/manifest.json", model_id, champion.model_id
            ),
            "executable_validation": self._latest(
                "executable_validation/*/manifest.json", model_id, champion.model_id
            ),
            "shadow_prediction": self._latest(
                "shadow_predictions/*/manifest.json", model_id, champion.model_id
            ),
            "performance_observation": self._latest(
                "performance_observation/*/manifest.json", model_id, champion.model_id
            ),
            "monitoring_summary": self._latest(
                "model_monitor/*/monitor_summary.json", model_id, champion.model_id
            ),
            "alerts": self._latest(
                "model_monitor/*/alerts/manifest.json", model_id, champion.model_id
            ),
            "paper_trading": self._latest(
                "paper_trading_daily/*/summary.json", model_id, champion.model_id
            ),
        }
        cutoff = max(_artifact_date(path, _load_json(path)) for path in discovered.values())
        result = PromotionGovernanceService(
            models_root=self.models_root,
            reports_root=self.reports_root,
        ).create(
            model_id=model_id,
            evidence_cutoff_date=cutoff,
            evidence_paths=PromotionEvidencePaths(
                challenger_evaluation=discovered["challenger_evaluation"],
                executable_validation=discovered["executable_validation"],
                shadow_prediction=discovered["shadow_prediction"],
                performance_observation=discovered["performance_observation"],
                monitoring_summary=discovered["monitoring_summary"],
                alerts=discovered["alerts"],
            ),
        )
        manifest_path = result.output_dir / "evidence_manifest.json"
        sources = [
            {
                "evidence_type": name,
                "source_path": str(path.resolve().relative_to(self.reports_root.resolve())),
                "sha256": file_sha256(path),
                "artifact_name": _load_json(path).get("artifact_name"),
                "evidence_date": _artifact_date(path, _load_json(path)),
            }
            for name, path in sorted(discovered.items())
        ]
        core = {
            "schema_version": 1,
            "artifact_name": "promotion_resolved_evidence",
            "request_id": result.request_id,
            "candidate_model_id": model_id,
            "champion_model_id": champion.model_id,
            "evidence_cutoff_date": cutoff,
            "sources": sources,
        }
        payload = {
            **core,
            "evidence_manifest_hash": canonical_payload_hash(core),
            "created_at": utc_now_iso(),
            "approval_created": False,
            "promotion_applied": False,
        }
        idempotent = manifest_path.exists()
        if idempotent:
            existing = _load_json(manifest_path)
            if existing.get("evidence_manifest_hash") != payload["evidence_manifest_hash"]:
                raise DataValidationError("immutable resolved evidence identity differs")
        else:
            atomic_write_json(manifest_path, payload)
        return EvidencePreparationResult(
            request_id=result.request_id,
            candidate_model_id=model_id,
            evidence_cutoff_date=cutoff,
            evidence_manifest_path=manifest_path,
            idempotent=idempotent,
        )

    def _latest(self, pattern: str, candidate: str, champion: str) -> Path:
        matches: list[tuple[str, Path]] = []
        for path in sorted(self.reports_root.glob(pattern)):
            try:
                payload = _load_json(path)
                if not _matches_model(payload, candidate, champion):
                    continue
                matches.append((_artifact_date(path, payload), path))
            except (DataValidationError, ValueError):
                continue
        if not matches:
            raise DataValidationError(
                f"no immutable lineage-matching evidence found: pattern={pattern}"
            )
        return max(matches, key=lambda item: (item[0], str(item[1])))[1]


def _matches_model(payload: dict[str, Any], candidate: str, champion: str) -> bool:
    if payload.get("challenger_model_id") is not None:
        return (
            payload.get("challenger_model_id") == candidate
            and payload.get("champion_model_id") == champion
        )
    models = payload.get("models")
    if isinstance(models, list):
        return any(
            isinstance(item, dict)
            and item.get("model_id") == candidate
            and item.get("access_policy") == "prospective_production"
            for item in models
        )
    model_ids = payload.get("model_ids")
    if isinstance(model_ids, list):
        return candidate in model_ids
    model_id = payload.get("model_id")
    if model_id is not None:
        return model_id in {candidate, champion}
    # Monitoring, alerts, and paper summaries bind the production date rather than one challenger.
    return payload.get("artifact_name") in {
        "production_monitor_summary",
        "alert_engine",
        "paper_trading_daily_report",
    }


def _artifact_date(path: Path, payload: dict[str, Any]) -> str:
    for key in (
        "observation_as_of",
        "as_of",
        "maximum_prediction_date",
        "maximum_signal_date",
        "evaluation_end",
        "end_date",
    ):
        value = payload.get(key)
        if isinstance(value, str) and len(value) == 8 and value.isdigit():
            return value
    inputs = payload.get("input_manifests")
    if isinstance(inputs, dict):
        challenger = inputs.get("challenger_predictions")
        if isinstance(challenger, dict):
            value = challenger.get("maximum_prediction_date")
            if isinstance(value, str) and len(value) == 8 and value.isdigit():
                return value
    for parent in path.parents:
        if len(parent.name) == 8 and parent.name.isdigit():
            return parent.name
    raise ValueError(f"cannot determine evidence date: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"evidence file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid evidence JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"evidence JSON must contain an object: {path}")
    return payload
