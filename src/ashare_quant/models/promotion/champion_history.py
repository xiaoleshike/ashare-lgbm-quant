"""Immutable Champion assignment history records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.utils.manifest import atomic_write_json


class ChampionAssignmentRecord(BaseModel):
    """One immutable deployment-slot Champion assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    champion_assignment_id: str = Field(min_length=1)
    deployment_slot: str
    model_id: str
    previous_champion_model_id: str
    promotion_request_id: str
    approval_event_id: str
    registry_version_id: str
    activated_at: str


def build_champion_assignment(
    *,
    deployment_slot: str,
    model_id: str,
    previous_champion_model_id: str,
    promotion_request_id: str,
    approval_event_id: str,
    registry_version_id: str,
    activated_at: str,
) -> ChampionAssignmentRecord:
    """Build a deterministic assignment identity."""

    core = {
        "deployment_slot": deployment_slot,
        "model_id": model_id,
        "previous_champion_model_id": previous_champion_model_id,
        "promotion_request_id": promotion_request_id,
        "approval_event_id": approval_event_id,
        "registry_version_id": registry_version_id,
    }
    assignment_id = f"champion_{canonical_payload_hash(core)[:24]}"
    return ChampionAssignmentRecord(
        champion_assignment_id=assignment_id,
        deployment_slot=deployment_slot,
        model_id=model_id,
        previous_champion_model_id=previous_champion_model_id,
        promotion_request_id=promotion_request_id,
        approval_event_id=approval_event_id,
        registry_version_id=registry_version_id,
        activated_at=activated_at,
    )


def publish_champion_assignment(models_root: Path, assignment: ChampionAssignmentRecord) -> Path:
    """Publish one assignment append-only and idempotently."""

    path = models_root / "champion_history" / f"{assignment.champion_assignment_id}.json"
    payload = assignment.model_dump(mode="json")
    if path.exists():
        if _load_json(path) != payload:
            raise DataValidationError(f"immutable Champion assignment differs: {path}")
        return path
    atomic_write_json(path, payload)
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid Champion history JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"Champion history must contain an object: {path}")
    return payload
