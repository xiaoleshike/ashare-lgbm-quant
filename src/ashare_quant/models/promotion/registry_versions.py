"""Immutable registry versions and atomic current-registry switching."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import RegisteredModel
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import atomic_write_json


def build_promoted_registry(
    *,
    records: tuple[RegisteredModel, ...],
    candidate_model_id: str,
    current_champion_model_id: str,
    parent_registry_hash: str,
    request_id: str,
    approval_event_id: str,
    activated_at: str,
) -> tuple[str, dict[str, Any], tuple[RegisteredModel, ...]]:
    """Create a deterministic new registry version without writing it."""

    candidate = next((item for item in records if item.model_id == candidate_model_id), None)
    champion = next((item for item in records if item.model_id == current_champion_model_id), None)
    if candidate is None or candidate.status != "candidate":
        raise DataValidationError("approved candidate is absent or no longer a candidate")
    if champion is None or champion.status != "champion":
        raise DataValidationError("approved current champion assignment has changed")
    if candidate.model_type != champion.model_type:
        raise DataValidationError("candidate and champion model types differ")
    updated = tuple(
        replace(item, status="champion")
        if item.model_id == candidate_model_id
        else (
            replace(item, status="retired") if item.model_id == current_champion_model_id else item
        )
        for item in records
    )
    identity = {
        "parent_registry_hash": parent_registry_hash,
        "promotion_request_id": request_id,
        "approval_event_id": approval_event_id,
        "candidate_model_id": candidate_model_id,
        "previous_champion_model_id": current_champion_model_id,
        "models": [item.to_dict() for item in updated],
    }
    registry_version_id = f"registry_{canonical_payload_hash(identity)[:24]}"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_name": "model_registry",
        "registry_version_id": registry_version_id,
        "parent_registry_hash": parent_registry_hash,
        "promotion_request_id": request_id,
        "approval_event_id": approval_event_id,
        "updated_at": activated_at,
        "models": [item.to_dict() for item in updated],
    }
    return registry_version_id, payload, updated


def publish_registry_versions(
    *,
    models_root: Path,
    old_registry_path: Path,
    registry_version_id: str,
    new_payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Preserve old bytes and publish the immutable new registry version."""

    root = models_root / "registry_versions"
    root.mkdir(parents=True, exist_ok=True)
    old_hash = file_sha256(old_registry_path)
    old_version = root / f"registry_source_{old_hash[:24]}.json"
    new_version = root / f"{registry_version_id}.json"
    _publish_bytes(old_version, old_registry_path.read_bytes())
    _publish_json(new_version, new_payload)
    return old_version, new_version


def switch_registry_atomically(registry_path: Path, version_path: Path) -> None:
    """Atomically replace registry.json with exact immutable version bytes."""

    _replace_bytes_atomically(registry_path, version_path.read_bytes())


def restore_registry_atomically(registry_path: Path, old_version_path: Path) -> None:
    """Restore the exact pre-apply registry bytes after a failed transaction."""

    _replace_bytes_atomically(registry_path, old_version_path.read_bytes())


def load_registry_records(path: Path) -> tuple[RegisteredModel, ...]:
    """Validate and load records from a current or versioned registry."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid registry JSON: {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DataValidationError(f"invalid registry schema: {path}")
    raw = payload.get("models")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise DataValidationError(f"registry models are invalid: {path}")
    records = tuple(RegisteredModel.from_dict(item) for item in raw)
    ids = [item.model_id for item in records]
    if len(ids) != len(set(ids)):
        raise DataValidationError("registry version contains duplicate models")
    champion_types = [item.model_type for item in records if item.status == "champion"]
    if len(champion_types) != len(set(champion_types)):
        raise DataValidationError("registry version contains duplicate champions")
    return records


def _publish_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        expected = canonical_payload_hash(payload)
        existing = _load_json(path)
        if canonical_payload_hash(existing) != expected:
            raise DataValidationError(f"immutable registry version differs: {path}")
        return
    atomic_write_json(path, payload)


def _publish_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise DataValidationError(f"immutable registry source differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, value = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(value)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _replace_bytes_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, value = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(value)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataValidationError(f"registry version must contain an object: {path}")
    return payload
