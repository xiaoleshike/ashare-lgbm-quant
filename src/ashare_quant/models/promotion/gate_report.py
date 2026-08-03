"""Immutable atomic publication of promotion gate results."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_schemas import GateManifest, GateResult
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.utils.manifest import atomic_write_json


def publish_gate_result(
    *,
    reports_root: Path,
    result: GateResult,
    gate_identity: str,
    source_request_manifest_hash: str,
) -> tuple[Path, bool]:
    """Publish one immutable gate report, with the manifest written last."""

    output_dir = reports_root / "promotion_gate" / result.request_id
    existing = read_gate_result(output_dir)
    if existing is not None:
        _, manifest = existing
        if manifest.gate_identity != gate_identity:
            raise DataValidationError(
                "promotion gate output has a different immutable evaluation identity"
            )
        return output_dir, True
    if output_dir.exists():
        raise DataValidationError(
            f"incomplete promotion gate output cannot be overwritten: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{result.request_id}."))
    try:
        atomic_write_json(staging / "gate_result.json", result.model_dump(mode="json"))
        (staging / "gate_report.md").write_text(_render(result), encoding="utf-8")
        hashes = {
            "gate_result.json": file_sha256(staging / "gate_result.json"),
            "gate_report.md": file_sha256(staging / "gate_report.md"),
        }
        manifest = GateManifest(
            request_id=result.request_id,
            gate_identity=gate_identity,
            status=result.status,
            policy_hash=result.policy_hash,
            source_request_manifest_hash=source_request_manifest_hash,
            artifact_hashes=hashes,
            created_at=result.created_at,
        )
        atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        os.replace(staging, output_dir)
        if read_gate_result(output_dir) is None:
            raise DataValidationError("published promotion gate result is incomplete")
        return output_dir, False
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_gate_result(output_dir: Path) -> tuple[GateResult, GateManifest] | None:
    """Read and physically validate one complete gate publication."""

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = GateManifest.model_validate(_load_json(manifest_path))
        result = GateResult.model_validate(_load_json(output_dir / "gate_result.json"))
    except ValidationError as error:
        raise DataValidationError(f"invalid promotion gate schema: {error}") from error
    if result.request_id != manifest.request_id or result.status != manifest.status:
        raise DataValidationError("promotion gate result and manifest identities differ")
    for name, digest in manifest.artifact_hashes.items():
        if file_sha256(output_dir / name) != digest:
            raise DataValidationError(f"promotion gate artifact changed: {name}")
    return result, manifest


def _render(result: GateResult) -> str:
    lines = [
        "# Model Promotion Gate",
        "",
        f"- Request: `{result.request_id}`",
        f"- Candidate: `{result.candidate_model_id}`",
        f"- Status: **{result.status}**",
        "",
        "## Checks",
        "",
        "| Check | Status | Evidence | Message |",
        "|---|---:|---|---|",
    ]
    lines.extend(
        f"| {check.name} | {check.status} | `{check.evidence_hash[:12]}` | "
        f"{check.message.replace('|', '/')} |"
        for check in result.checks
    )
    lines.extend(
        [
            "",
            "This gate only determines eligibility for human review. It does not promote a model.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"promotion gate artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid promotion gate JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"promotion gate artifact must contain an object: {path}")
    return payload
