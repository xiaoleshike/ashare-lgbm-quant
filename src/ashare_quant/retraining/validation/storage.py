"""Atomic immutable publication for retraining validation evidence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.validation.schemas import (
    ExecutableValidationEvidence,
    OfflineValidationEvidence,
    RetrainingValidationManifest,
    RetrainingValidationResult,
    ShadowEligibilityEvidence,
    ValidationEvidence,
)
from ashare_quant.utils.manifest import atomic_write_json


class RetrainingValidationStorage:
    def __init__(self, reports_root: Path) -> None:
        self.root = reports_root / "retraining_validation"

    def existing(
        self, run_id: str, identity: str, model_id: str
    ) -> RetrainingValidationResult | None:
        output = self.root / run_id
        if not output.exists():
            return None
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise DataValidationError("incomplete retraining validation artifact exists")
        manifest = RetrainingValidationManifest.model_validate(_json(manifest_path))
        if manifest.validation_identity != identity or manifest.model_id != model_id:
            raise DataValidationError("immutable retraining validation identity differs")
        _validate_hashes(output, manifest)
        return RetrainingValidationResult(
            run_id,
            model_id,
            "COMPLETED",
            manifest.promotion_ready,
            output,
            True,
        )

    def publish(
        self,
        *,
        manifest: RetrainingValidationManifest,
        offline: OfflineValidationEvidence,
        executable: ExecutableValidationEvidence,
        shadow: ShadowEligibilityEvidence,
        evidence: ValidationEvidence,
    ) -> RetrainingValidationResult:
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".validation-"))
        output = self.root / manifest.run_id
        try:
            (staging / "offline").mkdir()
            (staging / "executable").mkdir()
            (staging / "shadow").mkdir()
            atomic_write_json(staging / "offline" / "metrics.json", offline.model_dump(mode="json"))
            atomic_write_json(
                staging / "executable" / "summary.json",
                executable.model_dump(mode="json"),
            )
            atomic_write_json(
                staging / "shadow" / "eligibility.json", shadow.model_dump(mode="json")
            )
            atomic_write_json(staging / "evidence.json", evidence.model_dump(mode="json"))
            expected = manifest.model_copy(
                update={
                    "offline_validation_hash": file_sha256(staging / "offline" / "metrics.json"),
                    "executable_validation_hash": file_sha256(
                        staging / "executable" / "summary.json"
                    ),
                    "shadow_eligibility_hash": file_sha256(staging / "shadow" / "eligibility.json"),
                    "evidence_hash": file_sha256(staging / "evidence.json"),
                }
            )
            atomic_write_json(staging / "manifest.json", expected.model_dump(mode="json"))
            _validate_hashes(staging, expected)
            if output.exists():
                raise DataValidationError("immutable retraining validation already exists")
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return RetrainingValidationResult(
            manifest.run_id,
            manifest.model_id,
            "COMPLETED",
            manifest.promotion_ready,
            output,
        )

    def status(self, run_id: str) -> dict[str, object]:
        output = self.root / run_id
        if not (output / "manifest.json").is_file():
            return {"run_id": run_id, "status": "MISSING"}
        manifest = RetrainingValidationManifest.model_validate(_json(output / "manifest.json"))
        _validate_hashes(output, manifest)
        return {
            "run_id": run_id,
            "model_id": manifest.model_id,
            "status": "COMPLETED",
            "promotion_ready": manifest.promotion_ready,
            "output": str(output),
        }


def _validate_hashes(directory: Path, manifest: RetrainingValidationManifest) -> None:
    expected = {
        directory / "offline" / "metrics.json": manifest.offline_validation_hash,
        directory / "executable" / "summary.json": manifest.executable_validation_hash,
        directory / "shadow" / "eligibility.json": manifest.shadow_eligibility_hash,
        directory / "evidence.json": manifest.evidence_hash,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise DataValidationError(f"retraining validation evidence hash mismatch: {path}")


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid retraining validation JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"retraining validation JSON must be an object: {path}")
    return payload
