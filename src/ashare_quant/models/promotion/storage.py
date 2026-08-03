"""Append-only atomic storage for promotion governance bundles."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.schemas import (
    DeploymentContract,
    EvidenceSnapshot,
    PromotionBundleManifest,
    PromotionRequest,
)
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.utils.manifest import atomic_write_json


@dataclass(frozen=True, slots=True)
class PromotionBundle:
    """One complete, validated governance bundle."""

    request: PromotionRequest
    evidence: EvidenceSnapshot
    contract: DeploymentContract
    manifest: PromotionBundleManifest
    output_dir: Path


class PromotionStorage:
    """Publish and read immutable promotion requests under the models root."""

    def __init__(self, models_root: Path) -> None:
        self.root = models_root / "promotion_requests"

    def output_dir(self, request_id: str) -> Path:
        """Return the immutable directory for one logical request."""

        return self.root / request_id

    def publish(
        self,
        *,
        request: PromotionRequest,
        evidence: EvidenceSnapshot,
        contract: DeploymentContract,
        identity_hash: str,
    ) -> PromotionBundle:
        """Atomically publish all payloads and write the completion manifest last."""

        output_dir = self.output_dir(request.request_id)
        existing = self.read(request.request_id)
        if existing is not None:
            if existing.manifest.identity_hash != identity_hash:
                raise DataValidationError(
                    "promotion request identity differs from immutable existing request"
                )
            return existing
        if output_dir.exists():
            raise DataValidationError(
                f"incomplete promotion request directory cannot be overwritten: {output_dir}"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=f".{request.request_id}."))
        try:
            atomic_write_json(staging / "promotion_request.json", request.model_dump(mode="json"))
            atomic_write_json(staging / "evidence_snapshot.json", evidence.model_dump(mode="json"))
            atomic_write_json(
                staging / "deployment_contract.json", contract.model_dump(mode="json")
            )
            hashes = {
                name: file_sha256(staging / name)
                for name in (
                    "deployment_contract.json",
                    "evidence_snapshot.json",
                    "promotion_request.json",
                )
            }
            manifest = PromotionBundleManifest(
                request_id=request.request_id,
                identity_hash=identity_hash,
                artifact_hashes=hashes,
                created_time=request.created_time,
            )
            atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.replace(staging, output_dir)
            bundle = self.read(request.request_id)
            if bundle is None:
                raise DataValidationError("published promotion request is incomplete")
            return bundle
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def read(self, request_id: str) -> PromotionBundle | None:
        """Read a complete bundle; manifest absence means incomplete publication."""

        output_dir = self.output_dir(request_id)
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            return None
        try:
            manifest = PromotionBundleManifest.model_validate(_load_json(manifest_path))
            request = PromotionRequest.model_validate(
                _load_json(output_dir / "promotion_request.json")
            )
            evidence = EvidenceSnapshot.model_validate(
                _load_json(output_dir / "evidence_snapshot.json")
            )
            contract = DeploymentContract.model_validate(
                _load_json(output_dir / "deployment_contract.json")
            )
        except ValidationError as error:
            raise DataValidationError(f"invalid promotion governance schema: {error}") from error
        if manifest.request_id != request_id or request.request_id != request_id:
            raise DataValidationError("promotion request directory identity is inconsistent")
        for name, expected_hash in manifest.artifact_hashes.items():
            if file_sha256(output_dir / name) != expected_hash:
                raise DataValidationError(f"promotion governance artifact changed: {name}")
        return PromotionBundle(request, evidence, contract, manifest, output_dir)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"promotion governance artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid promotion governance JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"promotion governance artifact must be an object: {path}")
    return payload
