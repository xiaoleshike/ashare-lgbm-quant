"""Atomic append-only storage for qualification authorization artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.qualification.authorization_schemas import (
    AuthorizationArtifactManifest,
    AuthorizationConsumptionClaim,
    AuthorizationConsumptionReceipt,
    AuthorizationRevocation,
    AuthorizationStage,
    QualificationAuthorization,
)
from ashare_quant.utils.manifest import atomic_write_json

T = TypeVar("T", bound=BaseModel)


class QualificationAuthorizationStorage:
    """Store authorizations, revocations, and single-use consumption records."""

    def __init__(self, qualification_root: Path) -> None:
        self.qualification_root = qualification_root

    def authorization_dir(
        self, run_id: str, stage: AuthorizationStage, authorization_id: str
    ) -> Path:
        return self.qualification_root / run_id / "authorizations" / stage / authorization_id

    def revocation_dir(self, run_id: str, authorization_id: str, revocation_id: str) -> Path:
        return (
            self.qualification_root
            / run_id
            / "authorization_revocations"
            / authorization_id
            / revocation_id
        )

    def claim_dir(self, run_id: str, authorization_id: str, consumption_id: str) -> Path:
        return (
            self.qualification_root
            / run_id
            / "authorization_consumptions"
            / authorization_id
            / consumption_id
        )

    def receipt_dir(
        self, run_id: str, authorization_id: str, consumption_id: str, receipt_id: str
    ) -> Path:
        return self.claim_dir(run_id, authorization_id, consumption_id) / "receipts" / receipt_id

    def publish_authorization(self, authorization: QualificationAuthorization) -> tuple[Path, bool]:
        output = self.authorization_dir(
            authorization.qualification_run_id,
            authorization.stage,
            authorization.authorization_id,
        )
        return self._publish(
            output,
            "authorization.json",
            authorization,
            "qualification_authorization_manifest",
            authorization.authorization_id,
        )

    def publish_revocation(self, revocation: AuthorizationRevocation) -> tuple[Path, bool]:
        output = self.revocation_dir(
            revocation.qualification_run_id,
            revocation.authorization_id,
            revocation.revocation_id,
        )
        return self._publish(
            output,
            "revocation.json",
            revocation,
            "qualification_revocation_manifest",
            revocation.revocation_id,
        )

    def publish_claim(self, claim: AuthorizationConsumptionClaim) -> tuple[Path, bool]:
        output = self.claim_dir(
            claim.qualification_run_id, claim.authorization_id, claim.consumption_id
        )
        return self._publish(
            output,
            "claim.json",
            claim,
            "qualification_consumption_claim_manifest",
            claim.consumption_id,
        )

    def publish_receipt(self, receipt: AuthorizationConsumptionReceipt) -> tuple[Path, bool]:
        output = self.receipt_dir(
            receipt.qualification_run_id,
            receipt.authorization_id,
            receipt.consumption_id,
            receipt.receipt_id,
        )
        return self._publish(
            output,
            "receipt.json",
            receipt,
            "qualification_consumption_receipt_manifest",
            receipt.receipt_id,
        )

    def authorizations(
        self, run_id: str, stage: AuthorizationStage | None = None
    ) -> tuple[tuple[QualificationAuthorization, Path, str], ...]:
        root = self.qualification_root / run_id / "authorizations"
        stages = (stage,) if stage is not None else ("training", "shadow")
        records: list[tuple[QualificationAuthorization, Path, str]] = []
        for current_stage in stages:
            stage_root = root / current_stage
            if not stage_root.exists():
                continue
            for directory in sorted(path for path in stage_root.iterdir() if path.is_dir()):
                value, digest = self._read(
                    directory, "authorization.json", QualificationAuthorization
                )
                if value.stage != current_stage or value.authorization_id != directory.name:
                    raise DataValidationError(f"authorization path identity mismatch: {directory}")
                records.append((value, directory, digest))
        return tuple(records)

    def authorization(
        self, run_id: str, authorization_id: str
    ) -> tuple[QualificationAuthorization, Path, str]:
        matches = [
            item
            for item in self.authorizations(run_id)
            if item[0].authorization_id == authorization_id
        ]
        if len(matches) != 1:
            raise DataValidationError(
                f"qualification authorization identity is missing or ambiguous: {authorization_id}"
            )
        return matches[0]

    def revocations(
        self, run_id: str, authorization_id: str
    ) -> tuple[tuple[AuthorizationRevocation, Path, str], ...]:
        root = self.qualification_root / run_id / "authorization_revocations" / authorization_id
        if not root.exists():
            return ()
        records = []
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            value, digest = self._read(directory, "revocation.json", AuthorizationRevocation)
            records.append((value, directory, digest))
        return tuple(records)

    def claims(
        self, run_id: str, authorization_id: str
    ) -> tuple[tuple[AuthorizationConsumptionClaim, Path, str], ...]:
        root = self.qualification_root / run_id / "authorization_consumptions" / authorization_id
        if not root.exists():
            return ()
        records = []
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            value, digest = self._read(directory, "claim.json", AuthorizationConsumptionClaim)
            records.append((value, directory, digest))
        return tuple(records)

    def receipts(
        self, run_id: str, authorization_id: str, consumption_id: str
    ) -> tuple[tuple[AuthorizationConsumptionReceipt, Path, str], ...]:
        root = self.claim_dir(run_id, authorization_id, consumption_id) / "receipts"
        if not root.exists():
            return ()
        records = []
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            value, digest = self._read(directory, "receipt.json", AuthorizationConsumptionReceipt)
            records.append((value, directory, digest))
        return tuple(records)

    def staging_paths(self, run_id: str) -> tuple[Path, ...]:
        output = self.qualification_root / run_id
        return tuple(sorted(output.rglob(".*.tmp-*"))) if output.exists() else ()

    def _publish(
        self,
        output: Path,
        payload_file: str,
        payload: BaseModel,
        artifact_name: Literal[
            "qualification_authorization_manifest",
            "qualification_revocation_manifest",
            "qualification_consumption_claim_manifest",
            "qualification_consumption_receipt_manifest",
        ],
        identity: str,
    ) -> tuple[Path, bool]:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            existing, _ = self._read(output, payload_file, type(payload))
            if existing != payload:
                raise DataValidationError(f"immutable authorization artifact conflict: {output}")
            return output, True
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.tmp-"))
        try:
            atomic_write_json(staging / payload_file, payload.model_dump(mode="json"))
            manifest = AuthorizationArtifactManifest(
                artifact_name=artifact_name,
                identity=identity,
                payload_file=payload_file,
                payload_sha256=file_sha256(staging / payload_file),
            )
            atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            try:
                os.replace(staging, output)
            except OSError as error:
                if output.exists():
                    existing, _ = self._read(output, payload_file, type(payload))
                    if existing == payload:
                        return output, True
                raise DataValidationError(
                    f"authorization publication failed: {output}: {error}"
                ) from error
            self._read(output, payload_file, type(payload))
            return output, False
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _read(self, output: Path, payload_file: str, schema: type[T]) -> tuple[T, str]:
        payload_path = output / payload_file
        manifest_path = output / "manifest.json"
        if not payload_path.is_file() or not manifest_path.is_file():
            raise DataValidationError(f"incomplete authorization artifact: {output}")
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataValidationError(
                f"invalid authorization artifact {output}: {error}"
            ) from error
        manifest = AuthorizationArtifactManifest.model_validate(manifest_payload)
        digest = file_sha256(payload_path)
        if manifest.payload_file != payload_file or manifest.payload_sha256 != digest:
            raise DataValidationError(f"authorization artifact hash mismatch: {output}")
        value = schema.model_validate(payload)
        if manifest.identity not in {
            getattr(value, "authorization_id", None),
            getattr(value, "revocation_id", None),
            getattr(value, "consumption_id", None),
            getattr(value, "receipt_id", None),
        }:
            raise DataValidationError(f"authorization manifest identity mismatch: {output}")
        return value, digest
