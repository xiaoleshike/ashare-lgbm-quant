"""Transactional append-only storage for governed training requests."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pydantic import ValidationError

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import PromotionGatePolicy
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.alerts.storage import replace_targets_atomically
from ashare_quant.retraining.configuration import RetrainingPolicy
from ashare_quant.retraining.schemas import (
    ModelRole,
    RetrainingDecision,
    RetrainingEvidence,
    TrainingRequest,
    TrainingRequestManifest,
    TrainingTarget,
    TriggerReason,
)
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

Clock = Callable[[], datetime]
HISTORY_COLUMNS = (
    "request_id",
    "model_id",
    "model_role",
    "horizon",
    "as_of",
    "policy_hash",
    "evidence_hash",
    "trigger_reasons",
    "generation_mode",
    "created_at",
    "request_path",
    "manifest_hash",
)


class RetrainingRequestStorage:
    """Publish immutable requests and one append-only logical history transactionally."""

    def __init__(
        self,
        *,
        reports_root: Path,
        config_path: Path,
        policy: RetrainingPolicy,
        promotion_policy: PromotionGatePolicy,
        clock: Clock | None = None,
    ) -> None:
        self.reports_root = reports_root
        self.config_path = config_path
        self.policy = policy
        self.promotion_policy = promotion_policy
        self.clock = clock or (lambda: datetime.now(UTC))
        self.root = reports_root / "retraining"
        self.requests_root = self.root / "requests"
        self.history_path = self.root / "history" / "retraining_requests.parquet"

    def create(
        self,
        *,
        decision: RetrainingDecision,
        evidence: RetrainingEvidence,
        evidence_hash: str,
        as_of: str,
        generation_mode: str,
    ) -> RetrainingDecision:
        """Create or idempotently return one request, respecting cooldown."""

        identity = {
            "model_id": decision.model_id,
            "model_role": decision.model_role,
            "horizon": decision.horizon,
            "policy_hash": self.policy.policy_hash,
            "evidence_hash": evidence_hash,
            "generation_mode": generation_mode,
            "promotion_policy_hash": self.promotion_policy.policy_hash,
        }
        request_id = f"training_{canonical_payload_hash(identity)[:24]}"
        output_dir = self.requests_root / request_id
        existing = self.read(request_id)
        if existing is not None:
            request, _ = existing
            target = request.target_models[0]
            if (
                target.model_id != decision.model_id
                or target.model_role != decision.model_role
                or target.horizon != decision.horizon
                or request.policy_hash != self.policy.policy_hash
                or request.evidence_hash != evidence_hash
                or request.generation_mode != generation_mode
                or request.promotion_policy_hash != self.promotion_policy.policy_hash
            ):
                raise DataValidationError("immutable retraining request identity differs")
            if request_id not in set(self.history()["request_id"].astype(str)):
                raise DataValidationError("retraining request is missing from append-only history")
            return replace(
                decision,
                request_id=request_id,
                output_dir=output_dir,
                idempotent=True,
            )
        cooldown = self._cooldown(decision)
        if cooldown is not None:
            return replace(
                decision,
                status="NO_ACTION_REQUIRED",
                reasons=(f"cooldown_active_since:{cooldown}",),
            )
        created_at = self.clock().astimezone(UTC).isoformat()
        request = TrainingRequest(
            request_id=request_id,
            created_at=created_at,
            as_of=as_of,
            target_models=(
                TrainingTarget(
                    model_id=decision.model_id,
                    model_role=cast(ModelRole, decision.model_role),
                    horizon=cast(Any, decision.horizon),
                ),
            ),
            trigger_reason=cast(tuple[TriggerReason, ...], decision.reasons),
            evidence=evidence,
            evidence_hash=evidence_hash,
            policy_hash=self.policy.policy_hash,
            policy_version=self.policy.policy_version,
            promotion_policy_hash=self.promotion_policy.policy_hash,
            promotion_policy_version=self.promotion_policy.policy_version,
            generation_mode=cast(Any, generation_mode),
        )
        self._publish(request)
        return replace(decision, request_id=request_id, output_dir=output_dir)

    def read(self, request_id: str) -> tuple[TrainingRequest, TrainingRequestManifest] | None:
        output = self.requests_root / request_id
        request_path = output / "training_request.json"
        manifest_path = output / "manifest.json"
        if not request_path.is_file() and not manifest_path.is_file():
            return None
        if not request_path.is_file() or not manifest_path.is_file():
            raise DataValidationError(f"incomplete retraining request: {request_id}")
        try:
            request = TrainingRequest.model_validate(_json(request_path))
            manifest = TrainingRequestManifest.model_validate(_json(manifest_path))
        except ValidationError as error:
            raise DataValidationError(
                f"invalid retraining request {request_id}: {error}"
            ) from error
        if request.request_id != request_id or manifest.request_id != request_id:
            raise DataValidationError("retraining request directory identity mismatch")
        if file_sha256(request_path) != manifest.request_file_sha256:
            raise DataValidationError("retraining request payload hash mismatch")
        target = request.target_models[0]
        expected_evidence_hashes = {
            "monitor_snapshot": request.evidence.monitor_snapshot.sha256,
            "performance_observation": request.evidence.performance_observation.sha256,
            "alerts": request.evidence.alerts.sha256,
        }
        if (
            manifest.model_id != target.model_id
            or manifest.model_role != target.model_role
            or manifest.horizon != target.horizon
            or manifest.trigger_reasons != request.trigger_reason
            or manifest.evidence_hashes != expected_evidence_hashes
            or manifest.evidence_hash != request.evidence_hash
            or manifest.policy_hash != request.policy_hash
            or manifest.policy_version != request.policy_version
            or manifest.promotion_policy_hash != request.promotion_policy_hash
            or manifest.promotion_policy_version != request.promotion_policy_version
            or manifest.generated_at != request.created_at
        ):
            raise DataValidationError("retraining request and manifest identities differ")
        return request, manifest

    def history(self) -> pd.DataFrame:
        if not self.history_path.is_file():
            return pd.DataFrame(columns=list(HISTORY_COLUMNS))
        frame = pd.read_parquet(self.history_path)
        missing = sorted(set(HISTORY_COLUMNS) - set(frame.columns))
        if missing or frame["request_id"].duplicated().any():
            raise DataValidationError(f"invalid retraining history: missing={missing}")
        return frame.loc[:, list(HISTORY_COLUMNS)].sort_values(
            ["created_at", "request_id"], kind="mergesort"
        )

    def _cooldown(self, decision: RetrainingDecision) -> str | None:
        if self.policy.cooldown_days == 0:
            return None
        history = self.history()
        matches = history.loc[
            history["model_id"].astype(str).eq(decision.model_id)
            & pd.to_numeric(history["horizon"], errors="coerce").eq(decision.horizon)
        ]
        if matches.empty:
            return None
        for row in reversed(list(matches.to_dict("records"))):
            request_id = str(row["request_id"])
            stored = self.read(request_id)
            if stored is None:
                raise DataValidationError(f"retraining history request is missing: {request_id}")
            request, _ = stored
            if request.promotion_policy_hash != self.promotion_policy.policy_hash:
                continue
            latest = str(row["created_at"])
            created = datetime.fromisoformat(latest).astimezone(UTC)
            return (
                latest
                if self.clock().astimezone(UTC)
                < created + timedelta(days=self.policy.cooldown_days)
                else None
            )
        return None

    def _publish(self, request: TrainingRequest) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        staging_root = Path(tempfile.mkdtemp(dir=self.root, prefix=".retraining-request-"))
        try:
            staged_request = staging_root / request.request_id
            staged_request.mkdir()
            request_path = staged_request / "training_request.json"
            atomic_write_json(request_path, request.model_dump(mode="json"))
            target = request.target_models[0]
            git = current_git_info()
            manifest = TrainingRequestManifest(
                request_id=request.request_id,
                model_id=target.model_id,
                model_role=target.model_role,
                horizon=target.horizon,
                trigger_reasons=request.trigger_reason,
                evidence_hashes={
                    "monitor_snapshot": request.evidence.monitor_snapshot.sha256,
                    "performance_observation": request.evidence.performance_observation.sha256,
                    "alerts": request.evidence.alerts.sha256,
                },
                evidence_hash=request.evidence_hash,
                policy_hash=request.policy_hash,
                policy_version=request.policy_version,
                promotion_policy_hash=request.promotion_policy_hash,
                promotion_policy_version=request.promotion_policy_version,
                git_commit=git["commit"],
                git_dirty=bool(git["dirty"]),
                config_hash=config_hash(self.config_path),
                generated_at=request.created_at,
                request_file_sha256=file_sha256(request_path),
            )
            atomic_write_json(staged_request / "manifest.json", manifest.model_dump(mode="json"))
            history = self.history()
            row = {
                "request_id": request.request_id,
                "model_id": target.model_id,
                "model_role": target.model_role,
                "horizon": target.horizon,
                "as_of": request.as_of,
                "policy_hash": request.policy_hash,
                "evidence_hash": request.evidence_hash,
                "trigger_reasons": json.dumps(list(request.trigger_reason), separators=(",", ":")),
                "generation_mode": request.generation_mode,
                "created_at": request.created_at,
                "request_path": str(
                    (self.requests_root / request.request_id / "training_request.json").relative_to(
                        self.reports_root
                    )
                ),
                "manifest_hash": file_sha256(staged_request / "manifest.json"),
            }
            updated = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
            updated = updated.loc[:, list(HISTORY_COLUMNS)].sort_values(
                ["created_at", "request_id"], kind="mergesort"
            )
            staged_history = staging_root / "retraining_requests.parquet"
            updated.to_parquet(staged_history, index=False)
            replace_targets_atomically(
                (
                    (staged_request, self.requests_root / request.request_id),
                    (staged_history, self.history_path),
                ),
                backup_root=staging_root / "backups",
            )
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid retraining JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"retraining JSON must contain an object: {path}")
    return payload
