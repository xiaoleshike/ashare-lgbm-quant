"""Read-only trigger evaluation and immutable training-request orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.retraining.configuration import RetrainingPolicy, load_retraining_policy
from ashare_quant.retraining.evaluator import evaluate_sources, select_manual_target
from ashare_quant.retraining.schemas import (
    RetrainingDecision,
    RetrainingEvaluationResult,
    RetrainingValidationResult,
)
from ashare_quant.retraining.sources import load_retraining_sources
from ashare_quant.retraining.storage import RetrainingRequestStorage
from ashare_quant.retraining.validators import evidence_hash, validate_recorded_evidence


class RetrainingTriggerService:
    """Create governed requests without invoking any model lifecycle operation."""

    def __init__(
        self,
        *,
        reports_root: Path,
        config_path: Path,
        policy_path: Path,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.reports_root = reports_root
        self.config_path = config_path
        self.policy_path = policy_path
        self.policy: RetrainingPolicy = load_retraining_policy(policy_path)
        self.storage = RetrainingRequestStorage(
            reports_root=reports_root,
            config_path=config_path,
            policy=self.policy,
            clock=clock,
        )

    def evaluate(self, as_of: str) -> RetrainingEvaluationResult:
        """Evaluate every monitored trainable horizon and freeze triggered requests."""

        sources = load_retraining_sources(self.reports_root, as_of)
        decisions: list[RetrainingDecision] = []
        paths: list[Path] = []
        for decision, _ in evaluate_sources(sources, self.policy):
            stored = (
                self.storage.create(
                    decision=decision,
                    evidence=sources.evidence,
                    evidence_hash=sources.evidence_hash,
                    as_of=as_of,
                    generation_mode="automatic",
                )
                if decision.status == "TRIGGERED"
                else decision
            )
            decisions.append(stored)
            if stored.output_dir is not None:
                paths.extend(
                    (
                        stored.output_dir / "training_request.json",
                        stored.output_dir / "manifest.json",
                    )
                )
        return RetrainingEvaluationResult(as_of, tuple(decisions), tuple(paths))

    def create_request(self, *, model_id: str, as_of: str) -> RetrainingEvaluationResult:
        """Create one manual request using the same immutable evidence and maturity gates."""

        sources = load_retraining_sources(self.reports_root, as_of)
        decision, _ = select_manual_target(sources, model_id, self.policy)
        stored = self.storage.create(
            decision=decision,
            evidence=sources.evidence,
            evidence_hash=sources.evidence_hash,
            as_of=as_of,
            generation_mode="manual",
        )
        paths = (
            (stored.output_dir / "training_request.json", stored.output_dir / "manifest.json")
            if stored.output_dir is not None
            else ()
        )
        return RetrainingEvaluationResult(as_of, (stored,), paths)

    def validate(self, request_id: str) -> RetrainingValidationResult:
        """Revalidate active policy and request-bound immutable source bytes."""

        try:
            stored = self.storage.read(request_id)
            if stored is None:
                raise DataValidationError(f"retraining request does not exist: {request_id}")
            request, manifest = stored
            if request.policy_hash != self.policy.policy_hash:
                raise DataValidationError("retraining policy hash changed after request creation")
            if request.policy_version != self.policy.policy_version:
                raise DataValidationError(
                    "retraining policy version changed after request creation"
                )
            validate_recorded_evidence(self.reports_root, request.evidence)
            if evidence_hash(request.evidence) != request.evidence_hash:
                raise DataValidationError("retraining request evidence identity is invalid")
            if manifest.evidence_hash != request.evidence_hash:
                raise DataValidationError("retraining manifest evidence identity differs")
        except (DataValidationError, OSError, ValueError) as error:
            return RetrainingValidationResult(request_id, False, "INVALID", str(error))
        return RetrainingValidationResult(request_id, True, "VALID")

    def status(self) -> tuple[dict[str, object], ...]:
        """Return deterministic request history without changing any artifact."""

        frame = self.storage.history()
        return tuple(
            {str(key): value for key, value in row.items()} for row in frame.to_dict("records")
        )
