"""Read-only human review workflow over immutable request and gate artifacts."""

from __future__ import annotations

import getpass
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ashare_quant.config.settings import PromotionReviewSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.approval_schema import ApprovalEvent, ApprovalEventType
from ashare_quant.models.promotion.approval_storage import (
    ApprovalEventStorage,
    approval_event_identity,
)
from ashare_quant.models.promotion.gate_report import read_gate_result
from ashare_quant.models.promotion.review_policy import ReviewPolicy, parse_timestamp
from ashare_quant.models.promotion.storage import PromotionStorage
from ashare_quant.models.promotion.validation import validate_bundle
from ashare_quant.models.shadow.storage import file_sha256

type Clock = Callable[[], datetime]
type ReviewerProvider = Callable[[], str]


@dataclass(frozen=True, slots=True)
class ReviewWorkflowResult:
    """Public status for review readiness and immutable decisions."""

    request_id: str
    status: str
    reviewer: str
    requester: str
    event_id: str | None
    event_path: Path | None
    expires_at: str | None
    idempotent: bool = False


class HumanReviewService:
    """Validate and append human decisions without modifying model state."""

    def __init__(
        self,
        *,
        models_root: Path,
        reports_root: Path,
        settings: PromotionReviewSettings,
        reviewer_provider: ReviewerProvider | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.models_root = models_root
        self.reports_root = reports_root
        self.request_storage = PromotionStorage(models_root)
        self.event_storage = ApprovalEventStorage(models_root)
        self.policy = ReviewPolicy(settings)
        self.reviewer_provider = reviewer_provider or getpass.getuser
        self.clock = clock or (lambda: datetime.now(UTC))

    def review(self, request_id: str) -> ReviewWorkflowResult:
        """Validate that a request is ready for an authorized human decision."""

        reviewer, requester, _, _, _ = self._review_inputs(request_id, approving=False)
        existing = self.event_storage.list_events(request_id)
        if existing:
            return self.status(request_id)
        return ReviewWorkflowResult(
            request_id=request_id,
            status="READY_FOR_REVIEW",
            reviewer=reviewer,
            requester=requester,
            event_id=None,
            event_path=None,
            expires_at=None,
        )

    def approve(self, request_id: str, comments: str) -> ReviewWorkflowResult:
        """Append an APPROVED event after all immutable bindings pass."""

        return self._decide(request_id, "APPROVED", comments)

    def reject(self, request_id: str, comments: str) -> ReviewWorkflowResult:
        """Append a REJECTED event without changing registry or Champion state."""

        return self._decide(request_id, "REJECTED", comments)

    def status(self, request_id: str) -> ReviewWorkflowResult:
        """Validate the current terminal event and report expiry or invalidation."""

        events = self.event_storage.list_events(request_id)
        if not events:
            return self.review(request_id)
        if len(events) != 1:
            raise DataValidationError("promotion request has conflicting review events")
        stored = events[0]
        event = stored.event
        status: str = event.event_type
        request_path, gate_path, registry_path = self._bound_paths(request_id)
        if stored.manifest.policy_hash != self.policy.policy_hash:
            status = "INVALID"
        elif (
            file_sha256(request_path) != event.request_hash
            or file_sha256(gate_path) != event.gate_result_hash
            or file_sha256(registry_path) != event.registry_hash_at_review
        ):
            status = "INVALID"
        elif event.event_type == "APPROVED" and self.clock() > parse_timestamp(event.expires_at):
            status = "APPROVAL_EXPIRED"
        return ReviewWorkflowResult(
            request_id=request_id,
            status=status,
            reviewer=event.reviewer,
            requester=event.requester,
            event_id=event.event_id,
            event_path=stored.event_path,
            expires_at=event.expires_at,
            idempotent=True,
        )

    def _decide(
        self, request_id: str, event_type: ApprovalEventType, comments: str
    ) -> ReviewWorkflowResult:
        clean_comments = comments.strip()
        if not clean_comments:
            raise DataValidationError("review comments must not be empty")
        reviewer, requester, request_hash, gate_hash, registry_hash = self._review_inputs(
            request_id, approving=event_type == "APPROVED"
        )
        now = self.clock().astimezone(UTC)
        expires = self.policy.expires_at(now)
        provisional = ApprovalEvent(
            event_id="pending",
            event_type=event_type,
            request_id=request_id,
            request_hash=request_hash,
            gate_result_hash=gate_hash,
            registry_hash_at_review=registry_hash,
            reviewer=reviewer,
            requester=requester,
            decision="approved" if event_type == "APPROVED" else "rejected",
            comments=clean_comments,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )
        identity_hash = approval_event_identity(provisional, self.policy.policy_hash)
        event = provisional.model_copy(update={"event_id": f"review_{identity_hash[:24]}"})
        stored, idempotent = self.event_storage.publish(
            event=event,
            event_identity_hash=identity_hash,
            policy_hash=self.policy.policy_hash,
        )
        if file_sha256(self.models_root / "registry.json") != registry_hash:
            raise DataValidationError("registry changed during immutable human review")
        return ReviewWorkflowResult(
            request_id=request_id,
            status=stored.event.event_type,
            reviewer=stored.event.reviewer,
            requester=stored.event.requester,
            event_id=stored.event.event_id,
            event_path=stored.event_path,
            expires_at=stored.event.expires_at,
            idempotent=idempotent,
        )

    def _review_inputs(self, request_id: str, *, approving: bool) -> tuple[str, str, str, str, str]:
        bundle = self.request_storage.read(request_id)
        if bundle is None:
            raise DataValidationError(f"complete promotion request does not exist: {request_id}")
        validate_bundle(bundle, self.reports_root, self.models_root / "registry.json")
        gate_dir = self.reports_root / "promotion_gate" / request_id
        stored_gate = read_gate_result(gate_dir)
        if stored_gate is None:
            raise DataValidationError("promotion gate result is missing")
        gate_result, gate_manifest = stored_gate
        if approving and gate_result.status == "FAIL":
            raise DataValidationError("FAIL promotion gate cannot be approved")
        reviewer = self.reviewer_provider()
        requester = bundle.request.requester
        self.policy.validate_reviewer(requester=requester, reviewer=reviewer)
        self.policy.validate_request_not_expired(bundle.request.created_time, self.clock())
        request_path, gate_path, registry_path = self._bound_paths(request_id)
        if gate_manifest.source_request_manifest_hash != file_sha256(
            bundle.output_dir / "manifest.json"
        ):
            raise DataValidationError("gate result is not bound to current promotion request")
        registry_hash = file_sha256(registry_path)
        if registry_hash != bundle.request.registry_hash:
            raise DataValidationError("registry changed after promotion request")
        return (
            reviewer,
            requester,
            file_sha256(request_path),
            file_sha256(gate_path),
            registry_hash,
        )

    def _bound_paths(self, request_id: str) -> tuple[Path, Path, Path]:
        return (
            self.models_root / "promotion_requests" / request_id / "promotion_request.json",
            self.reports_root / "promotion_gate" / request_id / "gate_result.json",
            self.models_root / "registry.json",
        )
