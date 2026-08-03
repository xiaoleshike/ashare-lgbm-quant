"""OS-identity authorization and expiry policy for human reviews."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ashare_quant.config.settings import PromotionReviewSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash


class ReviewPolicy:
    """Validate reviewer authorization and separation of duties."""

    def __init__(self, settings: PromotionReviewSettings) -> None:
        self.settings = settings

    @property
    def policy_hash(self) -> str:
        """Return deterministic review policy identity."""

        return canonical_payload_hash(self.settings.model_dump(mode="json"))

    def validate_reviewer(self, *, requester: str, reviewer: str) -> None:
        """Require an allowlisted OS user and, by default, a distinct requester."""

        if reviewer not in self.settings.reviewer_allowlist:
            raise DataValidationError(f"OS reviewer is not allowlisted: {reviewer}")
        if requester == "unknown":
            raise DataValidationError(
                "legacy promotion request has no requester identity and cannot be approved"
            )
        if requester == reviewer and not self.settings.allow_requester_as_reviewer:
            raise DataValidationError("requester/reviewer separation of duties is required")

    def expires_at(self, created_at: datetime) -> datetime:
        """Calculate the approval expiration timestamp."""

        return created_at + timedelta(hours=self.settings.review_expire_hours)

    def validate_request_not_expired(self, request_created_at: str, now: datetime) -> None:
        """Reject decisions on requests outside the configured review window."""

        created = parse_timestamp(request_created_at)
        if now > self.expires_at(created):
            raise DataValidationError("promotion request review window has expired")


def parse_timestamp(value: str) -> datetime:
    """Parse an aware ISO timestamp and normalize it to UTC."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise DataValidationError(f"invalid review timestamp: {value}") from error
    if parsed.tzinfo is None:
        raise DataValidationError(f"review timestamp must include timezone: {value}")
    return parsed.astimezone(UTC)
