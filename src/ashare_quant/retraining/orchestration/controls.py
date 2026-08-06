"""Fail-closed operational budget and lifecycle cooldown controls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash
from ashare_quant.retraining.orchestration.schemas import LifecycleSnapshot
from ashare_quant.retraining.orchestration.storage import LifecycleStorage


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    operational_date: str
    configured_limit: int
    observed_attempts_before: int
    allowed: bool
    counted_lifecycle_run_ids: tuple[str, ...]
    counted_event_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CooldownDecision:
    operational_date: str
    cooldown_days: int
    allowed: bool
    previous_lifecycle_run_id: str | None = None
    previous_training_date: str | None = None
    cooldown_expiry_date: str | None = None


class LifecycleOperationalControls:
    """Inspect complete lifecycle histories without mutating them."""

    def __init__(
        self,
        *,
        storage: LifecycleStorage,
        timezone: str,
        max_daily_training_runs: int,
        cooldown_days: int,
        now: datetime,
    ) -> None:
        self.storage = storage
        self.zone = ZoneInfo(timezone)
        self.limit = max_daily_training_runs
        self.cooldown_days = cooldown_days
        self.now = now

    @property
    def operational_date(self) -> date:
        return self.now.astimezone(self.zone).date()

    def budget(self) -> BudgetDecision:
        attempts: list[tuple[str, str]] = []
        for snapshot in self._snapshots():
            for event in snapshot.events:
                if event.state != "TRAINING":
                    continue
                if _event_operational_date(event.created_at, self.zone) != self.operational_date:
                    continue
                attempts.append(
                    (
                        snapshot.summary.lifecycle_run_id,
                        canonical_payload_hash(event.model_dump(mode="json")),
                    )
                )
        return BudgetDecision(
            operational_date=self.operational_date.isoformat(),
            configured_limit=self.limit,
            observed_attempts_before=len(attempts),
            allowed=len(attempts) < self.limit,
            counted_lifecycle_run_ids=tuple(value[0] for value in attempts),
            counted_event_hashes=tuple(value[1] for value in attempts),
        )

    def cooldown(
        self,
        *,
        lifecycle_run_id: str,
        parent_model_id: str,
        horizon: int,
    ) -> CooldownDecision:
        if self.cooldown_days == 0:
            return CooldownDecision(self.operational_date.isoformat(), 0, True)
        latest: tuple[date, str] | None = None
        for snapshot in self._snapshots():
            if snapshot.summary.lifecycle_run_id == lifecycle_run_id:
                continue
            if (
                snapshot.summary.parent_model_id != parent_model_id
                or snapshot.summary.horizon != horizon
            ):
                continue
            for event in snapshot.events:
                if event.state != "TRAINING":
                    continue
                trained = _event_operational_date(event.created_at, self.zone)
                if latest is None or trained > latest[0]:
                    latest = (trained, snapshot.summary.lifecycle_run_id)
        if latest is None:
            return CooldownDecision(self.operational_date.isoformat(), self.cooldown_days, True)
        expiry = latest[0] + timedelta(days=self.cooldown_days)
        return CooldownDecision(
            operational_date=self.operational_date.isoformat(),
            cooldown_days=self.cooldown_days,
            allowed=self.operational_date >= expiry,
            previous_lifecycle_run_id=latest[1],
            previous_training_date=latest[0].isoformat(),
            cooldown_expiry_date=expiry.isoformat(),
        )

    def _snapshots(self) -> tuple[LifecycleSnapshot, ...]:
        if not self.storage.root.is_dir():
            return ()
        snapshots: list[LifecycleSnapshot] = []
        for directory in sorted(path for path in self.storage.root.iterdir() if path.is_dir()):
            if directory.name == ".tmp":
                continue
            try:
                snapshot = self.storage.read(directory.name)
            except (OSError, ValueError, DataValidationError) as error:
                raise DataValidationError(
                    "training budget history is unreadable; recovery required: "
                    f"{directory}: {error}"
                ) from error
            if snapshot is None:
                raise DataValidationError(
                    f"training budget history is incomplete; recovery required: {directory}"
                )
            snapshots.append(snapshot)
        return tuple(snapshots)


def _event_operational_date(value: str, zone: ZoneInfo) -> date:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise DataValidationError(f"invalid lifecycle event timestamp: {value}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(zone).date()
