"""Authoritative point-in-time security lifecycle events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame
type LifecycleEventType = Literal["LISTING_SUSPENDED"]


@dataclass(frozen=True, slots=True)
class SecurityLifecycleEvent:
    """One authoritative effective-dated lifecycle interval."""

    canonical_ts_code: str
    event_type: LifecycleEventType
    effective_from: str
    effective_to: str
    evidence_source: str
    evidence_reference: str

    def applies(self, ts_code: str, trade_date: str) -> bool:
        """Return whether the event applies to a canonical security/date key."""

        return (
            self.canonical_ts_code == ts_code
            and self.effective_from <= trade_date <= self.effective_to
        )


class SecurityLifecycleResolver:
    """Resolve explicit lifecycle intervals without inferring from missing quotes."""

    def __init__(
        self,
        *,
        policy_version: str,
        policy_hash: str,
        events: tuple[SecurityLifecycleEvent, ...],
        policy_path: Path | None = None,
    ) -> None:
        self.policy_version = policy_version
        self.policy_hash = policy_hash
        self.policy_path = policy_path
        self._events = events
        self._by_code: dict[str, tuple[SecurityLifecycleEvent, ...]] = {}
        for code in sorted({event.canonical_ts_code for event in events}):
            self._by_code[code] = tuple(
                event for event in events if event.canonical_ts_code == code
            )
        self._validate()

    @classmethod
    def from_path(cls, path: Path) -> SecurityLifecycleResolver:
        """Load and validate one immutable lifecycle policy artifact."""

        if not path.is_file():
            raise DataValidationError(
                f"SECURITY_LIFECYCLE_POLICY_INVALID: policy file does not exist: {path}"
            )
        raw = path.read_bytes()
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DataValidationError(
                f"SECURITY_LIFECYCLE_POLICY_INVALID: cannot parse {path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: root must be an object")
        if payload.get("schema_version") != 1:
            raise DataValidationError(
                "SECURITY_LIFECYCLE_POLICY_INVALID: unsupported schema_version"
            )
        if payload.get("artifact_name") != "security_lifecycle_events":
            raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: unexpected artifact_name")
        version = str(payload.get("policy_version", "")).strip()
        rows = payload.get("events")
        if not version or not isinstance(rows, list):
            raise DataValidationError(
                "SECURITY_LIFECYCLE_POLICY_INVALID: policy_version and events are required"
            )
        events = tuple(_parse_event(row) for row in rows)
        return cls(
            policy_version=version,
            policy_hash=hashlib.sha256(raw).hexdigest(),
            events=events,
            policy_path=path,
        )

    @classmethod
    def empty(cls) -> SecurityLifecycleResolver:
        """Return an explicit empty policy for isolated callers and fixtures."""

        return cls(
            policy_version="none",
            policy_hash=hashlib.sha256(b"security-lifecycle:none").hexdigest(),
            events=(),
        )

    @property
    def event_count(self) -> int:
        """Return the number of authoritative lifecycle intervals."""

        return len(self._events)

    def events(self) -> tuple[SecurityLifecycleEvent, ...]:
        """Return immutable verified events for lifecycle audit consumers."""

        return self._events

    def provenance(self) -> dict[str, str | int]:
        """Return stable lifecycle provenance for manifests and run identities."""

        return {
            "security_lifecycle_policy_version": self.policy_version,
            "security_lifecycle_policy_hash": self.policy_hash,
            "security_lifecycle_event_count": self.event_count,
        }

    def is_listing_suspended(self, ts_code: str, trade_date: str) -> bool:
        """Return whether authoritative evidence suspends listing on this date."""

        code = str(ts_code).strip().upper()
        date = _date(str(trade_date), "trade_date")
        return any(event.applies(code, date) for event in self._by_code.get(code, ()))

    def apply_suspension(self, frame: DataFrame) -> DataFrame:
        """Overlay lifecycle suspension evidence on canonical market rows."""

        if frame.empty:
            return frame.copy()
        required = {"ts_code", "trade_date", "is_suspended"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DataValidationError(
                f"SECURITY_LIFECYCLE_POLICY_INVALID: frame lacks columns={missing}"
            )
        working = frame.copy()
        codes = working["ts_code"].astype(str).str.strip().str.upper()
        dates = working["trade_date"].astype(str)
        lifecycle = pd.Series(False, index=working.index)
        for event in self._events:
            lifecycle |= (
                codes.eq(event.canonical_ts_code)
                & dates.ge(event.effective_from)
                & dates.le(event.effective_to)
            )
        working["is_suspended"] = working["is_suspended"].fillna(False).astype(bool) | lifecycle
        return working

    def suspension_keys(self, trade_dates: list[str]) -> DataFrame:
        """Expand explicit intervals to canonical security/date keys."""

        if not trade_dates or not self._events:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        dates = tuple(sorted({_date(value, "trade_date") for value in trade_dates}))
        rows = [
            {"ts_code": event.canonical_ts_code, "trade_date": trade_date}
            for event in self._events
            if event.event_type == "LISTING_SUSPENDED"
            for trade_date in dates
            if event.effective_from <= trade_date <= event.effective_to
        ]
        return pd.DataFrame(rows, columns=["ts_code", "trade_date"]).drop_duplicates()

    def _validate(self) -> None:
        seen: set[tuple[str, str, str, str]] = set()
        for event in self._events:
            key = (
                event.canonical_ts_code,
                event.event_type,
                event.effective_from,
                event.effective_to,
            )
            if key in seen:
                raise DataValidationError(
                    f"SECURITY_LIFECYCLE_POLICY_INVALID: duplicate event={key}"
                )
            seen.add(key)
        for code, events in self._by_code.items():
            ordered = sorted(events, key=lambda item: (item.effective_from, item.effective_to))
            for previous, current in zip(ordered, ordered[1:], strict=False):
                if current.effective_from <= previous.effective_to:
                    raise DataValidationError(
                        "SECURITY_LIFECYCLE_COLLISION: overlapping intervals "
                        f"ts_code={code} dates={current.effective_from}..{previous.effective_to}"
                    )


def _parse_event(value: object) -> SecurityLifecycleEvent:
    if not isinstance(value, dict):
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: event must be an object")
    code = str(value.get("canonical_ts_code", "")).strip().upper()
    event_type = str(value.get("event_type", "")).strip().upper()
    effective_from = _date(str(value.get("effective_from", "")), "effective_from")
    effective_to = _date(str(value.get("effective_to", "")), "effective_to")
    source = str(value.get("evidence_source", "")).strip()
    reference = str(value.get("evidence_reference", "")).strip()
    if not code or "." not in code or event_type != "LISTING_SUSPENDED":
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: invalid event identity")
    if effective_from > effective_to:
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: reversed effective interval")
    if not source or not reference:
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: evidence is required")
    return SecurityLifecycleEvent(
        canonical_ts_code=code,
        event_type="LISTING_SUSPENDED",
        effective_from=effective_from,
        effective_to=effective_to,
        evidence_source=source,
        evidence_reference=reference,
    )


def _date(value: str, field: str) -> str:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"SECURITY_LIFECYCLE_POLICY_INVALID: {field} must be YYYYMMDD")
    return value
