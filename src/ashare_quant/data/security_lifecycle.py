"""Authoritative point-in-time security lifecycle events."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_lifecycle_compiler import LifecycleIntervalCompiler
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]
type LifecycleEventType = Literal["ORDINARY_SUSPENSION", "LISTING_SUSPENDED"]

if TYPE_CHECKING:
    from ashare_quant.data.security_identity_transition import SecurityIdentityTransitionResolver


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
        if payload.get("schema_version") not in {1, 2, 3}:
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
        return any(
            event.event_type == "LISTING_SUSPENDED" and event.applies(code, date)
            for event in self._by_code.get(code, ())
        )

    def is_ordinary_suspended(self, ts_code: str, trade_date: str) -> bool:
        """Return whether verified full-day ordinary evidence applies on this date."""

        code = str(ts_code).strip().upper()
        date = _date(str(trade_date), "trade_date")
        return any(
            event.event_type == "ORDINARY_SUSPENSION" and event.applies(code, date)
            for event in self._by_code.get(code, ())
        )

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

    def suspension_keys(
        self, trade_dates: list[str], *, event_type: LifecycleEventType | None = None
    ) -> DataFrame:
        """Expand typed verified intervals to canonical security/date keys."""

        if not trade_dates or not self._events:
            return pd.DataFrame(columns=["ts_code", "trade_date"])
        dates = tuple(sorted({_date(value, "trade_date") for value in trade_dates}))
        rows = [
            {"ts_code": event.canonical_ts_code, "trade_date": trade_date}
            for event in self._events
            if event_type is None or event.event_type == event_type
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
    if (
        not code
        or "." not in code
        or event_type
        not in {
            "ORDINARY_SUSPENSION",
            "LISTING_SUSPENDED",
        }
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: invalid event identity")
    if effective_from > effective_to:
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: reversed effective interval")
    if not source or not reference:
        raise DataValidationError("SECURITY_LIFECYCLE_POLICY_INVALID: evidence is required")
    return SecurityLifecycleEvent(
        canonical_ts_code=code,
        event_type=event_type,  # type: ignore[arg-type]
        effective_from=effective_from,
        effective_to=effective_to,
        evidence_source=source,
        evidence_reference=reference,
    )


def _date(value: str, field: str) -> str:
    from datetime import datetime

    try:
        valid = datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        valid = False
    if not valid:
        raise DataValidationError(f"SECURITY_LIFECYCLE_POLICY_INVALID: {field} must be YYYYMMDD")
    return value


def publish_typed_lifecycle_catalog(
    *,
    official_index: Path,
    base_catalog: SecurityLifecycleResolver,
    reports_root: Path,
    catalog_version: str,
    open_trade_dates: tuple[str, ...],
    research_start: str,
    research_end: str,
    identity_transitions: SecurityIdentityTransitionResolver | None = None,
) -> Path:
    """Compile VERIFIED ordinary/listing evidence into one partial runtime catalog."""

    from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
    from ashare_quant.data.security_lifecycle_official_index import (
        validate_official_lifecycle_index,
    )

    index_manifest = validate_official_lifecycle_index(official_index)
    events = pd.read_parquet(official_index / "official_events.parquet")
    base_rows: list[JsonObject] = [
        {
            "canonical_ts_code": event.canonical_ts_code,
            "event_type": event.event_type,
            "effective_from": event.effective_from,
            "effective_to": event.effective_to,
            "evidence_source": event.evidence_source,
            "evidence_reference": event.evidence_reference,
        }
        for event in base_catalog.events()
    ]
    compilation = LifecycleIntervalCompiler(
        open_trade_dates=open_trade_dates,
        research_start=research_start,
        research_end=research_end,
        identity_transitions=identity_transitions,
    ).compile(events, base_events=tuple(base_rows), require_closed=False)
    rows_by_identity = {
        (
            str(row["canonical_ts_code"]),
            str(row["event_type"]),
            str(row["effective_from"]),
            str(row["effective_to"]),
        ): row
        for row in base_rows
    }
    duplicate_count = 0
    for compiled in compilation.events:
        identity = (
            str(compiled["canonical_ts_code"]),
            str(compiled["event_type"]),
            str(compiled["effective_from"]),
            str(compiled["effective_to"]),
        )
        # The base catalog remains immutable provenance when an official index
        # independently verifies the exact same lifecycle interval.
        if identity in rows_by_identity:
            duplicate_count += 1
        rows_by_identity.setdefault(identity, compiled)
    ordered = sorted(
        rows_by_identity.values(),
        key=lambda row: (
            str(row["canonical_ts_code"]),
            str(row["effective_from"]),
            str(row["event_type"]),
        ),
    )
    payload: JsonObject = {
        "schema_version": 3,
        "artifact_name": "security_lifecycle_events",
        "policy_version": catalog_version,
        "completeness": "partial",
        "events": ordered,
    }
    resolver = SecurityLifecycleResolver(
        policy_version=catalog_version,
        policy_hash="0" * 64,
        events=tuple(_parse_event(row) for row in ordered),
    )
    compiler_report = dict(compilation.report)
    compiler_report["counts"] = {
        **cast(JsonObject, compiler_report["counts"]),
        "base_runtime_intervals": len(base_rows),
        "deduplicated_intervals": duplicate_count
        + int(cast(JsonObject, compiler_report["counts"])["base_events_covering_official_starts"]),
        "published_runtime_intervals": resolver.event_count,
    }
    logical = {
        "catalog_version": catalog_version,
        "base_catalog_hash": base_catalog.policy_hash,
        "official_index_id": index_manifest["index_id"],
        "official_index_manifest_hash": file_sha256(official_index / "manifest.json"),
        "events_hash": canonical_payload_hash(ordered),
        "event_count": resolver.event_count,
        "completeness": "partial",
        "compiler_contract_version": compiler_report["compiler_contract_version"],
        "compiler_report_hash": canonical_payload_hash(compiler_report),
        "research_start": research_start,
        "research_end": research_end,
        "trade_calendar_hash": compiler_report["trade_calendar_hash"],
        "identity_transition_version": (
            "none" if identity_transitions is None else identity_transitions.artifact_version
        ),
        "identity_transition_hash": (
            "none" if identity_transitions is None else identity_transitions.artifact_hash
        ),
    }
    catalog_id = f"security_lifecycle_typed_catalog_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_lifecycle_typed_catalog" / catalog_id
    if output.exists():
        validate_typed_lifecycle_catalog(output, official_index=official_index)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{catalog_id}.staging-"))
    try:
        atomic_write_json(staging / "events.json", payload)
        atomic_write_json(staging / "compiler_report.json", compiler_report)
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": 2,
                "artifact_name": "security_lifecycle_typed_catalog",
                "catalog_id": catalog_id,
                "logical_identity": logical,
                "artifact_hashes": {
                    "compiler_report.json": file_sha256(staging / "compiler_report.json"),
                    "events.json": file_sha256(staging / "events.json"),
                },
            },
        )
        _validate_typed_catalog_contents(
            staging, expected_id=catalog_id, official_index=official_index
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def validate_typed_lifecycle_catalog(path: Path, *, official_index: Path) -> JsonObject:
    """Validate catalog bytes, logical identity and its source official index."""

    return _validate_typed_catalog_contents(
        path, expected_id=path.name, official_index=official_index
    )


def _validate_typed_catalog_contents(
    path: Path, *, expected_id: str, official_index: Path
) -> JsonObject:
    from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
    from ashare_quant.data.security_lifecycle_official_index import (
        validate_official_lifecycle_index,
    )

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        payload = json.loads((path / "events.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or not isinstance(payload, dict)
        or manifest.get("catalog_id") != expected_id
        or payload.get("completeness") != "partial"
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_INVALID")
    schema_version = manifest.get("schema_version")
    payload_schema = payload.get("schema_version")
    if schema_version == 1 and payload_schema == 2:
        expected_hashes = {"events.json": file_sha256(path / "events.json")}
        compiler_report: JsonObject | None = None
    elif schema_version == 2 and payload_schema == 3:
        try:
            parsed_report = json.loads((path / "compiler_report.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_INVALID") from error
        if not isinstance(parsed_report, dict):
            raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_INVALID")
        compiler_report = cast(JsonObject, parsed_report)
        expected_hashes = {
            "compiler_report.json": file_sha256(path / "compiler_report.json"),
            "events.json": file_sha256(path / "events.json"),
        }
    else:
        raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_INVALID")
    if manifest.get("artifact_hashes") != expected_hashes:
        raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_INVALID")
    source = validate_official_lifecycle_index(official_index)
    logical = cast(JsonObject, manifest.get("logical_identity", {}))
    if (
        logical.get("official_index_id") != source.get("index_id")
        or logical.get("official_index_manifest_hash")
        != file_sha256(official_index / "manifest.json")
        or logical.get("events_hash") != canonical_payload_hash(payload.get("events"))
        or f"security_lifecycle_typed_catalog_{canonical_payload_hash(logical)[:24]}" != expected_id
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_IDENTITY_MISMATCH")
    if compiler_report is not None and (
        logical.get("compiler_report_hash") != canonical_payload_hash(compiler_report)
        or logical.get("trade_calendar_hash") != compiler_report.get("trade_calendar_hash")
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_TYPED_CATALOG_IDENTITY_MISMATCH")
    SecurityLifecycleResolver(
        policy_version=str(payload.get("policy_version", "")),
        policy_hash=file_sha256(path / "events.json"),
        events=tuple(_parse_event(row) for row in cast(list[object], payload.get("events", []))),
    )
    return cast(JsonObject, manifest)
