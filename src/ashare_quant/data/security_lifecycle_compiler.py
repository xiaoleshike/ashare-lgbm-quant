"""Compile verified official lifecycle events into bounded runtime intervals."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

if TYPE_CHECKING:
    from ashare_quant.data.security_identity_transition import (
        SecurityIdentityTransitionResolver,
    )


def _canonical_payload_hash(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LifecycleIntervalCompilation:
    """Compiled official intervals plus fail-closed diagnostics."""

    events: tuple[JsonObject, ...]
    provenance: tuple[JsonObject, ...]
    report: JsonObject


class LifecycleIntervalCompiler:
    """Pair explicit listing-suspension starts with authoritative close events."""

    def __init__(
        self,
        *,
        open_trade_dates: tuple[str, ...],
        research_start: str,
        research_end: str,
        identity_transitions: SecurityIdentityTransitionResolver | None = None,
    ) -> None:
        self.trade_dates = tuple(sorted(set(open_trade_dates)))
        self.research_start = research_start
        self.research_end = research_end
        self.identity_transitions = identity_transitions
        if not self.trade_dates or research_start > research_end:
            raise DataValidationError("LIFECYCLE_INTERVAL_COMPILER_CALENDAR_INVALID")

    @property
    def trade_calendar_hash(self) -> str:
        """Return the content identity of the governed open-session calendar."""

        return _canonical_payload_hash(list(self.trade_dates))

    def compile(
        self,
        official_events: DataFrame,
        *,
        base_events: tuple[JsonObject, ...],
        require_closed: bool = False,
    ) -> LifecycleIntervalCompilation:
        """Compile supported verified events without inferring missing boundaries."""

        required = {
            "canonical_ts_code",
            "event_type",
            "effective_start",
            "effective_end",
            "row_identity",
            "package_id",
            "package_hash",
            "status",
        }
        if not required.issubset(official_events.columns):
            raise DataValidationError("LIFECYCLE_INTERVAL_COMPILER_EVENT_SCHEMA_INVALID")
        verified = official_events[official_events["status"].astype(str).eq("VERIFIED")].copy()
        verified = verified.sort_values(
            ["canonical_ts_code", "effective_start", "event_type", "row_identity"]
        )
        base_listing = {
            (
                str(row["canonical_ts_code"]),
                str(row["effective_from"]),
                str(row["effective_to"]),
            )
            for row in base_events
            if str(row["event_type"]) == "LISTING_SUSPENDED"
        }
        base_by_start = {(code, start): end for code, start, end in base_listing}
        compiled: list[JsonObject] = []
        provenance: list[JsonObject] = []
        unclosed: list[JsonObject] = []
        conflicts: list[JsonObject] = []
        base_covered = 0
        active: dict[str, JsonObject] = {}
        base_covered_active: dict[str, JsonObject] = {}
        terminal_dates: dict[str, str] = {}
        resumption_closed = 0
        terminal_closed = 0
        carry_in = 0

        for row in verified.itertuples(index=False):
            code = str(row.canonical_ts_code)
            event_type = str(row.event_type)
            date = str(row.effective_start)
            event = {
                "canonical_ts_code": code,
                "event_type": event_type,
                "effective_start": date,
                "effective_end": str(row.effective_end),
                "row_identity": str(row.row_identity),
                "package_id": str(row.package_id),
                "package_hash": str(row.package_hash),
            }
            if event_type == "ORDINARY_FULL_DAY_SUSPENSION":
                clipped = self._clip(str(row.effective_start), str(row.effective_end))
                if clipped is not None:
                    start, end = clipped
                    compiled.append(self._runtime_event(event, "ORDINARY_SUSPENSION", start, end))
                    provenance.append(
                        self._ordinary_provenance(event, compiled_start=start, compiled_end=end)
                    )
                continue
            if event_type == "FORMAL_LISTING_SUSPENSION_START":
                base_end = base_by_start.get((code, date))
                if base_end is not None and base_end == str(row.effective_end):
                    base_covered += 1
                    base_covered_active[code] = event
                    continue
                if code in terminal_dates and date >= terminal_dates[code]:
                    conflicts.append(self._conflict("START_AFTER_TERMINAL", event))
                    continue
                if code in active:
                    conflicts.append(self._conflict("DOUBLE_START", event, active[code]))
                    continue
                active[code] = event
                continue
            if event_type not in {"FORMAL_LISTING_RESUMPTION", "TERMINAL_DELISTING"}:
                continue
            if event_type == "TERMINAL_DELISTING":
                previous_terminal = terminal_dates.get(code)
                if previous_terminal is not None and previous_terminal != date:
                    conflicts.append(self._conflict("MULTIPLE_TERMINAL_EVENTS", event))
                    continue
                terminal_dates[code] = date
            opening = active.pop(code, None)
            if opening is None:
                if code in base_covered_active:
                    base_covered_active.pop(code)
                    continue
                if event_type == "FORMAL_LISTING_RESUMPTION":
                    conflicts.append(self._conflict("RESUME_WITHOUT_START", event))
                continue
            if date <= str(opening["effective_start"]):
                conflicts.append(self._conflict("CLOSE_NOT_AFTER_START", event, opening))
                continue
            transition_conflict = self._transition_conflict(
                code, str(opening["effective_start"]), date
            )
            if transition_conflict is not None:
                conflicts.append(transition_conflict)
                continue
            end = self._previous_open_session(date)
            clipped = self._clip(str(opening["effective_start"]), end)
            if clipped is None:
                continue
            compiled_start, compiled_end = clipped
            compiled.append(
                self._runtime_event(opening, "LISTING_SUSPENDED", compiled_start, compiled_end)
            )
            provenance.append(
                {
                    "canonical_ts_code": code,
                    "event_type": "LISTING_SUSPENDED",
                    "opening_event_row_identity": opening["row_identity"],
                    "opening_package_id": opening["package_id"],
                    "opening_package_hash": opening["package_hash"],
                    "closing_event_type": event_type,
                    "closing_event_row_identity": event["row_identity"],
                    "closing_package_id": event["package_id"],
                    "closing_package_hash": event["package_hash"],
                    "authoritative_start": opening["effective_start"],
                    "authoritative_close_date": date,
                    "compiled_effective_from": compiled_start,
                    "compiled_effective_to": compiled_end,
                    "trade_calendar_hash": self.trade_calendar_hash,
                }
            )
            if str(opening["effective_start"]) < self.research_start:
                carry_in += 1
            if event_type == "FORMAL_LISTING_RESUMPTION":
                resumption_closed += 1
            else:
                terminal_closed += 1

        for code, opening in sorted(active.items()):
            unclosed.append(
                {
                    "error_code": "LISTING_SUSPENSION_INTERVAL_UNCLOSED",
                    "canonical_ts_code": code,
                    "opening_event_row_identity": opening["row_identity"],
                    "authoritative_start": opening["effective_start"],
                }
            )
        # Exact base intervals are already executable runtime evidence. Their
        # official event-stream representation is retained as provenance, not
        # reclassified as an unclosed newly compiled interval.
        if require_closed and (unclosed or conflicts):
            first = (conflicts + unclosed)[0]
            raise DataValidationError(str(first["error_code"]))

        report: JsonObject = {
            "compiler_contract_version": "lifecycle_interval_compiler_v1",
            "research_start": self.research_start,
            "research_end": self.research_end,
            "trade_calendar_hash": self.trade_calendar_hash,
            "counts": {
                "verified_events_consumed": int(len(verified)),
                "ordinary_intervals_compiled": sum(
                    row["event_type"] == "ORDINARY_SUSPENSION" for row in compiled
                ),
                "listing_intervals_compiled": sum(
                    row["event_type"] == "LISTING_SUSPENDED" for row in compiled
                ),
                "resumption_closed": resumption_closed,
                "terminal_closed": terminal_closed,
                "carry_in": carry_in,
                "unclosed": len(unclosed),
                "conflicts": len(conflicts),
                "base_events_covering_official_starts": base_covered,
            },
            "unclosed_intervals": unclosed,
            "conflicts": conflicts,
            "compiled_provenance": provenance,
        }
        return LifecycleIntervalCompilation(tuple(compiled), tuple(provenance), report)

    def _previous_open_session(self, date: str) -> str:
        position = bisect_left(self.trade_dates, date)
        if position == 0:
            raise DataValidationError("LIFECYCLE_INTERVAL_COMPILER_CLOSE_BEFORE_CALENDAR")
        return self.trade_dates[position - 1]

    def _clip(self, start: str, end: str) -> tuple[str, str] | None:
        clipped_start = max(start, self.research_start)
        clipped_end = min(end, self.research_end)
        return None if clipped_start > clipped_end else (clipped_start, clipped_end)

    def _transition_conflict(self, code: str, start: str, close: str) -> JsonObject | None:
        if self.identity_transitions is None:
            return None
        for transition in self.identity_transitions.transition_records():
            if transition.predecessor_ts_code == code and start < transition.effective_date < close:
                return {
                    "error_code": "LIFECYCLE_INTERVAL_IDENTITY_TRANSITION_AMBIGUOUS",
                    "canonical_ts_code": code,
                    "effective_date": transition.effective_date,
                    "evidence_package_id": transition.evidence_package_id,
                }
        return None

    @staticmethod
    def _runtime_event(source: JsonObject, event_type: str, start: str, end: str) -> JsonObject:
        return {
            "canonical_ts_code": source["canonical_ts_code"],
            "event_type": event_type,
            "effective_from": start,
            "effective_to": end,
            "evidence_source": source["package_id"],
            "evidence_reference": source["row_identity"],
        }

    def _ordinary_provenance(
        self, source: JsonObject, *, compiled_start: str, compiled_end: str
    ) -> JsonObject:
        return {
            "canonical_ts_code": source["canonical_ts_code"],
            "event_type": "ORDINARY_SUSPENSION",
            "opening_event_row_identity": source["row_identity"],
            "opening_package_id": source["package_id"],
            "opening_package_hash": source["package_hash"],
            "closing_event_type": "SAME_EVENT_INTERVAL",
            "closing_event_row_identity": source["row_identity"],
            "closing_package_id": source["package_id"],
            "closing_package_hash": source["package_hash"],
            "authoritative_start": source["effective_start"],
            "authoritative_close_date": source["effective_end"],
            "compiled_effective_from": compiled_start,
            "compiled_effective_to": compiled_end,
            "trade_calendar_hash": self.trade_calendar_hash,
        }

    @staticmethod
    def _conflict(name: str, event: JsonObject, opening: JsonObject | None = None) -> JsonObject:
        result: JsonObject = {
            "error_code": f"LIFECYCLE_INTERVAL_COMPILER_{name}",
            "canonical_ts_code": event["canonical_ts_code"],
            "event_row_identity": event["row_identity"],
            "effective_date": event["effective_start"],
        }
        if opening is not None:
            result["opening_event_row_identity"] = opening["row_identity"]
        return result
