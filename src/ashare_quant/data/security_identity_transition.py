"""Versioned, evidence-backed security-code transition identities.

Ticker aliases describe alternate source keys for the same observation.  A code
transition instead changes the effective exchange code at an authoritative
corporate-action boundary, so it is kept in a separate artifact and graph.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.utils.manifest import atomic_write_json

type JsonObject = dict[str, Any]

TRANSITION_SCHEMA_VERSION = 2
LEGACY_TRANSITION_SCHEMA_VERSIONS = frozenset({1})
TRANSITION_ARTIFACT_NAME = "security_identity_transitions"
TRANSITION_EVIDENCE_SCHEMA_VERSION = 1
TRANSITION_EVIDENCE_ARTIFACT_NAME = "security_identity_transition_evidence"
TRANSITION_TYPES = frozenset(
    {
        "CODE_CHANGE_CONTINUITY",
        "RESTRUCTURING_CODE_CHANGE",
        "MERGER_SUCCESSOR",
        "SHARE_CONVERSION",
    }
)
CONTINUITY_TYPES = frozenset({"SAME_LISTED_ENTITY", "SUCCESSOR_ENTITY"})
OFFICIAL_HOST_SUFFIXES = ("sse.com.cn", "szse.cn", "bse.cn")
SOURCE_RECONCILIATION_SCHEMA_VERSION = 1
SOURCE_RECONCILIATION_ARTIFACT_NAME = "security_source_reconciliation"
SUPPORTED_SOURCE_REPRESENTATION_DATASETS = frozenset({"suspend_d"})
_SOURCE_RULE_FIELDS = (
    "dataset_name",
    "source_ts_code",
    "effective_ts_code",
    "effective_from",
    "effective_to",
    "resolution_rule_id",
)
_SOURCE_RECONCILIATION_REQUIRED_COLUMNS = frozenset(
    {
        "parent_interval_id",
        "trade_date",
        "source_dataset",
        "source_ts_code",
        "resolved_identity",
        "effective_trade_code",
        "resolution_rule_id",
        "row_present",
        "valid_quote",
        "suspend_type",
        "raw_suspend_timing",
        "source_partition",
        "source_content_hash",
        "identity_evidence_hash",
    }
)


@dataclass(frozen=True, slots=True)
class SecurityIdentityTransition:
    """One authoritative effective-dated predecessor/successor relationship."""

    predecessor_ts_code: str
    successor_ts_code: str
    predecessor_name: str
    successor_name: str
    transition_type: str
    effective_date: str
    continuity_type: str
    share_conversion_ratio: float | None
    evidence_package_id: str
    evidence_package_hash: str

    @property
    def execution_supported(self) -> bool:
        """Return whether this artifact alone proves executable share accounting.

        Even an explicit 1:1 ratio does not yet imply that the backtest has a
        reviewed position-transfer implementation.  The engine therefore blocks
        all crossing positions until that separate accounting feature exists.
        """

        return False


@dataclass(frozen=True, slots=True)
class SecuritySourceRepresentationRule:
    """Map one provider's historical dataset key without changing exchange identity."""

    dataset_name: str
    source_ts_code: str
    effective_ts_code: str
    effective_from: str
    effective_to: str
    resolution_rule_id: str
    evidence_package_id: str
    evidence_package_hash: str


class SecurityIdentityTransitionResolver:
    """Validate and traverse one immutable security-code transition graph."""

    def __init__(
        self,
        *,
        artifact_version: str,
        artifact_hash: str,
        transitions: tuple[SecurityIdentityTransition, ...],
        source_representation_rules: tuple[SecuritySourceRepresentationRule, ...] = (),
        artifact_path: Path | None = None,
    ) -> None:
        self.artifact_version = artifact_version
        self.artifact_hash = artifact_hash
        self.artifact_path = artifact_path
        self._transitions = transitions
        self._source_representation_rules = source_representation_rules
        self._by_predecessor = {item.predecessor_ts_code: item for item in transitions}
        self._validate()

    @classmethod
    def empty(cls) -> SecurityIdentityTransitionResolver:
        """Return an explicit no-transition resolver for legacy callers."""

        return cls(
            artifact_version="none",
            artifact_hash=canonical_payload_hash({"security_identity_transitions": []}),
            transitions=(),
            source_representation_rules=(),
        )

    @classmethod
    def from_path(cls, path: Path) -> SecurityIdentityTransitionResolver:
        """Load and validate a completed transition artifact directory."""

        manifest = validate_security_identity_transition_artifact(path)
        payload = _read_json(path / "transitions.json", "SECURITY_IDENTITY_TRANSITION_INVALID")
        rows = payload.get("transitions")
        if not isinstance(rows, list):
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
        transitions = tuple(_parse_transition(cast(JsonObject, row)) for row in rows)
        rule_rows = payload.get("source_representation_rules", [])
        if not isinstance(rule_rows, list):
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
        rules = tuple(_parse_source_representation_rule(cast(JsonObject, row)) for row in rule_rows)
        resolver = cls(
            artifact_version=str(payload.get("transition_version", "")),
            artifact_hash=file_sha256(path / "manifest.json"),
            transitions=transitions,
            source_representation_rules=rules,
            artifact_path=path,
        )
        if manifest.get("transition_version") != resolver.artifact_version:
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_IDENTITY_MISMATCH")
        return resolver

    @property
    def transition_count(self) -> int:
        return len(self._transitions)

    def transition_records(self) -> tuple[SecurityIdentityTransition, ...]:
        """Return immutable transition records for vectorized consumers."""

        return self._transitions

    @property
    def source_representation_rule_count(self) -> int:
        return len(self._source_representation_rules)

    def source_representation_records(self) -> tuple[SecuritySourceRepresentationRule, ...]:
        """Return immutable, dataset-scoped provider representation rules."""

        return self._source_representation_rules

    def normalize_source_frame(self, frame: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
        """Apply verified provider-key representation without changing raw bytes.

        The original provider key is retained in ``source_ts_code``. If both source
        keys occur for one logical observation, their semantic rows must agree.
        """

        rules = tuple(
            item for item in self._source_representation_rules if item.dataset_name == dataset_name
        )
        if not rules or frame.empty:
            return frame.copy()
        if not {"ts_code", "trade_date"}.issubset(frame.columns):
            raise DataValidationError("SECURITY_SOURCE_REPRESENTATION_INPUT_INVALID")
        working = frame.copy()
        working["source_ts_code"] = working["ts_code"].astype(str).str.strip().str.upper()
        working["ts_code"] = working["source_ts_code"]
        dates = working["trade_date"].astype(str)
        applied = pd.Series(False, index=working.index)
        for rule in rules:
            selected = (
                working["source_ts_code"].eq(rule.source_ts_code)
                & dates.ge(rule.effective_from)
                & dates.le(rule.effective_to)
            )
            working.loc[selected, "ts_code"] = rule.effective_ts_code
            applied |= selected
        if not applied.any():
            return working
        affected_keys = pd.MultiIndex.from_frame(working.loc[applied, ["ts_code", "trade_date"]])
        row_keys = pd.MultiIndex.from_frame(working[["ts_code", "trade_date"]])
        affected = row_keys.isin(affected_keys)
        normalized = _deduplicate_source_representations(
            working.loc[affected].copy(), dataset_name=dataset_name
        )
        return pd.concat([working.loc[~affected], normalized], ignore_index=True)

    def transition_for(
        self, predecessor_ts_code: str, as_of_date: str
    ) -> SecurityIdentityTransition | None:
        """Return an effective transition for the predecessor on a date."""

        item = self._by_predecessor.get(_code(predecessor_ts_code))
        return item if item is not None and as_of_date >= item.effective_date else None

    def effective_code(self, ts_code: str, *, as_of_date: str) -> str:
        """Resolve a valid transition chain as of a date."""

        current = _code(ts_code)
        visited: set[str] = set()
        while True:
            if current in visited:
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONFLICT: cycle")
            visited.add(current)
            transition = self.transition_for(current, as_of_date)
            if transition is None:
                return current
            current = transition.successor_ts_code

    def predecessor_expected(self, ts_code: str, *, trade_date: str) -> bool:
        """Return whether quotes are still expected under a predecessor code."""

        item = self._by_predecessor.get(_code(ts_code))
        return item is None or trade_date < item.effective_date

    def execution_code_closure(self, ts_codes: set[str], *, end_date: str) -> tuple[str, ...]:
        """Include every effective successor needed at an execution-price boundary."""

        selected = {_code(code) for code in ts_codes}
        pending = list(selected)
        while pending:
            current = pending.pop()
            item = self._by_predecessor.get(current)
            if item is None or item.effective_date > end_date:
                continue
            if item.successor_ts_code not in selected:
                selected.add(item.successor_ts_code)
                pending.append(item.successor_ts_code)
        return tuple(sorted(selected))

    def provenance(self) -> JsonObject:
        """Return stable transition provenance for scan/execution identities."""

        return {
            "security_identity_transition_version": self.artifact_version,
            "security_identity_transition_hash": self.artifact_hash,
            "security_identity_transition_count": self.transition_count,
            "security_source_representation_rule_count": self.source_representation_rule_count,
        }

    def validate_alias_coexistence(self, aliases: SecurityIdentityResolver) -> None:
        """Reject a transition that conflicts with the BSE/source-alias graph."""

        for item in self._transitions:
            predecessor_alias = aliases.canonicalize(item.predecessor_ts_code)
            successor_alias = aliases.canonicalize(item.successor_ts_code)
            if (
                predecessor_alias != item.predecessor_ts_code
                or successor_alias != item.successor_ts_code
            ):
                raise DataValidationError(
                    "SECURITY_IDENTITY_TRANSITION_CONFLICT: transition overlaps alias mapping"
                )
        for rule in self._source_representation_rules:
            if (
                aliases.canonicalize(rule.source_ts_code) != rule.source_ts_code
                or aliases.canonicalize(rule.effective_ts_code) != rule.effective_ts_code
            ):
                raise DataValidationError(
                    "SECURITY_SOURCE_REPRESENTATION_CONFLICT: rule overlaps alias mapping"
                )

    def _validate(self) -> None:
        if not self.artifact_version or len(self._by_predecessor) != len(self._transitions):
            raise DataValidationError(
                "SECURITY_IDENTITY_TRANSITION_CONFLICT: duplicate predecessor"
            )
        successors: dict[str, SecurityIdentityTransition] = {}
        for item in self._transitions:
            if (
                item.predecessor_ts_code == item.successor_ts_code
                or item.transition_type not in TRANSITION_TYPES
                or item.continuity_type not in CONTINUITY_TYPES
                or not _is_date(item.effective_date)
                or not item.evidence_package_id
                or not _is_sha256(item.evidence_package_hash)
                or (
                    item.share_conversion_ratio is not None
                    and (
                        isinstance(item.share_conversion_ratio, bool)
                        or not math.isfinite(item.share_conversion_ratio)
                        or item.share_conversion_ratio <= 0
                    )
                )
            ):
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
            if not _valid_type_continuity(item.transition_type, item.continuity_type):
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID: semantics")
            previous = successors.get(item.successor_ts_code)
            if previous is not None and previous.predecessor_ts_code != item.predecessor_ts_code:
                raise DataValidationError(
                    "SECURITY_IDENTITY_TRANSITION_UNSUPPORTED_TOPOLOGY: multiple predecessors"
                )
            successors[item.successor_ts_code] = item
        for predecessor in self._by_predecessor:
            current = predecessor
            seen: set[str] = set()
            while current in self._by_predecessor:
                if current in seen:
                    raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONFLICT: cycle")
                seen.add(current)
                transition = self._by_predecessor[current]
                successor = transition.successor_ts_code
                if successor in seen:
                    raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONFLICT: cycle")
                next_transition = self._by_predecessor.get(successor)
                if (
                    next_transition is not None
                    and next_transition.effective_date < transition.effective_date
                ):
                    raise DataValidationError(
                        "SECURITY_IDENTITY_TRANSITION_CONFLICT: transition chain is out of order"
                    )
                current = successor
        seen_rules: dict[tuple[str, str], list[SecuritySourceRepresentationRule]] = {}
        for rule in self._source_representation_rules:
            if (
                rule.dataset_name not in SUPPORTED_SOURCE_REPRESENTATION_DATASETS
                or rule.source_ts_code == rule.effective_ts_code
                or not _is_date(rule.effective_from)
                or not _is_date(rule.effective_to)
                or rule.effective_from > rule.effective_to
                or not rule.resolution_rule_id
                or not rule.evidence_package_id
                or not _is_sha256(rule.evidence_package_hash)
            ):
                raise DataValidationError("SECURITY_SOURCE_REPRESENTATION_RULE_INVALID")
            identity_transition = self._by_predecessor.get(rule.effective_ts_code)
            if (
                identity_transition is None
                or identity_transition.successor_ts_code != rule.source_ts_code
                or identity_transition.continuity_type != "SAME_LISTED_ENTITY"
                or rule.effective_to >= identity_transition.effective_date
            ):
                raise DataValidationError("SECURITY_SOURCE_REPRESENTATION_RULE_IDENTITY_UNVERIFIED")
            key = (rule.dataset_name, rule.source_ts_code)
            previous_rules = seen_rules.setdefault(key, [])
            if any(
                rule.effective_from <= previous_rule.effective_to
                and previous_rule.effective_from <= rule.effective_to
                for previous_rule in previous_rules
            ):
                raise DataValidationError("SECURITY_SOURCE_REPRESENTATION_CONFLICT")
            previous_rules.append(rule)


def load_identity_transition_contract(
    *, mode: str, artifact_path: Path | None
) -> SecurityIdentityTransitionResolver:
    """Resolve an explicit configured artifact or explicit no-transition contract."""

    if mode == "artifact" and artifact_path is not None:
        return SecurityIdentityTransitionResolver.from_path(artifact_path)
    if mode == "none" and artifact_path is None:
        return SecurityIdentityTransitionResolver.empty()
    raise DataValidationError("SECURITY_IDENTITY_TRANSITION_CONTRACT_INVALID")


def publish_transition_evidence_package(
    *,
    evidence: JsonObject,
    document: Path,
    reports_root: Path,
) -> Path:
    """Freeze one reviewed official transition document and structured fact."""

    normalized = _normalize_evidence(evidence, document)
    logical = {key: value for key, value in normalized.items() if key != "retrieved_at"}
    package_id = f"security_identity_transition_evidence_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_identity_transition_evidence" / package_id
    if output.exists():
        validate_transition_evidence_package(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{package_id}.staging-"))
    try:
        documents = staging / "documents"
        documents.mkdir()
        document_name = f"official_document{document.suffix.lower()}"
        shutil.copyfile(document, documents / document_name)
        payload = {
            **normalized,
            "evidence_package_id": package_id,
            "document_path": f"documents/{document_name}",
        }
        atomic_write_json(staging / "evidence.json", payload)
        hashes = {
            "evidence.json": file_sha256(staging / "evidence.json"),
            f"documents/{document_name}": file_sha256(documents / document_name),
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": TRANSITION_EVIDENCE_SCHEMA_VERSION,
                "artifact_name": TRANSITION_EVIDENCE_ARTIFACT_NAME,
                "evidence_package_id": package_id,
                "logical_identity": logical,
                "artifact_hashes": hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_transition_evidence_package(output)
    return output


def validate_transition_evidence_package(path: Path) -> tuple[JsonObject, str]:
    """Validate all evidence metadata and frozen document bytes."""

    manifest = _read_json(path / "manifest.json", "SECURITY_IDENTITY_TRANSITION_EVIDENCE_INVALID")
    evidence = _read_json(path / "evidence.json", "SECURITY_IDENTITY_TRANSITION_EVIDENCE_INVALID")
    if (
        manifest.get("schema_version") != TRANSITION_EVIDENCE_SCHEMA_VERSION
        or manifest.get("artifact_name") != TRANSITION_EVIDENCE_ARTIFACT_NAME
        or manifest.get("evidence_package_id") != path.name
        or evidence.get("evidence_package_id") != path.name
    ):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != {
        "evidence.json",
        str(evidence.get("document_path", "")),
    }:
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_INVALID")
    for relative, expected in hashes.items():
        child = (path / str(relative)).resolve()
        if (
            path.resolve() not in child.parents
            or not child.is_file()
            or file_sha256(child) != expected
        ):
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_HASH_MISMATCH")
    normalized = _normalize_evidence(evidence, path / str(evidence["document_path"]))
    logical = {key: value for key, value in normalized.items() if key != "retrieved_at"}
    if logical != manifest.get("logical_identity"):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_IDENTITY_MISMATCH")
    return evidence, file_sha256(path / "manifest.json")


def publish_source_reconciliation_artifact(
    *,
    source_reconciliation: pd.DataFrame,
    session_resolution: pd.DataFrame,
    source_representation_rules: tuple[JsonObject, ...],
    input_identity: JsonObject,
    reports_root: Path,
) -> Path:
    """Freeze dataset-scoped source-key proof without changing canonical raw bytes."""

    if not _SOURCE_RECONCILIATION_REQUIRED_COLUMNS.issubset(source_reconciliation.columns):
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
    normalized_rules = [
        {key: rule[key] for key in _SOURCE_RULE_FIELDS} for rule in source_representation_rules
    ]
    for rule in normalized_rules:
        _parse_source_representation_rule(
            {
                **rule,
                "evidence_package_id": "pending",
                "evidence_package_hash": "0" * 64,
            }
        )
    if not isinstance(input_identity, dict) or not input_identity:
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
    logical = {
        "input_identity": input_identity,
        "source_reconciliation_hash": _frame_logical_hash(source_reconciliation),
        "session_resolution_hash": _frame_logical_hash(session_resolution),
        "source_representation_rules": sorted(
            normalized_rules,
            key=lambda item: (
                str(item["dataset_name"]),
                str(item["source_ts_code"]),
                str(item["effective_from"]),
            ),
        ),
    }
    artifact_id = f"{SOURCE_RECONCILIATION_ARTIFACT_NAME}_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / SOURCE_RECONCILIATION_ARTIFACT_NAME / artifact_id
    if output.exists():
        validate_source_reconciliation_artifact(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{artifact_id}.staging-"))
    try:
        source_reconciliation.to_parquet(staging / "source_reconciliation.parquet", index=False)
        session_resolution.to_parquet(staging / "session_resolution.parquet", index=False)
        summary = {
            "artifact_id": artifact_id,
            "source_rows": len(source_reconciliation),
            "session_rows": len(session_resolution),
            "source_representation_rules": normalized_rules,
            "input_identity": input_identity,
        }
        atomic_write_json(staging / "summary.json", summary)
        hashes = {
            name: file_sha256(staging / name)
            for name in (
                "source_reconciliation.parquet",
                "session_resolution.parquet",
                "summary.json",
            )
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": SOURCE_RECONCILIATION_SCHEMA_VERSION,
                "artifact_name": SOURCE_RECONCILIATION_ARTIFACT_NAME,
                "artifact_id": artifact_id,
                "logical_identity": logical,
                "artifact_hashes": hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        validate_source_reconciliation_artifact(staging, expected_artifact_id=artifact_id)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_source_reconciliation_artifact(output)
    return output


def validate_source_reconciliation_artifact(
    path: Path, *, expected_artifact_id: str | None = None
) -> tuple[JsonObject, str]:
    """Validate frozen source rows, exact session reconciliation, and rule identity."""

    manifest = _read_json(path / "manifest.json", "SECURITY_SOURCE_RECONCILIATION_INVALID")
    summary = _read_json(path / "summary.json", "SECURITY_SOURCE_RECONCILIATION_INVALID")
    artifact_id = expected_artifact_id or path.name
    if (
        manifest.get("schema_version") != SOURCE_RECONCILIATION_SCHEMA_VERSION
        or manifest.get("artifact_name") != SOURCE_RECONCILIATION_ARTIFACT_NAME
        or manifest.get("artifact_id") != artifact_id
        or summary.get("artifact_id") != artifact_id
    ):
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
    expected_files = {
        "source_reconciliation.parquet",
        "session_resolution.parquet",
        "summary.json",
    }
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != expected_files:
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
    for name, expected in hashes.items():
        child = path / str(name)
        if not child.is_file() or file_sha256(child) != expected:
            raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_HASH_MISMATCH")
    rows = pd.read_parquet(path / "source_reconciliation.parquet")
    sessions = pd.read_parquet(path / "session_resolution.parquet")
    if (
        not _SOURCE_RECONCILIATION_REQUIRED_COLUMNS.issubset(rows.columns)
        or len(rows) != summary.get("source_rows")
        or len(sessions) != summary.get("session_rows")
        or sessions.empty
        or not {"parent_interval_id", "trade_date", "resolution"}.issubset(sessions.columns)
        or sessions.duplicated(["parent_interval_id", "trade_date"]).any()
    ):
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
    rules = summary.get("source_representation_rules")
    if not isinstance(rules, list):
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
    for rule in rules:
        if not isinstance(rule, dict):
            raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
        _parse_source_representation_rule(
            {
                **rule,
                "evidence_package_id": artifact_id,
                "evidence_package_hash": "0" * 64,
            }
        )
    logical = {
        "input_identity": summary.get("input_identity"),
        "source_reconciliation_hash": _frame_logical_hash(rows),
        "session_resolution_hash": _frame_logical_hash(sessions),
        "source_representation_rules": rules,
    }
    if logical != manifest.get("logical_identity"):
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_IDENTITY_MISMATCH")
    expected_id = f"{SOURCE_RECONCILIATION_ARTIFACT_NAME}_{canonical_payload_hash(logical)[:24]}"
    if expected_id != artifact_id:
        raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_IDENTITY_MISMATCH")
    return summary, file_sha256(path / "manifest.json")


def publish_security_identity_transitions(
    *,
    evidence_packages: tuple[Path, ...],
    reports_root: Path,
    transition_version: str,
    source_representation_artifacts: tuple[Path, ...] = (),
) -> Path:
    """Publish a versioned transition graph from VERIFIED official packages."""

    records: list[JsonObject] = []
    package_hashes: list[str] = []
    for package in evidence_packages:
        evidence, package_hash = validate_transition_evidence_package(package)
        if evidence.get("status") != "VERIFIED":
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_NOT_VERIFIED")
        package_hashes.append(package_hash)
        records.append(
            {
                **{
                    key: evidence[key]
                    for key in (
                        "predecessor_ts_code",
                        "successor_ts_code",
                        "predecessor_name",
                        "successor_name",
                        "transition_type",
                        "effective_date",
                        "continuity_type",
                        "share_conversion_ratio",
                    )
                },
                "evidence_package_id": package.name,
                "evidence_package_hash": package_hash,
            }
        )
    parsed = tuple(_parse_transition(row) for row in records)
    representation_records: list[JsonObject] = []
    representation_hashes: list[str] = []
    for artifact in source_representation_artifacts:
        reconciliation, reconciliation_hash = validate_source_reconciliation_artifact(artifact)
        representation_hashes.append(reconciliation_hash)
        rules = reconciliation.get("source_representation_rules")
        if not isinstance(rules, list):
            raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
        for raw_rule in rules:
            if not isinstance(raw_rule, dict):
                raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_INVALID")
            representation_records.append(
                {
                    **raw_rule,
                    "evidence_package_id": artifact.name,
                    "evidence_package_hash": reconciliation_hash,
                }
            )
    parsed_rules = tuple(_parse_source_representation_rule(row) for row in representation_records)
    SecurityIdentityTransitionResolver(
        artifact_version=transition_version,
        artifact_hash="0" * 64,
        transitions=parsed,
        source_representation_rules=parsed_rules,
    )
    ordered = sorted(
        records, key=lambda item: (str(item["effective_date"]), str(item["predecessor_ts_code"]))
    )
    payload = {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "artifact_name": TRANSITION_ARTIFACT_NAME,
        "transition_version": transition_version,
        "transitions": ordered,
        "source_representation_rules": sorted(
            representation_records,
            key=lambda item: (
                str(item["dataset_name"]),
                str(item["source_ts_code"]),
                str(item["effective_from"]),
            ),
        ),
    }
    logical = {
        "transition_version": transition_version,
        "evidence_package_hashes": sorted(package_hashes),
        "transitions_hash": canonical_payload_hash(ordered),
        "source_reconciliation_hashes": sorted(representation_hashes),
        "source_representation_rules_hash": canonical_payload_hash(
            payload["source_representation_rules"]
        ),
    }
    artifact_id = f"security_identity_transitions_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_identity_transitions" / artifact_id
    if output.exists():
        validate_security_identity_transition_artifact(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{artifact_id}.staging-"))
    try:
        atomic_write_json(staging / "transitions.json", payload)
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": TRANSITION_SCHEMA_VERSION,
                "artifact_name": TRANSITION_ARTIFACT_NAME,
                "artifact_id": artifact_id,
                "transition_version": transition_version,
                "transition_count": len(ordered),
                "source_representation_rule_count": len(representation_records),
                "logical_identity": logical,
                "artifact_hashes": {"transitions.json": file_sha256(staging / "transitions.json")},
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        _validate_transition_artifact_contents(staging, expected_artifact_id=artifact_id)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_security_identity_transition_artifact(output)
    return output


def validate_security_identity_transition_artifact(path: Path) -> JsonObject:
    """Validate one complete transition graph artifact."""

    return _validate_transition_artifact_contents(path, expected_artifact_id=path.name)


def _validate_transition_artifact_contents(path: Path, *, expected_artifact_id: str) -> JsonObject:
    """Validate an artifact in staging or at its immutable final path."""

    manifest = _read_json(path / "manifest.json", "SECURITY_IDENTITY_TRANSITION_INVALID")
    payload = _read_json(path / "transitions.json", "SECURITY_IDENTITY_TRANSITION_INVALID")
    if (
        manifest.get("schema_version")
        not in LEGACY_TRANSITION_SCHEMA_VERSIONS | {TRANSITION_SCHEMA_VERSION}
        or manifest.get("artifact_name") != TRANSITION_ARTIFACT_NAME
        or manifest.get("artifact_id") != expected_artifact_id
        or payload.get("artifact_name") != TRANSITION_ARTIFACT_NAME
        or payload.get("transition_version") != manifest.get("transition_version")
    ):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
    hashes = manifest.get("artifact_hashes")
    if hashes != {"transitions.json": file_sha256(path / "transitions.json")}:
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_HASH_MISMATCH")
    rows = payload.get("transitions")
    if not isinstance(rows, list) or len(rows) != manifest.get("transition_count"):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
    parsed = tuple(_parse_transition(cast(JsonObject, row)) for row in rows)
    schema_version = int(manifest["schema_version"])
    rule_rows = payload.get("source_representation_rules", [])
    if not isinstance(rule_rows, list):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
    parsed_rules = tuple(
        _parse_source_representation_rule(cast(JsonObject, row)) for row in rule_rows
    )
    if schema_version == 1 and rule_rows:
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
    if schema_version == TRANSITION_SCHEMA_VERSION and len(rule_rows) != manifest.get(
        "source_representation_rule_count"
    ):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID")
    SecurityIdentityTransitionResolver(
        artifact_version=str(payload.get("transition_version", "")),
        artifact_hash=file_sha256(path / "manifest.json"),
        transitions=parsed,
        source_representation_rules=parsed_rules,
    )
    logical = _transition_logical_identity(
        transition_version=str(payload["transition_version"]),
        transitions=rows,
        parsed=parsed,
        source_representation_rules=rule_rows,
        parsed_rules=parsed_rules,
        schema_version=schema_version,
    )
    if logical != manifest.get("logical_identity"):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_IDENTITY_MISMATCH")
    expected_id = f"security_identity_transitions_{canonical_payload_hash(logical)[:24]}"
    if expected_id != expected_artifact_id:
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_IDENTITY_MISMATCH")
    evidence_root = path.parent.parent / "security_identity_transition_evidence"
    if path.name.startswith("."):
        evidence_root = path.parent.parent / "security_identity_transition_evidence"
    for row, item in zip(rows, parsed, strict=True):
        package = evidence_root / item.evidence_package_id
        evidence, package_hash = validate_transition_evidence_package(package)
        if package_hash != item.evidence_package_hash:
            raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_HASH_MISMATCH")
        for field in (
            "predecessor_ts_code",
            "successor_ts_code",
            "predecessor_name",
            "successor_name",
            "transition_type",
            "effective_date",
            "continuity_type",
            "share_conversion_ratio",
        ):
            if row.get(field) != evidence.get(field):
                raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_IDENTITY_MISMATCH")
    if schema_version == TRANSITION_SCHEMA_VERSION:
        reconciliation_root = path.parent.parent / SOURCE_RECONCILIATION_ARTIFACT_NAME
        for rule_row, parsed_rule in zip(rule_rows, parsed_rules, strict=True):
            artifact = reconciliation_root / parsed_rule.evidence_package_id
            reconciliation, artifact_hash = validate_source_reconciliation_artifact(artifact)
            if artifact_hash != parsed_rule.evidence_package_hash:
                raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_HASH_MISMATCH")
            rules = reconciliation.get("source_representation_rules")
            if not isinstance(rules, list) or not any(
                all(candidate.get(key) == rule_row.get(key) for key in _SOURCE_RULE_FIELDS)
                for candidate in rules
                if isinstance(candidate, dict)
            ):
                raise DataValidationError("SECURITY_SOURCE_RECONCILIATION_IDENTITY_MISMATCH")
    return manifest


def _normalize_evidence(evidence: JsonObject, document: Path) -> JsonObject:
    required = {
        "predecessor_ts_code",
        "successor_ts_code",
        "predecessor_name",
        "successor_name",
        "transition_type",
        "effective_date",
        "continuity_type",
        "share_conversion_ratio",
        "official_source_type",
        "official_url",
        "official_document_id",
        "publication_date",
        "reviewed_fact",
        "status",
    }
    if (
        not required.issubset(evidence)
        or evidence.get("status") != "VERIFIED"
        or not document.is_file()
    ):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_INVALID")
    host = (urlparse(str(evidence["official_url"])).hostname or "").lower()
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_SOURCE_INVALID")
    ratio = evidence.get("share_conversion_ratio")
    if isinstance(ratio, bool):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_EVIDENCE_INVALID")
    normalized: JsonObject = {
        **{key: evidence[key] for key in required},
        "predecessor_ts_code": _code(str(evidence["predecessor_ts_code"])),
        "successor_ts_code": _code(str(evidence["successor_ts_code"])),
        "share_conversion_ratio": None if ratio is None else float(cast(float, ratio)),
        "document_sha256": file_sha256(document),
        "retrieved_at": str(evidence.get("retrieved_at", "")),
    }
    transition = _parse_transition(
        {
            **normalized,
            "evidence_package_id": str(evidence.get("evidence_package_id", "pending")),
            "evidence_package_hash": str(evidence.get("evidence_package_hash", "0" * 64)),
        }
    )
    normalized.update(asdict(transition))
    normalized.pop("evidence_package_id", None)
    normalized.pop("evidence_package_hash", None)
    return normalized


def _parse_transition(row: JsonObject) -> SecurityIdentityTransition:
    try:
        ratio = row.get("share_conversion_ratio")
        return SecurityIdentityTransition(
            predecessor_ts_code=_code(str(row["predecessor_ts_code"])),
            successor_ts_code=_code(str(row["successor_ts_code"])),
            predecessor_name=str(row["predecessor_name"]).strip(),
            successor_name=str(row["successor_name"]).strip(),
            transition_type=str(row["transition_type"]).strip(),
            effective_date=str(row["effective_date"]).strip(),
            continuity_type=str(row["continuity_type"]).strip(),
            share_conversion_ratio=None if ratio is None else float(cast(float, ratio)),
            evidence_package_id=str(row["evidence_package_id"]).strip(),
            evidence_package_hash=str(row["evidence_package_hash"]).strip(),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID") from error


def _code(value: str) -> str:
    code = value.strip().upper()
    if (
        len(code) != 9
        or code[6] != "."
        or code[7:] not in {"SH", "SZ", "BJ"}
        or not code[:6].isdigit()
    ):
        raise DataValidationError("SECURITY_IDENTITY_TRANSITION_INVALID: ts_code")
    return code


def _is_date(value: str) -> bool:
    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        return False


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _valid_type_continuity(transition_type: str, continuity_type: str) -> bool:
    if transition_type == "CODE_CHANGE_CONTINUITY":
        return continuity_type == "SAME_LISTED_ENTITY"
    if transition_type in {"MERGER_SUCCESSOR", "SHARE_CONVERSION"}:
        return continuity_type == "SUCCESSOR_ENTITY"
    return transition_type == "RESTRUCTURING_CODE_CHANGE"


def _parse_source_representation_rule(row: JsonObject) -> SecuritySourceRepresentationRule:
    try:
        return SecuritySourceRepresentationRule(
            dataset_name=str(row["dataset_name"]),
            source_ts_code=_code(str(row["source_ts_code"])),
            effective_ts_code=_code(str(row["effective_ts_code"])),
            effective_from=str(row["effective_from"]),
            effective_to=str(row["effective_to"]),
            resolution_rule_id=str(row["resolution_rule_id"]),
            evidence_package_id=str(row["evidence_package_id"]),
            evidence_package_hash=str(row["evidence_package_hash"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DataValidationError("SECURITY_SOURCE_REPRESENTATION_RULE_INVALID") from error


def _transition_logical_identity(
    *,
    transition_version: str,
    transitions: list[object],
    parsed: tuple[SecurityIdentityTransition, ...],
    source_representation_rules: list[object],
    parsed_rules: tuple[SecuritySourceRepresentationRule, ...],
    schema_version: int,
) -> JsonObject:
    logical: JsonObject = {
        "transition_version": transition_version,
        "evidence_package_hashes": sorted(item.evidence_package_hash for item in parsed),
        "transitions_hash": canonical_payload_hash(transitions),
    }
    if schema_version == TRANSITION_SCHEMA_VERSION:
        logical.update(
            {
                "source_reconciliation_hashes": sorted(
                    {item.evidence_package_hash for item in parsed_rules}
                ),
                "source_representation_rules_hash": canonical_payload_hash(
                    source_representation_rules
                ),
            }
        )
    return logical


def _deduplicate_source_representations(frame: pd.DataFrame, *, dataset_name: str) -> pd.DataFrame:
    if dataset_name != "suspend_d":
        return frame
    semantic_columns = [
        column for column in ("suspend_type", "suspend_timing") if column in frame.columns
    ]
    keys = ["ts_code", "trade_date"]
    output: list[pd.DataFrame] = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        source_codes = tuple(sorted(group["source_ts_code"].astype(str).unique()))
        if len(source_codes) > 1:
            signatures = {
                source: {
                    tuple("<NULL>" if pd.isna(value) else str(value) for value in row)
                    for row in selected[semantic_columns].itertuples(index=False, name=None)
                }
                for source, selected in group.groupby("source_ts_code", sort=False)
            }
            if len({tuple(sorted(value)) for value in signatures.values()}) != 1:
                raise DataValidationError("SECURITY_SOURCE_REPRESENTATION_CONFLICT")
        deduplicated = group.drop_duplicates(subset=keys + semantic_columns, keep="first").copy()
        deduplicated["source_ts_codes"] = ",".join(source_codes)
        output.append(deduplicated)
    return pd.concat(output, ignore_index=True) if output else frame.iloc[0:0].copy()


def _frame_logical_hash(frame: pd.DataFrame) -> str:
    columns = sorted(str(column) for column in frame.columns)
    ordered = frame[columns].sort_values(columns, na_position="first")
    records = [
        {
            column: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for column, value in zip(columns, row, strict=True)
        }
        for row in ordered.itertuples(index=False, name=None)
    ]
    return canonical_payload_hash({"columns": columns, "records": records})


def _read_json(path: Path, message: str) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(message) from error
    if not isinstance(value, dict):
        raise DataValidationError(message)
    return cast(JsonObject, value)


def canonical_payload_hash(payload: object) -> str:
    """Hash a JSON-compatible logical payload without path or time metadata."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DataValidationError("CANONICAL_JSON_INVALID") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one immutable artifact child."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
