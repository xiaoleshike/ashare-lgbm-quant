"""Immutable official-exchange lifecycle source packages and normalized index."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import canonical_payload_hash, file_sha256
from ashare_quant.data.security_lifecycle_resolution import validate_official_evidence_package
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

BULK_PACKAGE_SCHEMA_VERSION = 1
OFFICIAL_INDEX_SCHEMA_VERSION = 3
OFFICIAL_INDEX_CONTRACT_VERSION = 3
BULK_ARTIFACT_NAME = "security_lifecycle_bulk_official_source"
INDEX_ARTIFACT_NAME = "security_lifecycle_official_index"
OFFICIAL_HOST_SUFFIXES = ("sse.com.cn", "szse.cn", "bse.cn")
SUPPORTED_EVENT_TYPES = frozenset(
    {
        "FORMAL_LISTING_SUSPENSION_START",
        "FORMAL_LISTING_RESUMPTION",
        "TERMINAL_DELISTING",
        "ORDINARY_FULL_DAY_SUSPENSION",
        "ORDINARY_RESUMPTION",
        "MERGER_TERMINATION",
        "SECURITY_CODE_TRANSITION",
        "OTHER_LIFECYCLE_EVENT",
    }
)
BULK_FILES = frozenset({"source.json", "normalized_rows.parquet"})
INDEX_FILES = frozenset({"summary.json", "official_events.parquet", "source_inventory.json"})


@dataclass(frozen=True, slots=True)
class OfficialLifecycleIndexResult:
    """Published normalized official lifecycle index."""

    index_id: str
    output_dir: Path
    counts: JsonObject
    idempotent: bool


def publish_bulk_official_source_package(
    *,
    source: JsonObject,
    records: list[JsonObject],
    document: Path,
    reports_root: Path,
) -> Path:
    """Freeze one official bulk document and deterministic normalized rows."""

    normalized_source = _validate_source(source, document)
    rows = _normalize_records(records, normalized_source)
    logical = {
        "source": {key: value for key, value in normalized_source.items() if key != "retrieved_at"},
        "normalized_rows_hash": _frame_hash(rows),
    }
    package_id = f"security_lifecycle_bulk_source_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_lifecycle_bulk_source" / package_id
    if output.exists():
        validate_bulk_official_source_package(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{package_id}.staging-"))
    try:
        documents = staging / "documents"
        documents.mkdir()
        document_name = f"document{document.suffix.lower()}"
        shutil.copyfile(document, documents / document_name)
        source_payload = {**normalized_source, "document_path": f"documents/{document_name}"}
        atomic_write_json(staging / "source.json", source_payload)
        rows.to_parquet(staging / "normalized_rows.parquet", index=False)
        hashes = {
            "source.json": file_sha256(staging / "source.json"),
            "normalized_rows.parquet": file_sha256(staging / "normalized_rows.parquet"),
            f"documents/{document_name}": file_sha256(documents / document_name),
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": BULK_PACKAGE_SCHEMA_VERSION,
                "artifact_name": BULK_ARTIFACT_NAME,
                "package_id": package_id,
                "logical_identity": logical,
                "record_count": int(len(rows)),
                "artifact_hashes": hashes,
                "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_bulk_official_source_package(output)
    return output


def validate_bulk_official_source_package(path: Path) -> tuple[JsonObject, DataFrame]:
    """Validate a bulk source package through document and normalized-row hashes."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        source = json.loads((path / "source.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != BULK_PACKAGE_SCHEMA_VERSION
        or manifest.get("artifact_name") != BULK_ARTIFACT_NAME
        or manifest.get("package_id") != path.name
        or not isinstance(source, dict)
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != {
        *BULK_FILES,
        str(source.get("document_path", "")),
    }:
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_SET_INVALID")
    for relative, expected in hashes.items():
        child = (path / str(relative)).resolve()
        if path.resolve() not in child.parents or not child.is_file():
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_PATH_INVALID")
        if not isinstance(expected, str) or file_sha256(child) != expected:
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_HASH_MISMATCH")
    document = path / str(source["document_path"])
    normalized_source = _validate_source(cast(JsonObject, source), document)
    rows = pd.read_parquet(path / "normalized_rows.parquet")
    checked = _normalize_records(cast(list[JsonObject], rows.to_dict("records")), normalized_source)
    logical = {
        "source": {key: value for key, value in normalized_source.items() if key != "retrieved_at"},
        "normalized_rows_hash": _frame_hash(checked),
    }
    if manifest.get("logical_identity") != logical or len(checked) != manifest.get("record_count"):
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_IDENTITY_MISMATCH")
    return {
        **cast(JsonObject, source),
        "package_id": path.name,
        "package_hash": file_sha256(path / "manifest.json"),
    }, checked


class OfficialLifecycleIndexService:
    """Combine verified bulk, individual and existing-v3 evidence into one index."""

    def __init__(
        self,
        *,
        reports_root: Path,
        identity_resolver: SecurityIdentityResolver,
        lifecycle_evidence: SecurityLifecycleResolver,
        bulk_source_packages: tuple[Path, ...] = (),
        official_evidence_packages: tuple[Path, ...] = (),
    ) -> None:
        self.reports_root = reports_root
        self.identity = identity_resolver
        self.lifecycle = lifecycle_evidence
        self.bulk_paths = bulk_source_packages
        self.evidence_paths = official_evidence_packages

    def build(self) -> OfficialLifecycleIndexResult:
        """Validate sources, pair lifecycle transitions and publish the index."""

        inventory: list[JsonObject] = []
        frames: list[DataFrame] = []
        for path in self.bulk_paths:
            source, rows = validate_bulk_official_source_package(path)
            source["record_count"] = len(rows)
            inventory.append(_source_inventory(source, "BULK_OFFICIAL_SOURCE"))
            frames.append(_attach_package(rows, source))
        for path in self.evidence_paths:
            evidence = validate_official_evidence_package(path)
            inventory.append(_source_inventory(evidence, "INDIVIDUAL_OFFICIAL_EVIDENCE"))
            frames.append(_individual_evidence_frame(evidence))
        v3 = _v3_frame(self.lifecycle)
        if not v3.empty:
            frames.append(v3)
            inventory.append(
                {
                    "source_kind": "EXISTING_V3",
                    "package_id": self.lifecycle.policy_version,
                    "package_hash": self.lifecycle.policy_hash,
                    "record_count": self.lifecycle.event_count,
                    "exchange": "MULTI",
                }
            )
        events = pd.concat(frames, ignore_index=True) if frames else _empty_events()
        events = _normalize_index_events(events, self.identity)
        logical: JsonObject = {
            "schema_version": OFFICIAL_INDEX_SCHEMA_VERSION,
            "contract_version": OFFICIAL_INDEX_CONTRACT_VERSION,
            "security_identity_mapping_version": self.identity.mapping_version,
            "security_identity_mapping_hash": self.identity.mapping_hash,
            "lifecycle_evidence_version": self.lifecycle.policy_version,
            "lifecycle_evidence_hash": self.lifecycle.policy_hash,
            "source_package_hashes": sorted(str(item["package_hash"]) for item in inventory),
            "official_events_hash": _frame_hash(events),
        }
        index_id = f"security_lifecycle_official_index_{canonical_payload_hash(logical)[:24]}"
        output = self.reports_root / "security_lifecycle_official_index" / index_id
        if output.exists():
            manifest = validate_official_lifecycle_index(output)
            return OfficialLifecycleIndexResult(
                index_id, output, cast(JsonObject, manifest["counts"]), True
            )
        counts = {
            "source_packages": len(inventory),
            "events": int(len(events)),
            "securities": int(events["canonical_ts_code"].nunique()) if not events.empty else 0,
            "bulk_records": sum(
                int(cast(int, item["record_count"]))
                for item in inventory
                if item["source_kind"] == "BULK_OFFICIAL_SOURCE"
            ),
        }
        self._publish(output, logical, counts, inventory, events)
        return OfficialLifecycleIndexResult(index_id, output, counts, False)

    def _publish(
        self,
        output: Path,
        logical: JsonObject,
        counts: JsonObject,
        inventory: list[JsonObject],
        events: DataFrame,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}.staging-"))
        try:
            atomic_write_json(staging / "summary.json", {"index_id": output.name, "counts": counts})
            events.to_parquet(staging / "official_events.parquet", index=False)
            atomic_write_json(staging / "source_inventory.json", {"sources": inventory})
            hashes = {name: file_sha256(staging / name) for name in sorted(INDEX_FILES)}
            atomic_write_json(
                staging / "manifest.json",
                {
                    "schema_version": OFFICIAL_INDEX_SCHEMA_VERSION,
                    "artifact_name": INDEX_ARTIFACT_NAME,
                    "index_id": output.name,
                    "logical_identity": logical,
                    "counts": counts,
                    "artifact_hashes": hashes,
                    "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
            )
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def validate_official_lifecycle_index(path: Path) -> JsonObject:
    """Validate one immutable official lifecycle index."""

    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_INDEX_INVALID") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") not in {1, OFFICIAL_INDEX_SCHEMA_VERSION}
        or manifest.get("artifact_name") != INDEX_ARTIFACT_NAME
        or manifest.get("index_id") != path.name
    ):
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_INDEX_INVALID")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != INDEX_FILES:
        raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_INDEX_SET_INVALID")
    for name, expected in hashes.items():
        if not isinstance(expected, str) or file_sha256(path / name) != expected:
            raise DataValidationError(f"SECURITY_LIFECYCLE_OFFICIAL_INDEX_HASH_MISMATCH: {name}")
    return cast(JsonObject, manifest)


def _validate_source(source: JsonObject, document: Path) -> JsonObject:
    required = {
        "exchange",
        "official_source_type",
        "official_url",
        "official_document_id",
        "source_year",
        "retrieved_at",
        "document_hash",
    }
    if not required.issubset(source):
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_METADATA_INCOMPLETE")
    exchange = str(source["exchange"]).upper()
    if exchange not in {"SSE", "SZSE", "BSE"}:
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_EXCHANGE_INVALID")
    host = (urlparse(str(source["official_url"])).hostname or "").lower()
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES):
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_DOMAIN_INVALID")
    if not document.is_file() or file_sha256(document) != source["document_hash"]:
        raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_DOCUMENT_HASH_MISMATCH")
    return {key: source[key] for key in sorted(required)}


def _normalize_records(records: list[JsonObject], source: JsonObject) -> DataFrame:
    columns = [
        "canonical_ts_code",
        "security_name",
        "exchange",
        "event_type",
        "announcement_date",
        "effective_start",
        "effective_end",
        "source_locator",
        "source_year",
        "row_identity",
    ]
    rows: list[JsonObject] = []
    for record in records:
        code = str(record.get("canonical_ts_code", "")).strip().upper()
        exchange = str(record.get("exchange", source["exchange"])).strip().upper()
        event_type = str(record.get("event_type", "")).strip().upper()
        start = _date(str(record.get("effective_start", "")), "effective_start")
        end = _date(str(record.get("effective_end", start)), "effective_end")
        announcement = _date(str(record.get("announcement_date", start)), "announcement_date")
        if event_type not in SUPPORTED_EVENT_TYPES or start > end:
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_EVENT_INVALID")
        expected_suffix = {"SSE": ".SH", "SZSE": ".SZ", "BSE": ".BJ"}[exchange]
        if not code.endswith(expected_suffix):
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_SECURITY_MISMATCH")
        normalized = {
            "canonical_ts_code": code,
            "security_name": str(record.get("security_name", "")).strip(),
            "exchange": exchange,
            "event_type": event_type,
            "announcement_date": announcement,
            "effective_start": start,
            "effective_end": end,
            "source_locator": str(record.get("source_locator", "")).strip(),
            "source_year": int(str(record.get("source_year", source["source_year"]))),
        }
        if not normalized["security_name"] or not normalized["source_locator"]:
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_ROW_INCOMPLETE")
        normalized["row_identity"] = f"official_row_{canonical_payload_hash(normalized)[:24]}"
        supplied_id = record.get("row_identity")
        if supplied_id is not None and str(supplied_id) != normalized["row_identity"]:
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_ROW_IDENTITY_MISMATCH")
        rows.append(normalized)
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates().reset_index(drop=True)
    _validate_record_conflicts(frame)
    return frame.sort_values(
        ["canonical_ts_code", "effective_start", "event_type", "row_identity"]
    ).reset_index(drop=True)


def _validate_record_conflicts(frame: DataFrame) -> None:
    if frame.empty:
        return
    identity = ["canonical_ts_code", "effective_start", "effective_end"]
    for _, group in frame.groupby(identity, sort=False):
        types = set(group["event_type"].astype(str))
        if len(types) > 1 and not types.issubset(
            {"FORMAL_LISTING_RESUMPTION", "ORDINARY_RESUMPTION"}
        ):
            raise DataValidationError("SECURITY_LIFECYCLE_BULK_SOURCE_CONFLICT")


def _normalize_index_events(frame: DataFrame, identity: SecurityIdentityResolver) -> DataFrame:
    if frame.empty:
        return _empty_events()
    result = frame.copy()
    result["canonical_ts_code"] = [
        identity.canonicalize(code, as_of_date=date)
        for code, date in zip(
            result["canonical_ts_code"].astype(str),
            result["effective_start"].astype(str),
            strict=True,
        )
    ]
    result = result.drop_duplicates(
        [
            "canonical_ts_code",
            "event_type",
            "effective_start",
            "effective_end",
            "document_hash",
            "source_locator",
        ]
    ).reset_index(drop=True)
    _validate_record_conflicts(result)
    return result.sort_values(
        ["canonical_ts_code", "effective_start", "event_type", "package_id"]
    ).reset_index(drop=True)


def _attach_package(rows: DataFrame, source: JsonObject) -> DataFrame:
    result = rows.copy()
    result["package_id"] = source["package_id"]
    result["package_hash"] = source["package_hash"]
    result["document_hash"] = source["document_hash"]
    result["official_url"] = source["official_url"]
    result["official_document_id"] = source["official_document_id"]
    result["official_source_type"] = source["official_source_type"]
    result["evidence_source_category"] = f"{source['exchange']}_BULK_LIFECYCLE_INDEX"
    result["status"] = "VERIFIED"
    result["carry_in"] = result["effective_start"].astype(str).lt("20100101")
    return result


def _individual_evidence_frame(evidence: JsonObject) -> DataFrame:
    event_map = {
        "ORDINARY_SUSPENSION": "ORDINARY_FULL_DAY_SUSPENSION",
        "LISTING_SUSPENDED": "FORMAL_LISTING_SUSPENSION_START",
        "TERMINAL_DELISTING": "TERMINAL_DELISTING",
        "MERGER_TERMINATION": "MERGER_TERMINATION",
        "SECURITY_CODE_TRANSITION": "SECURITY_CODE_TRANSITION",
    }
    event_type = event_map.get(str(evidence["event_type"]), "OTHER_LIFECYCLE_EVENT")
    exchange = _exchange(str(evidence["canonical_ts_code"]))
    return pd.DataFrame(
        [
            {
                "canonical_ts_code": evidence["canonical_ts_code"],
                "security_name": "",
                "exchange": exchange,
                "event_type": event_type,
                "announcement_date": evidence["publication_date"],
                "effective_start": evidence["effective_start"],
                "effective_end": evidence["effective_end"],
                "source_locator": evidence["official_document_id"],
                "source_year": int(str(evidence["publication_date"])[:4]),
                "row_identity": (
                    f"official_row_{canonical_payload_hash(_evidence_row_identity(evidence))[:24]}"
                ),
                "package_id": evidence["package_id"],
                "package_hash": evidence["package_hash"],
                "document_hash": evidence["document_hash"],
                "official_url": evidence["official_url"],
                "official_document_id": evidence["official_document_id"],
                "official_source_type": evidence["official_source_type"],
                "evidence_source_category": f"{exchange}_COMPANY_ANNOUNCEMENT",
                "status": evidence["evidence_status"],
                "carry_in": str(evidence["effective_start"]) < "20100101",
            }
        ]
    )


def _v3_frame(lifecycle: SecurityLifecycleResolver) -> DataFrame:
    rows = []
    for event in lifecycle.events():
        exchange = _exchange(event.canonical_ts_code)
        identity = {
            "canonical_ts_code": event.canonical_ts_code,
            "event_type": "FORMAL_LISTING_SUSPENSION_START",
            "effective_start": event.effective_from,
            "effective_end": event.effective_to,
            "source_locator": event.evidence_reference,
        }
        rows.append(
            {
                **identity,
                "security_name": "",
                "exchange": exchange,
                "announcement_date": event.effective_from,
                "source_year": int(event.effective_from[:4]),
                "row_identity": f"official_row_{canonical_payload_hash(identity)[:24]}",
                "package_id": lifecycle.policy_version,
                "package_hash": lifecycle.policy_hash,
                "document_hash": lifecycle.policy_hash,
                "official_url": event.evidence_reference,
                "official_document_id": event.evidence_reference,
                "official_source_type": "EXISTING_V3",
                "evidence_source_category": "EXISTING_V3",
                "status": "VERIFIED",
                "carry_in": event.effective_from < "20100101",
            }
        )
    return pd.DataFrame(rows, columns=_empty_events().columns)


def _empty_events() -> DataFrame:
    return pd.DataFrame(
        columns=[
            "canonical_ts_code",
            "security_name",
            "exchange",
            "event_type",
            "announcement_date",
            "effective_start",
            "effective_end",
            "source_locator",
            "source_year",
            "row_identity",
            "package_id",
            "package_hash",
            "document_hash",
            "official_url",
            "official_document_id",
            "official_source_type",
            "evidence_source_category",
            "status",
            "carry_in",
        ]
    )


def _source_inventory(source: JsonObject, kind: str) -> JsonObject:
    exchange = source.get("exchange")
    if exchange is None:
        exchange = _exchange(str(source.get("canonical_ts_code", "")))
    return {
        "source_kind": kind,
        "package_id": source["package_id"],
        "package_hash": source["package_hash"],
        "record_count": int(source.get("record_count", 1)),
        "exchange": exchange,
    }


def _evidence_row_identity(evidence: JsonObject) -> JsonObject:
    return {
        "canonical_ts_code": evidence["canonical_ts_code"],
        "event_type": evidence["event_type"],
        "effective_start": evidence["effective_start"],
        "effective_end": evidence["effective_end"],
        "document_hash": evidence["document_hash"],
    }


def _exchange(code: str) -> str:
    if code.endswith(".SH"):
        return "SSE"
    if code.endswith(".SZ"):
        return "SZSE"
    if code.endswith(".BJ"):
        return "BSE"
    raise DataValidationError("SECURITY_LIFECYCLE_OFFICIAL_INDEX_SECURITY_INVALID")


def _date(value: str, field: str) -> str:
    normalized = value.replace("-", "").replace("/", "")
    if len(normalized) != 8 or not normalized.isdigit():
        raise DataValidationError(f"SECURITY_LIFECYCLE_BULK_SOURCE_DATE_INVALID: {field}")
    return normalized


def _frame_hash(frame: DataFrame) -> str:
    if frame.empty:
        return canonical_payload_hash({"columns": list(frame.columns), "rows": []})
    columns = sorted(frame.columns)
    working = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
    rows = working.sort_values(columns, key=lambda values: values.astype(str)).to_dict("records")
    return canonical_payload_hash({"columns": columns, "rows": rows})
