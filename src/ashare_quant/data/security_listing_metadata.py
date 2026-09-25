"""Verified research overlays for authoritative security listing metadata."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json

type DataFrame = pd.DataFrame
type JsonObject = dict[str, Any]

EVIDENCE_SCHEMA_VERSION = 1
OVERLAY_SCHEMA_VERSION = 1
EVIDENCE_ARTIFACT_NAME = "security_listing_metadata_evidence"
OVERLAY_ARTIFACT_NAME = "security_listing_metadata_overlay"
OFFICIAL_HOST_SUFFIXES = ("sse.com.cn", "szse.cn", "bse.cn", "chinaclear.cn")


def canonical_payload_hash(value: object) -> str:
    """Hash canonical JSON without accepting non-finite values."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash one immutable artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SecurityListingMetadataRecord:
    """One authoritative listing date and its immutable evidence identity."""

    canonical_ts_code: str
    authoritative_list_date: str
    evidence_package_id: str
    evidence_package_hash: str


class SecurityListingMetadataResolver:
    """Apply verified listing dates without mutating canonical ``stock_basic``."""

    def __init__(
        self,
        *,
        overlay_version: str,
        overlay_hash: str,
        records: tuple[SecurityListingMetadataRecord, ...],
        overlay_path: Path | None = None,
    ) -> None:
        self.overlay_version = overlay_version
        self.overlay_hash = overlay_hash
        self.overlay_path = overlay_path
        self._records = records
        codes = [record.canonical_ts_code for record in records]
        if len(codes) != len(set(codes)):
            raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_DUPLICATE_SECURITY")

    @classmethod
    def empty(cls) -> SecurityListingMetadataResolver:
        """Return an explicit no-overlay contract."""

        return cls(
            overlay_version="none",
            overlay_hash=hashlib.sha256(b"security-listing-metadata:none").hexdigest(),
            records=(),
        )

    @classmethod
    def from_path(cls, path: Path | None) -> SecurityListingMetadataResolver:
        """Load and recursively validate an immutable overlay artifact."""

        if path is None:
            return cls.empty()
        manifest = validate_listing_metadata_overlay(path)
        rows = pd.read_parquet(path / "listing_metadata.parquet")
        records = tuple(
            SecurityListingMetadataRecord(
                canonical_ts_code=str(row.canonical_ts_code),
                authoritative_list_date=str(row.authoritative_list_date),
                evidence_package_id=str(row.evidence_package_id),
                evidence_package_hash=str(row.evidence_package_hash),
            )
            for row in rows.itertuples(index=False)
        )
        return cls(
            overlay_version=str(manifest["overlay_id"]),
            overlay_hash=file_sha256(path / "manifest.json"),
            records=records,
            overlay_path=path,
        )

    @property
    def record_count(self) -> int:
        """Return the number of governed metadata overrides."""

        return len(self._records)

    def records(self) -> tuple[SecurityListingMetadataRecord, ...]:
        """Return immutable verified records."""

        return self._records

    def frame(self) -> DataFrame:
        """Return records in scanner-friendly tabular form."""

        frame = pd.DataFrame(
            [
                {
                    "canonical_ts_code": record.canonical_ts_code,
                    "authoritative_list_date": record.authoritative_list_date,
                    "evidence_package_id": record.evidence_package_id,
                    "evidence_package_hash": record.evidence_package_hash,
                }
                for record in self._records
            ],
            columns=[
                "canonical_ts_code",
                "authoritative_list_date",
                "evidence_package_id",
                "evidence_package_hash",
            ],
        )
        if frame.empty:
            return frame.astype(
                {
                    "canonical_ts_code": "string",
                    "authoritative_list_date": "string",
                    "evidence_package_id": "string",
                    "evidence_package_hash": "string",
                }
            )
        return frame

    def apply_to_stock_basic(self, stock_basic: DataFrame) -> DataFrame:
        """Overlay authoritative list dates, appending absent historical securities."""

        if not self._records:
            return stock_basic.copy()
        working = stock_basic.copy()
        if "ts_code" not in working.columns:
            working["ts_code"] = pd.Series(dtype="object")
        if "list_date" not in working.columns:
            working["list_date"] = pd.Series(dtype="object")
        working["ts_code"] = working["ts_code"].astype(str).str.strip().str.upper()
        overlay = self.frame().rename(
            columns={
                "canonical_ts_code": "ts_code",
                "authoritative_list_date": "overlay_list_date",
            }
        )
        existing = working.merge(
            overlay[["ts_code", "overlay_list_date"]],
            on="ts_code",
            how="left",
            validate="many_to_one",
        )
        existing["list_date"] = existing["overlay_list_date"].fillna(existing["list_date"])
        existing = existing.drop(columns="overlay_list_date")
        missing_codes = sorted(set(overlay["ts_code"]) - set(existing["ts_code"]))
        if missing_codes:
            additions = pd.DataFrame(
                [{column: pd.NA for column in existing.columns} for _ in missing_codes]
            )
            additions["ts_code"] = missing_codes
            dates = overlay.set_index("ts_code")["overlay_list_date"].to_dict()
            additions["list_date"] = additions["ts_code"].map(dates)
            existing = pd.concat([existing, additions], ignore_index=True)
        return existing


def publish_listing_metadata_evidence(
    *, payload: JsonObject, document: Path, reports_root: Path
) -> Path:
    """Freeze one reviewed official listing fact and its source bytes."""

    normalized = _normalize_evidence(payload, document)
    logical = {key: value for key, value in normalized.items() if key != "retrieved_at"}
    package_id = f"security_listing_metadata_evidence_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_listing_metadata_evidence" / package_id
    if output.exists():
        validate_listing_metadata_evidence(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{package_id}.staging-"))
    try:
        documents = staging / "documents"
        documents.mkdir()
        document_name = f"document{document.suffix.lower()}"
        shutil.copyfile(document, documents / document_name)
        atomic_write_json(
            staging / "evidence.json",
            {**normalized, "document_path": f"documents/{document_name}"},
        )
        hashes = {
            "evidence.json": file_sha256(staging / "evidence.json"),
            f"documents/{document_name}": file_sha256(documents / document_name),
        }
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "artifact_name": EVIDENCE_ARTIFACT_NAME,
                "package_id": package_id,
                "logical_identity": logical,
                "artifact_hashes": hashes,
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_listing_metadata_evidence(output)
    return output


def validate_listing_metadata_evidence(path: Path) -> JsonObject:
    """Validate one listing-metadata evidence package recursively."""

    manifest = _read_json(path / "manifest.json", "SECURITY_LISTING_METADATA_EVIDENCE_INVALID")
    evidence = _read_json(path / "evidence.json", "SECURITY_LISTING_METADATA_EVIDENCE_INVALID")
    if (
        manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION
        or manifest.get("artifact_name") != EVIDENCE_ARTIFACT_NAME
        or manifest.get("package_id") != path.name
    ):
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_INVALID")
    document = path / str(evidence.get("document_path", ""))
    normalized = _normalize_evidence(evidence, document)
    logical = {key: value for key, value in normalized.items() if key != "retrieved_at"}
    hashes = cast(JsonObject, manifest.get("artifact_hashes", {}))
    expected_names = {"evidence.json", str(evidence.get("document_path", ""))}
    if set(hashes) != expected_names:
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_SET_INVALID")
    for name, expected in hashes.items():
        child = (path / name).resolve()
        if path.resolve() not in child.parents or file_sha256(child) != expected:
            raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_HASH_MISMATCH")
    if manifest.get("logical_identity") != logical:
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_IDENTITY_MISMATCH")
    return {
        **evidence,
        "package_id": path.name,
        "package_hash": file_sha256(path / "manifest.json"),
    }


def publish_listing_metadata_overlay(
    *, evidence_packages: tuple[Path, ...], reports_root: Path
) -> Path:
    """Publish one immutable, typed overlay from VERIFIED evidence packages."""

    evidence = [validate_listing_metadata_evidence(path) for path in evidence_packages]
    rows = pd.DataFrame(
        [
            {
                "canonical_ts_code": item["canonical_ts_code"],
                "authoritative_list_date": item["authoritative_list_date"],
                "evidence_package_id": item["package_id"],
                "evidence_package_hash": item["package_hash"],
            }
            for item in evidence
        ]
    ).sort_values("canonical_ts_code")
    if rows["canonical_ts_code"].duplicated().any():
        raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_DUPLICATE_SECURITY")
    inventory = [
        {"package_id": item["package_id"], "package_hash": item["package_hash"]}
        for item in evidence
    ]
    logical = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "evidence_package_hashes": sorted(str(item["package_hash"]) for item in inventory),
        "records_hash": _frame_hash(rows),
    }
    overlay_id = f"security_listing_metadata_overlay_{canonical_payload_hash(logical)[:24]}"
    output = reports_root / "security_listing_metadata_overlay" / overlay_id
    if output.exists():
        validate_listing_metadata_overlay(output)
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{overlay_id}.staging-"))
    try:
        rows.to_parquet(staging / "listing_metadata.parquet", index=False)
        atomic_write_json(staging / "source_inventory.json", {"sources": inventory})
        atomic_write_json(
            staging / "summary.json", {"overlay_id": overlay_id, "records": len(rows)}
        )
        names = {"listing_metadata.parquet", "source_inventory.json", "summary.json"}
        atomic_write_json(
            staging / "manifest.json",
            {
                "schema_version": OVERLAY_SCHEMA_VERSION,
                "artifact_name": OVERLAY_ARTIFACT_NAME,
                "overlay_id": overlay_id,
                "logical_identity": logical,
                "record_count": len(rows),
                "artifact_hashes": {name: file_sha256(staging / name) for name in sorted(names)},
            },
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    validate_listing_metadata_overlay(output)
    return output


def validate_listing_metadata_overlay(path: Path) -> JsonObject:
    """Validate overlay content and every referenced evidence package."""

    manifest = _read_json(path / "manifest.json", "SECURITY_LISTING_METADATA_OVERLAY_INVALID")
    if (
        manifest.get("schema_version") != OVERLAY_SCHEMA_VERSION
        or manifest.get("artifact_name") != OVERLAY_ARTIFACT_NAME
        or manifest.get("overlay_id") != path.name
    ):
        raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_INVALID")
    expected_files = {"listing_metadata.parquet", "source_inventory.json", "summary.json"}
    hashes = cast(JsonObject, manifest.get("artifact_hashes", {}))
    if set(hashes) != expected_files:
        raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_SET_INVALID")
    for name, expected in hashes.items():
        if file_sha256(path / name) != expected:
            raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_HASH_MISMATCH")
    rows = pd.read_parquet(path / "listing_metadata.parquet")
    inventory = _read_json(
        path / "source_inventory.json", "SECURITY_LISTING_METADATA_OVERLAY_INVALID"
    ).get("sources")
    if not isinstance(inventory, list) or len(rows) != manifest.get("record_count"):
        raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_INVALID")
    reports_root = path.parent.parent
    actual_hashes: list[str] = []
    for source in inventory:
        if not isinstance(source, dict):
            raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_INVALID")
        package = validate_listing_metadata_evidence(
            reports_root / "security_listing_metadata_evidence" / str(source.get("package_id"))
        )
        if package["package_hash"] != source.get("package_hash"):
            raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_SOURCE_MISMATCH")
        actual_hashes.append(str(package["package_hash"]))
    logical = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "evidence_package_hashes": sorted(actual_hashes),
        "records_hash": _frame_hash(rows),
    }
    if (
        manifest.get("logical_identity") != logical
        or path.name != f"security_listing_metadata_overlay_{canonical_payload_hash(logical)[:24]}"
    ):
        raise DataValidationError("SECURITY_LISTING_METADATA_OVERLAY_IDENTITY_MISMATCH")
    return manifest


def _normalize_evidence(payload: JsonObject, document: Path) -> JsonObject:
    required = {
        "canonical_ts_code",
        "authoritative_list_date",
        "exchange",
        "official_source_type",
        "official_url",
        "official_document_id",
        "publication_date",
        "retrieved_at",
        "document_hash",
        "reviewed_fact",
        "evidence_status",
        "document_security_codes",
        "document_listing_dates",
    }
    if not required.issubset(payload):
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_INCOMPLETE")
    code = str(payload["canonical_ts_code"]).strip().upper()
    date = _date(str(payload["authoritative_list_date"]), "authoritative_list_date")
    if str(payload["evidence_status"]).upper() != "VERIFIED":
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_NOT_VERIFIED")
    host = (urlparse(str(payload["official_url"])).hostname or "").lower()
    if not any(host == suffix or host.endswith(f".{suffix}") for suffix in OFFICIAL_HOST_SUFFIXES):
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_SOURCE_INVALID")
    codes = {
        str(value).strip().upper()
        for value in cast(list[object], payload["document_security_codes"])
    }
    dates = {str(value) for value in cast(list[object], payload["document_listing_dates"])}
    if code not in codes or date not in dates:
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_FACT_MISMATCH")
    if not document.is_file() or file_sha256(document) != payload["document_hash"]:
        raise DataValidationError("SECURITY_LISTING_METADATA_EVIDENCE_DOCUMENT_MISMATCH")
    return {key: payload[key] for key in sorted(required)}


def _date(value: str, field: str) -> str:
    try:
        valid = datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d") == value
    except ValueError:
        valid = False
    if not valid:
        raise DataValidationError(f"SECURITY_LISTING_METADATA_INVALID_DATE: {field}")
    return value


def _frame_hash(frame: DataFrame) -> str:
    rows = frame.fillna("").astype(str).to_dict("records")
    return canonical_payload_hash(rows)


def _read_json(path: Path, error_code: str) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(error_code) from error
    if not isinstance(payload, dict):
        raise DataValidationError(error_code)
    return cast(JsonObject, payload)
