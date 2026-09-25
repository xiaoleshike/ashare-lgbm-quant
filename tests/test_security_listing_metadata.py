from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.security_listing_metadata import (
    SecurityListingMetadataResolver,
    publish_listing_metadata_evidence,
    publish_listing_metadata_overlay,
    validate_listing_metadata_evidence,
    validate_listing_metadata_overlay,
)


def test_verified_listing_metadata_overlay_is_immutable_and_applied(tmp_path: Path) -> None:
    document = tmp_path / "listing.html"
    document.write_text("300114 中航电测 上市日期 20100827", encoding="utf-8")
    evidence = publish_listing_metadata_evidence(
        payload=_payload(document), document=document, reports_root=tmp_path / "reports"
    )
    overlay = publish_listing_metadata_overlay(
        evidence_packages=(evidence,), reports_root=tmp_path / "reports"
    )

    validate_listing_metadata_evidence(evidence)
    validate_listing_metadata_overlay(overlay)
    resolver = SecurityListingMetadataResolver.from_path(overlay)
    stock = pd.DataFrame([{"ts_code": "000001.SZ", "list_date": "19910403", "name": "fixture"}])
    applied = resolver.apply_to_stock_basic(stock)

    added = applied.loc[applied["ts_code"].eq("300114.SZ")].iloc[0]
    assert added["list_date"] == "20100827"
    assert resolver.record_count == 1


def test_listing_metadata_overlay_overrides_incomplete_stock_basic(tmp_path: Path) -> None:
    document = tmp_path / "listing.html"
    document.write_text("300114 中航电测 上市日期 20100827", encoding="utf-8")
    evidence = publish_listing_metadata_evidence(
        payload=_payload(document), document=document, reports_root=tmp_path / "reports"
    )
    overlay = publish_listing_metadata_overlay(
        evidence_packages=(evidence,), reports_root=tmp_path / "reports"
    )
    resolver = SecurityListingMetadataResolver.from_path(overlay)

    applied = resolver.apply_to_stock_basic(
        pd.DataFrame([{"ts_code": "300114.SZ", "list_date": None, "name": "中航电测"}])
    )

    assert applied.loc[0, "list_date"] == "20100827"


def test_listing_metadata_evidence_tamper_fails_closed(tmp_path: Path) -> None:
    document = tmp_path / "listing.html"
    document.write_text("300114 中航电测 上市日期 20100827", encoding="utf-8")
    evidence = publish_listing_metadata_evidence(
        payload=_payload(document), document=document, reports_root=tmp_path / "reports"
    )
    frozen = next((evidence / "documents").iterdir())
    frozen.write_text("tampered", encoding="utf-8")

    with pytest.raises(DataValidationError, match="DOCUMENT_MISMATCH|HASH_MISMATCH"):
        validate_listing_metadata_evidence(evidence)


def test_nonofficial_listing_metadata_source_is_rejected(tmp_path: Path) -> None:
    document = tmp_path / "listing.html"
    document.write_text("300114 中航电测 上市日期 20100827", encoding="utf-8")
    payload = _payload(document)
    payload["official_url"] = "https://example.com/listing"

    with pytest.raises(DataValidationError, match="SOURCE_INVALID"):
        publish_listing_metadata_evidence(
            payload=payload, document=document, reports_root=tmp_path / "reports"
        )


def _payload(document: Path) -> dict[str, object]:
    return {
        "canonical_ts_code": "300114.SZ",
        "authoritative_list_date": "20100827",
        "exchange": "SZSE",
        "official_source_type": "SZSE_LISTING_ANNOUNCEMENT",
        "official_url": "https://www.szse.cn/aboutus/trends/news/example.html",
        "official_document_id": "SZSE-300114-20100827",
        "publication_date": "20100827",
        "retrieved_at": "2026-09-25T00:00:00Z",
        "document_hash": hashlib.sha256(document.read_bytes()).hexdigest(),
        "reviewed_fact": "300114.SZ listed on 20100827",
        "evidence_status": "VERIFIED",
        "document_security_codes": ["300114.SZ"],
        "document_listing_dates": ["20100827"],
    }
