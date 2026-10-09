"""Immutable announcement discovery/PDF evidence, separate from economic review."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.announcement_documents import inspect_pdf, tool_identity
from ashare_quant.data.announcement_http import (
    Limits,
    SafeHTTPClient,
    check,
    digest,
    request_spec,
    validate_capture,
)
from ashare_quant.data.cninfo_announcement_provider import (
    CATALOG_COLUMNS,
    AnnouncementQuery,
    CNINFOAnnouncementProvider,
    SearchResult,
    analyze_search,
)
from ashare_quant.data.economic_event_sources import publish, safe_file, validate_envelope

KIND = "official_announcement_source"
CONTRACT = "official_announcement_source_v1"

# Read compatibility v1, scoped only to CONTRACT. Entries authorize reconstruction,
# never acceptance of historical derived results. No fixture code is imported.
SOURCE_IMPLEMENTATION_COMPATIBILITY_VERSION = 1
COMPATIBLE_SOURCE_COMPONENTS = {
    "5692b05331de047ef937663449ddc08cbaab031f": {
        "announcement_sources.py": (
            "ac1fcfbfd47d131b75d547918bdd16fe3b918588c1828f95b5d2e511677b2478"
        ),
        "announcement_http.py": "c3e6173b66509dadcbb90d696d4c9b18ba09d5dd5c474230f2008682f1101f9a",
        "cninfo_announcement_provider.py": (
            "7ca9f936d6db591d7a4c1098f117bec00ef16d07a224b82874276da88b78d62f"
        ),
        "announcement_documents.py": (
            "5071c5aa5bdcb4ded36993e40edce7bd3e6c1dbd2c501e9a07477cb4c2b01ef9"
        ),
    },
}


def implementation_hash() -> str:
    return payload_hash(
        {
            name: file_hash(Path(__file__).with_name(name))
            for name in (
                "announcement_sources.py",
                "announcement_http.py",
                "cninfo_announcement_provider.py",
                "announcement_documents.py",
            )
        }
    )


def compatible_source_implementation(value: object) -> bool:
    """Allow only current bytes or explicitly pinned, reviewed v1 producer bytes."""
    return (
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        and (
            value == implementation_hash()
            or value in {payload_hash(c) for c in COMPATIBLE_SOURCE_COMPONENTS.values()}
        )
    )


def derive(
    body: dict[str, Any], blobs: dict[str, bytes]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, bytes]]:
    query = AnnouncementQuery.model_validate(body["query"])
    policy = Limits.model_validate(body["limits"])
    captures = body["captures"]
    search_count = body["search_capture_count"]
    check(type(search_count) is int and 1 <= search_count <= len(captures), "SEARCH_CAPTURE_COUNT")
    referenced = {
        a["body_sha256"] for c in captures for a in c["attempts"] if a["body_sha256"] is not None
    }
    check(set(blobs) == referenced, "RESPONSE_BLOB_SET")
    for capture in captures:
        validate_capture(capture, blobs, policy)
    progress = analyze_search(query, captures[:search_count], blobs)
    selected = body["download_ids"]
    check(
        isinstance(selected, list) and selected == sorted(set(selected)) and len(selected) <= 30,
        "DOCUMENT_SELECTION",
    )
    check(not selected or progress.status == "COMPLETE", "DISCOVERY_INCOMPLETE")
    by_id = {r["announcement_id"]: r for r in progress.records}
    check(
        set(selected) <= set(by_id)
        and search_count <= len(captures) <= search_count + len(selected),
        "DOCUMENT_REQUEST_SET",
    )
    if len(captures) < search_count + len(selected):
        check(
            len(captures) > search_count
            and captures[-1]["status"]
            in {"PERMISSION_REQUIRED", "RATE_LIMITED", "SERVER_REJECTED"},
            "UNATTEMPTED_DOCUMENT_REASON",
        )
    files = {f"responses/{h}.bin": data for h, data in blobs.items()}
    documents = []
    for index, aid in enumerate(selected):
        row = by_id[aid]
        if search_count + index >= len(captures):
            documents.append(
                {
                    "announcement_id": aid,
                    "official_document_url": row["official_document_url"],
                    "download_request_id": payload_hash(request_spec(row["official_document_url"])),
                    "document_sha256": None,
                    "document_size": None,
                    "inspection": None,
                    "fetch_status": "NOT_ATTEMPTED",
                }
            )
            continue
        capture = captures[search_count + index]
        check(
            capture["request"] == request_spec(row["official_document_url"]),
            "DOCUMENT_REQUEST_IDENTITY",
        )
        status = capture["status"]
        h, size, inspection = None, None, None
        if status == "CAPTURED":
            data = blobs[capture["attempts"][-1]["body_sha256"]]
            inspection = inspect_pdf(data)
            status = inspection["status"]
            if data.startswith(b"%PDF-"):
                h, size = digest(data), len(data)
                files[f"documents/{h}.pdf"] = data
        row.update(document_sha256=h, document_size=size, fetch_status=status)
        documents.append(
            {
                "announcement_id": aid,
                "official_document_url": row["official_document_url"],
                "download_request_id": capture["request_id"],
                "document_sha256": h,
                "document_size": size,
                "inspection": inspection,
                "fetch_status": status,
            }
        )
    statuses = {d["fetch_status"] for d in documents}
    acquisition = (
        "NOT_ATTEMPTED"
        if not selected
        else "COMPLETE"
        if statuses == {"VALID_PDF"}
        else "PERMISSION_REQUIRED"
        if "PERMISSION_REQUIRED" in statuses
        else "PARTIAL"
        if "VALID_PDF" in statuses
        else "FAILED"
    )
    summary = {
        "contract": CONTRACT,
        "discovery_status": progress.status,
        "discovery_issues": progress.issues,
        "announcement_count": len(progress.records),
        "document_selection_count": len(selected),
        "acquisition_status": acquisition,
        "unique_document_count": len(
            {d["document_sha256"] for d in documents if d["document_sha256"]}
        ),
        "extraction_status": "NOT_ATTEMPTED",
        "economic_evidence_approved": False,
        "execution_authorized": False,
        "empty_result_semantics": "QUERY_RETURNED_ZERO_NOT_GENERAL_ABSENCE_PROOF",
        "category_semantics": "PROVIDER_QUERY_FILTER_NOT_ECONOMIC_CLASSIFICATION",
    }
    frame = pd.DataFrame(progress.records, columns=CATALOG_COLUMNS).astype(
        {c: "Int64" if c == "document_size" else "string" for c in CATALOG_COLUMNS}
    )
    objects = {
        "worklist.json": {
            "query": body["query"],
            "download_ids": selected,
            "limits": body["limits"],
            "resume_unit": "COMPLETE_HTTP_REQUEST",
        },
        "requests.json": {"requests": [c["request"] for c in captures]},
        "responses.json": body,
        "document_inventory.json": {"documents": documents},
        "acquisition_summary.json": summary,
    }
    return frame, objects, files


def _identity(body: dict[str, Any], parent_hash: str | None) -> dict[str, Any]:
    return {
        "contract": CONTRACT,
        "implementation_hash": implementation_hash(),
        "acquisition_content_hash": payload_hash(body),
        "parent_manifest_hash": parent_hash,
        "pdf_inspector": tool_identity("pdfinfo") if body["download_ids"] else None,
        "economic_evidence_approved": False,
        "execution_authorized": False,
    }


def _publish(
    body: dict[str, Any], blobs: dict[str, bytes], output_root: Path, parent: Path | None
) -> Path:
    frame, objects, files = derive(body, blobs)
    files["announcement_catalog.parquet"] = frame.to_parquet(index=False)
    return publish(
        output_root,
        KIND,
        _identity(body, file_hash(parent / "manifest.json") if parent else None),
        {"parent": str(parent.resolve()) if parent else None},
        {},
        objects,
        validate_announcement_source,
        files,
    )


def publish_discovery(result: SearchResult, policy: Limits, output_root: Path) -> Path:
    body = {
        "query": result.query.model_dump(),
        "limits": policy.model_dump(),
        "captures": result.captures,
        "search_capture_count": len(result.captures),
        "download_ids": [],
    }
    return _publish(body, result.blobs, output_root, None)


def probe(query: AnnouncementQuery, client: SafeHTTPClient, output_root: Path) -> Path:
    if client.cache:
        client.cache.register_worklist(
            {"mode": "discovery", "query": query.model_dump(), "limits": client.policy.model_dump()}
        )
    return publish_discovery(
        CNINFOAnnouncementProvider(client).search(query), client.policy, output_root
    )


def _blobs(root: Path, body: dict[str, Any]) -> dict[str, bytes]:
    return {
        h: safe_file(root, f"responses/{h}.bin").read_bytes()
        for h in {
            a["body_sha256"]
            for c in body["captures"]
            for a in c["attempts"]
            if a["body_sha256"] is not None
        }
    }


def acquire(
    source: Path, announcement_ids: list[str], client: SafeHTTPClient, output_root: Path
) -> Path:
    validate_announcement_source(source)
    original = read_json(source / "responses.json")
    check(original["download_ids"] == [], "ACQUIRE_REQUIRES_DISCOVERY_SOURCE")
    check(original["limits"] == client.policy.model_dump(), "ACQUIRE_LIMITS_MISMATCH")
    body = json.loads(json.dumps(original))
    blobs = _blobs(source, body)
    progress = analyze_search(
        AnnouncementQuery.model_validate(body["query"]), body["captures"], blobs
    )
    check(progress.status == "COMPLETE", "DISCOVERY_INCOMPLETE")
    check(
        bool(announcement_ids) and len(set(announcement_ids)) == len(announcement_ids) <= 30,
        "DOCUMENT_SELECTION",
    )
    rows = {r["announcement_id"]: r for r in progress.records}
    check(set(announcement_ids) <= set(rows), "DOCUMENT_SELECTION_UNKNOWN")
    body["download_ids"] = sorted(announcement_ids)
    if client.cache:
        client.cache.register_worklist(
            {
                "mode": "documents",
                "source_manifest_hash": file_hash(source / "manifest.json"),
                "download_ids": body["download_ids"],
                "limits": client.policy.model_dump(),
            }
        )
    for aid in body["download_ids"]:
        capture, data = client.fetch(request_spec(rows[aid]["official_document_url"]))
        body["captures"].append(capture)
        blobs.update(data)
        if capture["status"] in {"PERMISSION_REQUIRED", "RATE_LIMITED", "SERVER_REJECTED"}:
            # Stop immediately: a subset publication cannot imply the full worklist completed.
            break
    return _publish(body, blobs, output_root, source)


def validate_announcement_source(path: Path) -> dict[str, Any]:
    manifest = validate_envelope(path, KIND)
    logical = manifest["logical_identity"]
    check(logical.get("contract") == CONTRACT, "ANNOUNCEMENT_SOURCE_CONTRACT")
    producer = logical.get("implementation_hash")
    check(compatible_source_implementation(producer), "SOURCE_IMPLEMENTATION_UNSUPPORTED")
    body = read_json(path / "responses.json")
    check(
        set(body) == {"query", "limits", "captures", "search_capture_count", "download_ids"},
        "SOURCE_INPUT_SCHEMA",
    )
    parent_locator = manifest["locators"].get("parent")
    parent_hash = None
    if parent_locator is not None:
        parent = Path(parent_locator)
        parent_body = read_json(parent / "responses.json")
        check(
            parent_body["download_ids"] == []
            and read_json(parent / "manifest.json")["locators"].get("parent") is None,
            "DISCOVERY_PARENT_ONLY",
        )
        validate_announcement_source(parent)
        parent_hash = file_hash(parent / "manifest.json")
        check(
            body["query"] == parent_body["query"]
            and body["limits"] == parent_body["limits"]
            and body["captures"][: body["search_capture_count"]] == parent_body["captures"]
            and body["search_capture_count"] == parent_body["search_capture_count"],
            "ACQUISITION_PARENT_MISMATCH",
        )
    else:
        check(body["download_ids"] == [], "DOCUMENT_PARENT_REQUIRED")
    expected = _identity(body, parent_hash)
    check(logical.get("pdf_inspector") == expected["pdf_inspector"], "PDF_INSPECTOR_UNSUPPORTED")
    # Retain producer provenance; all other identity and business fields are current
    # reconstructions, including the exact native inspector environment.
    expected["implementation_hash"] = producer
    check(logical == expected, "ANNOUNCEMENT_SOURCE_IDENTITY")
    frame, objects, files = derive(body, _blobs(path, body))
    check(
        set(manifest["artifact_hashes"])
        == set(objects) | set(files) | {"announcement_catalog.parquet"},
        "ANNOUNCEMENT_FILE_SET",
    )
    for name, obj in objects.items():
        check(read_json(path / name) == obj, "ANNOUNCEMENT_BUSINESS_RECONSTRUCTION")
    for name, data in files.items():
        check(safe_file(path, name).read_bytes() == data, "ANNOUNCEMENT_DOCUMENT_RECONSTRUCTION")
    actual = pd.read_parquet(path / "announcement_catalog.parquet")
    check(actual.equals(frame), "ANNOUNCEMENT_CATALOG_RECONSTRUCTION")
    return manifest
