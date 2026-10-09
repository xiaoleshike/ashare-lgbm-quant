"""Immutable page-level native text, independently derived from frozen PDF sources."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ashare_quant.backtest.continuous_source import file_hash, payload_hash, read_json
from ashare_quant.data.announcement_documents import (
    PAGE_TEXT_BYTES_MAX,
    inspect_pdf,
    run_native,
    tool_identity,
    tool_path,
)
from ashare_quant.data.announcement_http import AnnouncementError, check, digest
from ashare_quant.data.economic_event_sources import publish, safe_file, validate_envelope

KIND = "official_announcement_text"
CONTRACT = "official_announcement_text_extraction_v1"


def parser_identity() -> dict[str, Any]:
    return {
        "parser": "poppler_pdftotext",
        "tools": [tool_identity("pdfinfo"), tool_identity("pdftotext")],
        "config": {
            "layout": True,
            "encoding": "UTF-8",
            "ocr": False,
            "page_text_bytes_max": PAGE_TEXT_BYTES_MAX,
        },
        "implementation_hash": payload_hash(
            {
                name: file_hash(Path(__file__).with_name(name))
                for name in ("announcement_documents.py", "announcement_text_extraction.py")
            }
        ),
    }


def page_text(path: Path, page: int) -> str:
    return run_native(
        [
            tool_path("pdftotext"),
            "-f",
            str(page),
            "-l",
            str(page),
            "-layout",
            "-enc",
            "UTF-8",
            str(path),
            "-",
        ],
        limit=PAGE_TEXT_BYTES_MAX,
    ).decode("utf-8", errors="strict")


def extract_pdf(data: bytes, parser: dict[str, Any]) -> dict[str, Any]:
    inspection = inspect_pdf(data)
    pages: list[dict[str, Any]] = []
    if inspection["status"] == "VALID_PDF":
        with tempfile.TemporaryDirectory(prefix="announcement-text-") as temporary:
            path = Path(temporary) / "document.pdf"
            path.write_bytes(data)
            for number in range(1, inspection["total_pages"] + 1):
                text, error = "", None
                try:
                    text = page_text(path, number)
                    visible = [c for c in text if not c.isspace()]
                    status = (
                        "COMPLETE"
                        if visible
                        and any(c.isalnum() for c in visible)
                        and visible.count("\ufffd") / len(visible) < 0.05
                        else "SCANNED_OR_NO_TEXT"
                    )
                except (AnnouncementError, UnicodeError) as exc:
                    status = "PAGE_PARSE_FAILED"
                    error = str(exc) if isinstance(exc, AnnouncementError) else type(exc).__name__
                pages.append(
                    {
                        "page_number": number,
                        "text": text,
                        "text_sha256": digest(text.encode("utf-8")),
                        "char_count": len(text),
                        "extraction_status": status,
                        "layout_status": "NATIVE_LAYOUT_UNVERIFIED",
                        "error": error,
                    }
                )
    statuses = {p["extraction_status"] for p in pages}
    status = (
        "FAILED"
        if not pages
        else "COMPLETE"
        if statuses == {"COMPLETE"}
        else "SCANNED_OR_NO_TEXT"
        if statuses == {"SCANNED_OR_NO_TEXT"}
        else "PARTIAL"
    )
    return {
        "schema": CONTRACT,
        "document_sha256": digest(data),
        "parser_identity": parser,
        "extraction_status": status,
        "inspection": inspection,
        "total_pages": inspection["total_pages"],
        "extracted_pages": sum(p["extraction_status"] == "COMPLETE" for p in pages),
        "pages": pages,
        "economic_evidence_approved": False,
        "limitations": (
            "Native layout text only. Reading order, columns, tables and headers are not "
            "semantically reconstructed. No OCR or economic interpretation."
        ),
    }


def full_text(extraction: dict[str, Any]) -> bytes:
    return "\n".join(
        f"===== PAGE {p['page_number']} =====\n{p['text']}" for p in extraction["pages"]
    ).encode("utf-8")


def _derive(source: Path, parser: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    inventory = read_json(source / "document_inventory.json")["documents"]
    objects, files = {}, {}
    for h in sorted({d["document_sha256"] for d in inventory if d["document_sha256"]}):
        data = safe_file(source, f"documents/{h}.pdf").read_bytes()
        check(digest(data) == h, "TEXT_SOURCE_HASH")
        extraction = extract_pdf(data, parser)
        objects[f"extracted_text/{h}/extraction.json"] = extraction
        files[f"extracted_text/{h}/full_text.txt"] = full_text(extraction)
    return objects, files


def publish_text(source: Path, output_root: Path) -> Path:
    from ashare_quant.data.announcement_sources import validate_announcement_source

    validate_announcement_source(source)
    parser = parser_identity()
    objects, files = _derive(source, parser)
    logical = {
        "contract": CONTRACT,
        "source_manifest_hash": file_hash(source / "manifest.json"),
        "parser_identity": parser,
        "derived_content_hash": payload_hash(objects),
        "economic_evidence_approved": False,
    }
    return publish(
        output_root,
        KIND,
        logical,
        {"source": str(source.resolve())},
        {},
        objects,
        validate_text_artifact,
        files,
    )


def validate_text_artifact(path: Path) -> dict[str, Any]:
    from ashare_quant.data.announcement_sources import validate_announcement_source

    manifest = validate_envelope(path, KIND)
    logical = manifest["logical_identity"]
    source = Path(manifest["locators"]["source"])
    check(
        file_hash(source / "manifest.json") == logical["source_manifest_hash"], "TEXT_PARENT_HASH"
    )
    validate_announcement_source(source)
    check(logical["parser_identity"] == parser_identity(), "PARSER_ENVIRONMENT_UNSUPPORTED")
    objects, files = _derive(source, logical["parser_identity"])
    expected = {
        "contract": CONTRACT,
        "source_manifest_hash": file_hash(source / "manifest.json"),
        "parser_identity": parser_identity(),
        "derived_content_hash": payload_hash(objects),
        "economic_evidence_approved": False,
    }
    check(logical == expected, "TEXT_IDENTITY")
    check(set(manifest["artifact_hashes"]) == set(objects) | set(files), "TEXT_FILE_SET")
    for name, obj in objects.items():
        check(read_json(path / name) == obj, "TEXT_BUSINESS_RECONSTRUCTION")
    for name, data in files.items():
        check(safe_file(path, name).read_bytes() == data, "TEXT_BUSINESS_RECONSTRUCTION")
    return manifest
