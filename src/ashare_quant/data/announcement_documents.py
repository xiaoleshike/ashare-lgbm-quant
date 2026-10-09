"""Native PDF inspection with bounded local processes; no network or OCR."""

from __future__ import annotations

import os
import re
import selectors
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ashare_quant.backtest.continuous_source import file_hash
from ashare_quant.data.announcement_http import AnnouncementError, check

PDF_PAGES_MAX = 1000
PAGE_TEXT_BYTES_MAX = 2_000_000
PARSER_TIMEOUT = 30.0


def tool_path(name: str) -> str:
    check(name in {"pdfinfo", "pdftotext"}, "PARSER_TOOL_UNSUPPORTED")
    path = shutil.which(name)
    check(path is not None, "ANNOUNCEMENT_POPPLER_REQUIRED")
    return str(path)


def run_native(
    args: list[str], *, limit: int, timeout: float = PARSER_TIMEOUT, version: bool = False
) -> bytes:
    """Bound wall time and captured output before materializing it in memory."""
    with subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:  # noqa: S603
        assert process.stdout is not None and process.stderr is not None
        result = bytearray()
        errors = bytearray()
        deadline = time.monotonic() + timeout
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, result)
                selector.register(process.stderr, selectors.EVENT_READ, errors)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    check(remaining > 0, "PARSER_TIMEOUT")
                    ready = selector.select(min(remaining, 0.2))
                    if not ready:
                        continue
                    for key, _ in ready:
                        chunk = os.read(key.fd, min(65536, limit + 1 - len(result) - len(errors)))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        key.data.extend(chunk)
                        check(len(result) + len(errors) <= limit, "PARSER_OUTPUT_LIMIT")
            check(
                process.wait(timeout=max(0.01, deadline - time.monotonic())) == 0, "PARSER_FAILED"
            )
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.wait()
            raise AnnouncementError("PARSER_TIMEOUT") from error
        except BaseException:
            process.kill()
            process.wait()
            raise
    check(version or not errors, "PARSER_STDERR")
    return bytes(result + errors if version else result)


def tool_identity(name: str) -> dict[str, str]:
    path = tool_path(name)
    version = (
        run_native([path, "-v"], limit=16384, version=True)
        .decode("utf-8", errors="strict")
        .splitlines()[0]
    )
    return {"name": name, "version": version, "binary_sha256": file_hash(Path(path))}


def inspect_pdf(data: bytes) -> dict[str, Any]:
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-4096:]:
        return {"status": "INVALID_PDF", "total_pages": None}
    with tempfile.TemporaryDirectory(prefix="announcement-pdf-") as temporary:
        path = Path(temporary) / "document.pdf"
        path.write_bytes(data)
        try:
            text = run_native([tool_path("pdfinfo"), str(path)], limit=65536).decode(
                "utf-8", errors="strict"
            )
            pages = re.search(r"^Pages:\s+(\d+)\s*$", text, re.MULTILINE)
            check(pages is not None, "PDF_PAGE_COUNT_MISSING")
            assert pages is not None
            count = int(pages[1])
            check(0 < count <= PDF_PAGES_MAX, "PDF_PAGE_LIMIT")
            check(re.search(r"^Encrypted:\s+no\b", text, re.MULTILINE) is not None, "PDF_ENCRYPTED")
            return {"status": "VALID_PDF", "total_pages": count}
        except (AnnouncementError, UnicodeError, subprocess.TimeoutExpired) as error:
            return {
                "status": "PDF_PARSE_FAILED",
                "total_pages": None,
                "error": str(error)
                if isinstance(error, AnnouncementError)
                else type(error).__name__,
            }
