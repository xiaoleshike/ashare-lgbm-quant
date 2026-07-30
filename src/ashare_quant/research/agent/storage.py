"""Immutable atomic publication for research-agent outputs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.utils.manifest import atomic_write_json


def publish_research_agent(
    *,
    output_dir: Path,
    summary_payload: dict[str, Any],
    markdown: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Publish a complete immutable directory with manifest written last."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        atomic_write_json(staging / "research_summary.json", summary_payload)
        (staging / "daily_research.md").write_text(markdown, encoding="utf-8")
        completed = {
            **manifest,
            "output_hashes": {
                "research_summary.json": file_sha256(staging / "research_summary.json"),
                "daily_research.md": file_sha256(staging / "daily_research.md"),
            },
        }
        atomic_write_json(staging / "manifest.json", completed)
        if output_dir.exists():
            raise DataValidationError(
                f"research-agent output already exists and cannot be overwritten: {output_dir}"
            )
        os.replace(staging, output_dir)
        return completed
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def read_complete_output(output_dir: Path) -> dict[str, Any] | None:
    """Return a validated manifest only for a complete output."""

    paths = {
        "summary": output_dir / "research_summary.json",
        "markdown": output_dir / "daily_research.md",
        "manifest": output_dir / "manifest.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid research-agent manifest: {error}") from error
    if not isinstance(manifest, dict):
        raise DataValidationError("research-agent manifest must be an object")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_name") != "llm_research_agent"
        or manifest.get("status") not in {"success", "success_with_fallback"}
    ):
        raise DataValidationError("invalid research-agent manifest identity")
    hashes = manifest.get("output_hashes")
    if not isinstance(hashes, dict):
        raise DataValidationError("research-agent manifest lacks output hashes")
    if hashes.get("research_summary.json") != file_sha256(paths["summary"]):
        raise DataValidationError("research-agent JSON hash mismatch")
    if hashes.get("daily_research.md") != file_sha256(paths["markdown"]):
        raise DataValidationError("research-agent Markdown hash mismatch")
    return manifest


def load_summary_payload(output_dir: Path) -> dict[str, Any]:
    """Load the already hash-validated structured summary."""

    if read_complete_output(output_dir) is None:
        raise DataValidationError(f"research-agent output is incomplete: {output_dir}")
    try:
        value = json.loads((output_dir / "research_summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid research-agent summary: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError("research-agent summary must be an object")
    return value
