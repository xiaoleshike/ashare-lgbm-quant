"""Strict allowlist collection of immutable research reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.research.agent.schemas import CollectedArtifacts
from ashare_quant.research.agent.validation import validate_collected_artifacts

_OPTIONAL_SOURCE_GROUPS = (frozenset({"performance_metrics", "performance_manifest"}),)


def collect_artifacts(reports_root: Path, as_of: str) -> CollectedArtifacts:
    """Read only explicitly allowlisted files for one report date."""

    _validate_date(as_of)
    root = reports_root.resolve()
    paths = _allowlisted_paths(reports_root, as_of)
    payloads: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    source_paths: dict[str, str] = {}
    for name, (path, kind) in paths.items():
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise DataValidationError(f"research-agent source escapes reports root: {path}")
        if not path.is_file():
            if _is_optional_source(name):
                continue
            raise DataValidationError(f"required research-agent artifact is missing: {path}")
        hashes[name] = file_sha256(path)
        source_paths[name] = path.relative_to(reports_root).as_posix()
        if kind == "json":
            payloads[name] = _load_json(path, name)
        elif kind == "parquet":
            try:
                payloads[name] = pd.read_parquet(path)
            except (OSError, ValueError) as error:
                raise DataValidationError(f"cannot read {name}: {error}") from error
        else:
            # Markdown is untrusted and is never admitted into ResearchContext.
            payloads[name] = {"untrusted_markdown": True, "sha256": hashes[name]}
    _validate_optional_groups(payloads)
    if "performance_metrics" not in payloads:
        payloads["performance_metrics"] = pd.DataFrame(
            columns=["model_id", "model_role", "horizon", "rank_ic", "alpha_decay_ratio"]
        )
        payloads["performance_manifest"] = None
    collected = CollectedArtifacts(as_of, payloads, hashes, source_paths)
    validate_collected_artifacts(collected)
    return collected


def allowed_source_paths(reports_root: Path, as_of: str) -> tuple[Path, ...]:
    """Expose the exact path allowlist for tests and audits."""

    return tuple(path for path, _ in _allowlisted_paths(reports_root, as_of).values())


def _allowlisted_paths(reports_root: Path, as_of: str) -> dict[str, tuple[Path, str]]:
    daily = reports_root / as_of
    monitor = reports_root / "model_monitor" / as_of
    paper = reports_root / "paper_trading_daily" / as_of
    return {
        "production_summary": (daily / "production_summary.json", "json"),
        "production_manifest": (daily / "manifest.json", "json"),
        "candidates_manifest": (daily / "candidates_manifest.json", "json"),
        "decision": (daily / "decision.json", "json"),
        "decision_markdown": (daily / "decision_report.md", "markdown"),
        "explanations": (daily / "explanations.json", "json"),
        "explanations_markdown": (daily / "explanations.md", "markdown"),
        "research_summary": (daily / "research_summary.json", "json"),
        "monitor_manifest": (monitor / "manifest.json", "json"),
        "monitor_summary": (monitor / "monitor_summary.json", "json"),
        "health": (monitor / "health.json", "json"),
        "alerts": (monitor / "alerts" / "alerts.json", "json"),
        "alerts_manifest": (monitor / "alerts" / "manifest.json", "json"),
        "performance_metrics": (
            monitor / "performance" / "performance_metrics.parquet",
            "parquet",
        ),
        "performance_manifest": (monitor / "performance" / "manifest.json", "json"),
        "portfolio_metrics": (monitor / "portfolio_metrics.parquet", "parquet"),
        "paper_summary": (paper / "summary.json", "json"),
        "paper_report": (paper / "report.md", "markdown"),
    }


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {name}: {error}") from error
    if not isinstance(value, dict):
        raise DataValidationError(f"{name} must contain a JSON object")
    return value


def _validate_date(value: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"research-agent as_of must use YYYYMMDD: {value}")


def _is_optional_source(name: str) -> bool:
    return any(name in group for group in _OPTIONAL_SOURCE_GROUPS)


def _validate_optional_groups(payloads: dict[str, Any]) -> None:
    for group in _OPTIONAL_SOURCE_GROUPS:
        present = group & payloads.keys()
        if present and present != group:
            missing = sorted(group - present)
            raise DataValidationError(
                f"optional research-agent source group is incomplete: missing={missing}"
            )
