"""Validation of completed daily production publications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ashare_quant.data.exceptions import DataValidationError

REQUIRED_REPORT_ARTIFACTS = (
    "predictions.parquet",
    "ranking.csv",
    "candidates.csv",
    "daily_report.md",
    "research_summary.json",
    "explanations.json",
    "explanations.md",
    "decision.json",
    "decision_report.md",
)


def validate_production_publication(
    *,
    reports_root: Path,
    runs_root: Path,
    as_of: str,
    expected_run_id: str | None = None,
    run_manifest_path: Path | None = None,
    require_successful_run: bool = True,
) -> dict[str, Any]:
    """Return a valid production summary or raise a precise integrity error."""

    summary_path = reports_root / as_of / "production_summary.json"
    summary = _read_json(summary_path, "production summary")
    if summary.get("artifact_name") != "production_daily_summary":
        raise DataValidationError(f"invalid production summary identity: {summary_path}")
    if summary.get("as_of") != as_of:
        raise DataValidationError(
            f"production summary date mismatch: expected={as_of} actual={summary.get('as_of')}"
        )
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise DataValidationError("production summary has no run_id")
    if expected_run_id is not None and run_id != expected_run_id:
        raise DataValidationError(
            f"production summary run_id mismatch: expected={expected_run_id} actual={run_id}"
        )

    manifest_path = run_manifest_path or _find_run_manifest(runs_root, run_id)
    manifest = _read_json(manifest_path, "production run manifest")
    if manifest.get("run_id") != run_id:
        raise DataValidationError(f"run manifest identity does not match summary: {manifest_path}")
    if manifest.get("as_of") != as_of:
        raise DataValidationError(f"run manifest date does not match summary: {manifest_path}")
    if require_successful_run and manifest.get("status") != "success":
        raise DataValidationError(
            f"production run is not successful: run_id={run_id} status={manifest.get('status')}"
        )

    report_dir = reports_root / as_of
    missing = [
        str(report_dir / name)
        for name in REQUIRED_REPORT_ARTIFACTS
        if not (report_dir / name).is_file()
    ]
    observation = summary.get("observation_log_path")
    if not isinstance(observation, str) or not Path(observation).is_file():
        missing.append(str(observation or "missing observation_log_path"))
    artifact_paths = summary.get("artifacts")
    if not isinstance(artifact_paths, list) or not artifact_paths:
        raise DataValidationError("production summary has no artifact references")
    missing.extend(
        str(path) for path in artifact_paths if not isinstance(path, str) or not Path(path).exists()
    )
    if missing:
        raise DataValidationError(f"production publication has missing artifacts: {missing}")
    return summary


def _find_run_manifest(runs_root: Path, run_id: str) -> Path:
    matches = sorted(Path(runs_root).glob(f"[0-9]*/{run_id}/manifest.json"))
    if len(matches) != 1:
        raise DataValidationError(
            f"expected one run manifest for run_id={run_id}, found={len(matches)}"
        )
    return matches[0]


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"{description} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"{description} must contain a JSON object: {path}")
    return payload
