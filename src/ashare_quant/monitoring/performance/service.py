"""Read-only performance monitoring service with atomic publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.monitoring.performance.aggregation import aggregate_performance
from ashare_quant.monitoring.performance.loader import load_observation_sources
from ashare_quant.monitoring.performance.reporting import (
    build_performance_summary,
)
from ashare_quant.monitoring.performance.schemas import (
    PerformanceBuild,
    PerformanceMonitorResult,
    PerformanceValidationResult,
)
from ashare_quant.monitoring.performance_observation.storage import (
    logical_observation_hash,
)
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info


class PerformanceMonitoringService:
    """Aggregate immutable Phase C observations without reading their raw sources."""

    def __init__(
        self,
        *,
        reports_root: Path,
        config_path: Path,
    ) -> None:
        self.reports_root = reports_root
        self.config_path = config_path

    def build(self, as_of: str) -> PerformanceBuild:
        """Validate and aggregate in memory for standalone or core integration."""

        _validate_date(as_of)
        sources = load_observation_sources(self.reports_root, as_of)
        metrics, details, warnings = aggregate_performance(
            sources.observations,
            sources.model_lineage,
        )
        summary = build_performance_summary(
            as_of=as_of,
            metrics=metrics,
            details=details,
            warnings=warnings,
        )
        git = current_git_info()
        models = [
            {
                "model_id": str(row["model_id"]),
                "model_role": str(row["model_role"]),
                "model_origin": str(row["model_origin"]),
                "horizon": int(row["horizon"]),
                "feature_hash": str(row["feature_hash"]),
                "universe_hash": str(row["universe_hash"]),
                "observation_hash": logical_observation_hash(
                    sources.observations.loc[
                        sources.observations["model_id"].astype(str).eq(str(row["model_id"]))
                        & sources.observations["model_origin"]
                        .astype(str)
                        .eq(str(row["model_origin"]))
                        & pd.to_numeric(sources.observations["horizon"], errors="coerce").eq(
                            int(row["horizon"])
                        )
                    ]
                ),
                "source_models": sources.model_lineage[str(row["model_id"])]["source_models"],
                "fusion_method": sources.model_lineage[str(row["model_id"])]["fusion_method"],
                "parent_model_id": sources.model_lineage[str(row["model_id"])]["parent_model_id"],
                "training_request_id": sources.model_lineage[str(row["model_id"])][
                    "training_request_id"
                ],
                "training_run_id": sources.model_lineage[str(row["model_id"])]["training_run_id"],
                "validation_run_id": sources.model_lineage[str(row["model_id"])][
                    "validation_run_id"
                ],
            }
            for row in metrics.to_dict("records")
        ]
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_name": "performance_monitor",
            "as_of": as_of,
            "source_observation_manifests": list(sources.source_manifests),
            "source_observation_hashes": sources.source_hashes,
            "models": models,
            "row_counts": {
                "observations": len(sources.observations),
                "available_observations": int(
                    sources.observations["label_status"].astype(str).eq("available").sum()
                ),
                "model_horizon_metrics": len(metrics),
            },
            "git_commit": git["commit"],
            "config_hash": config_hash(self.config_path),
            "metrics_generated": True,
            "labels_read": False,
            "status": "success",
            "warnings": warnings,
        }
        manifest["identity_hash"] = _identity_hash(manifest)
        return PerformanceBuild(as_of, metrics, summary, manifest)

    def run(self, as_of: str) -> PerformanceMonitorResult:
        """Build and atomically publish a standalone performance monitor."""

        built = self.build(as_of)
        output_dir = self.output_dir(as_of)
        existing = _read_complete_output(output_dir)
        if existing is not None:
            if existing.get("identity_hash") != built.manifest["identity_hash"]:
                raise DataValidationError(
                    "existing performance monitor has different immutable source identity"
                )
            return PerformanceMonitorResult(
                as_of,
                output_dir,
                len(built.metrics),
                int(built.manifest["row_counts"]["observations"]),
                idempotent=True,
            )
        if output_dir.exists():
            raise DataValidationError(f"incomplete performance monitor output exists: {output_dir}")
        publish_performance_build(output_dir, built)
        return PerformanceMonitorResult(
            as_of,
            output_dir,
            len(built.metrics),
            int(built.manifest["row_counts"]["observations"]),
        )

    def validate(self, as_of: str) -> PerformanceValidationResult:
        """Validate source observations without publishing outputs."""

        try:
            built = self.build(as_of)
        except (DataValidationError, OSError, ValueError) as error:
            return PerformanceValidationResult(as_of, False, False, 0, 0, error=str(error))
        return PerformanceValidationResult(
            as_of,
            True,
            self.output_dir(as_of).is_dir(),
            len(built.metrics),
            int(built.manifest["row_counts"]["observations"]),
            tuple(built.summary["warnings"]),
        )

    def status(self, as_of: str) -> PerformanceValidationResult:
        """Validate a published output and report its identity."""

        output_dir = self.output_dir(as_of)
        try:
            manifest = _read_complete_output(output_dir)
        except (DataValidationError, OSError, ValueError) as error:
            return PerformanceValidationResult(
                as_of,
                False,
                output_dir.exists(),
                0,
                0,
                error=str(error),
            )
        if manifest is None:
            return PerformanceValidationResult(
                as_of, False, False, 0, 0, error="performance monitor manifest is missing"
            )
        return PerformanceValidationResult(
            as_of,
            True,
            True,
            len(manifest.get("models", [])),
            int(manifest.get("row_counts", {}).get("observations", 0)),
            tuple(str(value) for value in manifest.get("warnings", [])),
        )

    def output_dir(self, as_of: str) -> Path:
        return self.reports_root / "model_monitor" / as_of / "performance"


def publish_performance_build(output_dir: Path, built: PerformanceBuild) -> None:
    """Publish a complete immutable output directory, with manifest written last."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=".performance-"))
    try:
        _write_build_files(staging, built)
        if output_dir.exists():
            raise DataValidationError(f"performance monitor output already exists: {output_dir}")
        os.replace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_performance_build(staging: Path, built: PerformanceBuild) -> None:
    """Write a performance subtree inside a larger atomic monitor staging area."""

    staging.mkdir(parents=True, exist_ok=True)
    _write_build_files(staging, built)


def _write_build_files(staging: Path, built: PerformanceBuild) -> None:
    metrics_path = staging / "performance_metrics.parquet"
    built.metrics.to_parquet(metrics_path, index=False)
    atomic_write_json(staging / "performance_summary.json", built.summary)
    completed = {
        **built.manifest,
        "metrics_file_sha256": file_sha256(metrics_path),
        "summary_file_sha256": file_sha256(staging / "performance_summary.json"),
    }
    atomic_write_json(staging / "manifest.json", completed)


def _read_complete_output(output_dir: Path) -> dict[str, Any] | None:
    paths = {
        "metrics": output_dir / "performance_metrics.parquet",
        "summary": output_dir / "performance_summary.json",
        "manifest": output_dir / "manifest.json",
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    value = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataValidationError("performance monitor manifest must be an object")
    if (
        value.get("schema_version") != 1
        or value.get("artifact_name") != "performance_monitor"
        or value.get("status") != "success"
    ):
        raise DataValidationError("invalid performance monitor manifest identity")
    if file_sha256(paths["metrics"]) != value.get("metrics_file_sha256"):
        raise DataValidationError("performance monitor metrics hash mismatch")
    if file_sha256(paths["summary"]) != value.get("summary_file_sha256"):
        raise DataValidationError("performance monitor summary hash mismatch")
    return value


def _identity_hash(manifest: dict[str, Any]) -> str:
    return canonical_payload_hash(
        {
            "schema_version": manifest["schema_version"],
            "as_of": manifest["as_of"],
            "source_observation_hashes": manifest["source_observation_hashes"],
            "models": manifest["models"],
            "config_hash": manifest["config_hash"],
            "git_commit": manifest["git_commit"],
        }
    )


def _validate_date(value: str) -> None:
    if len(value) != 8 or not value.isdigit():
        raise DataValidationError(f"performance monitor as_of must use YYYYMMDD: {value}")
