from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ashare_quant.orchestration.run_manifest import (
    ProductionRun,
    create_run,
    record_failure,
    record_stage_end,
    record_stage_start,
    update_run_status,
    update_source_provenance,
)
from ashare_quant.utils.manifest import config_hash


def test_create_run_manifest(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("project_name: run-test\n", encoding="utf-8")

    run = create_run(
        "ashare-quant pipeline daily",
        config_path=config,
        runs_root=tmp_path / "runs",
        stages=("data_update", "features"),
        run_id="test-run",
    )
    manifest = read_manifest(run)

    assert run.manifest_path == run.run_dir / "manifest.json"
    assert run.run_dir.parent.name.isdigit()
    assert manifest["run_id"] == "test-run"
    assert manifest["command"] == "ashare-quant pipeline daily"
    assert manifest["status"] == "running"
    assert manifest["current_stage"] is None
    assert manifest["end_time"] is None
    assert manifest["pid"] == os.getpid()
    assert [stage["status"] for stage in manifest["stages"]] == ["pending", "pending"]


def test_update_stage_success_and_complete_run(tmp_path: Path) -> None:
    run = create_run(
        "pipeline",
        runs_root=tmp_path / "runs",
        stages=("data_update",),
        run_id="successful-run",
    )

    started = record_stage_start(run, "data_update")
    ended = record_stage_end(run, "data_update")
    completed = update_run_status(run, "success")

    assert started["current_stage"] == "data_update"
    assert started["stages"][0]["status"] == "running"
    assert ended["current_stage"] is None
    assert ended["stages"][0]["status"] == "success"
    assert ended["stages"][0]["start_time"] is not None
    assert ended["stages"][0]["end_time"] is not None
    assert completed["status"] == "success"
    assert completed["end_time"] is not None


def test_record_stage_failure_makes_run_terminal(tmp_path: Path) -> None:
    run = create_run(
        "pipeline",
        runs_root=tmp_path / "runs",
        stages=("data_update", "features"),
        run_id="failed-run",
    )
    record_stage_start(run, "data_update")

    failed = record_failure(run, RuntimeError("upstream validation failed"))

    assert failed["status"] == "failed"
    assert failed["end_time"] is not None
    assert failed["error_message"] == "RuntimeError: upstream validation failed"
    assert failed["stages"][0]["status"] == "failed"
    assert failed["stages"][0]["error_message"] == failed["error_message"]
    assert failed["stages"][1]["status"] == "pending"
    with pytest.raises(ValueError, match="terminal run status"):
        update_run_status(run, "success")


def test_record_stage_end_failure_records_stage_error(tmp_path: Path) -> None:
    run = create_run(
        "pipeline",
        runs_root=tmp_path / "runs",
        stages=("feature_build",),
        run_id="stage-end-failure",
    )
    record_stage_start(run, "feature_build")

    failed = record_stage_end(
        run,
        "feature_build",
        status="failed",
        error_message="feature validation failed",
    )

    assert failed["status"] == "failed"
    assert failed["error_message"] == "feature validation failed"
    assert failed["stages"][0]["status"] == "failed"
    assert failed["stages"][0]["error_message"] == "feature validation failed"


def test_atomic_stage_update_preserves_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = create_run(
        "pipeline",
        runs_root=tmp_path / "runs",
        stages=("data_update",),
        run_id="atomic-run",
    )
    before = run.manifest_path.read_text(encoding="utf-8")
    original_replace = Path.replace

    def fail_manifest_replace(self: Path, target: str | Path) -> Path:
        if Path(target) == run.manifest_path:
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="replace failed"):
        record_stage_start(run, "data_update")

    assert run.manifest_path.read_text(encoding="utf-8") == before


def test_run_manifest_records_reproducibility_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("project_name: provenance-test\n", encoding="utf-8")
    monkeypatch.setattr(
        "ashare_quant.orchestration.run_manifest.current_git_info",
        lambda: {"commit": "abc123", "dirty": False},
    )
    upstream = {"features_daily": {"git_commit": "features123", "row_count": 42}}
    fingerprint = {"daily": {"max_date": "20260717", "rows": 100}}

    run = create_run(
        "pipeline",
        config_path=config,
        runs_root=tmp_path / "runs",
        upstream_manifests=upstream,
        model_id="ranker-202607",
        feature_hash="feature-hash",
        data_fingerprint=fingerprint,
        run_id="provenance-run",
    )
    manifest = read_manifest(run)

    assert manifest["hostname"]
    assert manifest["start_time"]
    assert manifest["git_commit"] == "abc123"
    assert manifest["git_dirty"] is False
    assert manifest["config_hash"] == config_hash(config)
    assert manifest["source_provenance"] == {
        "upstream_manifests": upstream,
        "input_manifests": upstream,
        "resulting_manifests": {},
        "model_id": "ranker-202607",
        "feature_hash": "feature-hash",
        "data_fingerprint": fingerprint,
    }


def test_new_artifact_updates_current_provenance_and_preserves_input(tmp_path: Path) -> None:
    previous = {"git_commit": "old", "row_count": 10}
    current = {"git_commit": "new", "row_count": 11}
    run = create_run(
        "pipeline",
        runs_root=tmp_path / "runs",
        upstream_manifests={"universe_daily": previous},
        run_id="provenance-update",
    )

    updated = update_source_provenance(run, "universe_daily", current)
    provenance = updated["source_provenance"]

    assert provenance["input_manifests"]["universe_daily"] == previous
    assert provenance["upstream_manifests"]["universe_daily"] == current
    assert provenance["resulting_manifests"]["universe_daily"] == current


def read_manifest(run: ProductionRun) -> dict[str, object]:
    loaded = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
