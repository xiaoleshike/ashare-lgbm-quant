from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.orchestration.daily import (
    DailyPipelineOrchestrator,
    DailyPipelineResult,
    daily_pipeline_stages,
    resolve_completed_trading_date,
    stage_result,
)
from ashare_quant.orchestration.freshness import GateResult
from ashare_quant.orchestration.lock import ProductionLockError, production_lock
from ashare_quant.orchestration.run_manifest import ProductionRun
from ashare_quant.utils.manifest import atomic_write_json

EXPECTED_STAGES = [
    "data_update",
    "data_validate",
    "raw_freshness_gate",
    "universe_build",
    "universe_validate",
    "universe_readiness_gate",
    "features_build",
    "features_validate",
    "features_readiness_gate",
]


def test_successful_daily_pipeline_records_manifest_and_stage_order(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    processed = tmp_path / "processed"
    atomic_write_json(
        processed / "universe_daily" / "_manifest.json",
        {"artifact_name": "universe_daily", "git_commit": "before-run"},
    )
    orchestrator = make_orchestrator(
        tmp_path,
        executor=lambda arguments: calls.append(arguments) or 0,
        processed_root=processed,
    )

    result = orchestrator.run("20240105")
    manifest = read_manifest(result.run.manifest_path)

    assert result.status == "success"
    assert result.exit_code == 0
    assert [stage.name for stage in daily_pipeline_stages("20240105")] == EXPECTED_STAGES
    assert [stage["name"] for stage in manifest["stages"]] == EXPECTED_STAGES
    assert [stage["status"] for stage in manifest["stages"]] == ["success"] * 9
    assert all(stage["result"] is not None for stage in manifest["stages"])
    assert all(stage["elapsed_seconds"] is not None for stage in manifest["stages"])
    assert manifest["status"] == "success"
    assert manifest["elapsed_seconds"] is not None
    assert "universe_daily" in manifest["source_provenance"]["upstream_manifests"]
    assert len(calls) == 6
    assert [arguments[:2] for arguments in calls] == [
        ("data", "update"),
        ("data", "validate"),
        ("universe", "build"),
        ("universe", "validate"),
        ("features", "build"),
        ("features", "validate"),
    ]
    assert calls[0] == (
        "data",
        "update",
        "--repair-gaps",
        "--end-date",
        "20240105",
    )


def test_failed_stage_stops_daily_pipeline_immediately(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    def executor(arguments: tuple[str, ...]) -> int:
        calls.append(arguments)
        return 1 if arguments[:2] == ("universe", "build") else 0

    result = make_orchestrator(tmp_path, executor=executor).run("20240105")
    manifest = read_manifest(result.run.manifest_path)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.failed_stage == "universe_build"
    assert len(calls) == 3
    assert [stage["status"] for stage in manifest["stages"]] == [
        "success",
        "success",
        "success",
        "failed",
        "pending",
        "pending",
        "pending",
        "pending",
        "pending",
    ]
    assert manifest["status"] == "failed"
    assert manifest["error_message"] == "stage universe_build returned exit code 1"
    assert manifest["stages"][3]["result"]["exit_code"] == 1


def test_failed_readiness_gate_stops_pipeline_and_records_structured_result(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []

    def readiness(gate_name: str, as_of: str) -> GateResult:
        if gate_name == "raw_freshness_gate":
            return GateResult(
                gate_name,
                as_of,
                ("daily is stale",),
                ("stock_basic snapshot is old",),
                {"actual_max_dates": {"daily": "20240104"}, "row_counts": {"daily": 0}},
            )
        return successful_gate(gate_name, as_of)

    orchestrator = make_orchestrator(
        tmp_path,
        executor=lambda arguments: calls.append(arguments) or 0,
        readiness_executor=readiness,
    )

    result = orchestrator.run("20240105")
    manifest = read_manifest(result.run.manifest_path)
    gate_stage = manifest["stages"][2]

    assert result.failed_stage == "raw_freshness_gate"
    assert len(calls) == 2
    assert gate_stage["status"] == "failed"
    assert gate_stage["result"]["expected_as_of"] == "20240105"
    assert gate_stage["result"]["actual_max_dates"] == {"daily": "20240104"}
    assert gate_stage["result"]["hard_failures"] == ["daily is stale"]


def test_daily_pipeline_uses_production_lock(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []
    orchestrator = make_orchestrator(
        tmp_path,
        executor=lambda arguments: calls.append(arguments) or 0,
    )
    lock_path = tmp_path / "runs" / ".production.lock"

    with production_lock(lock_path, command="active production job"):
        with pytest.raises(ProductionLockError, match="active production job"):
            orchestrator.run("20240105")

    assert calls == []


def test_repeated_as_of_runs_create_distinct_run_ids(tmp_path: Path) -> None:
    orchestrator = make_orchestrator(tmp_path, executor=lambda arguments: 0)

    first = orchestrator.run("20240105")
    second = orchestrator.run("20240105")

    assert first.run.run_id != second.run.run_id
    assert first.run.manifest_path.exists()
    assert second.run.manifest_path.exists()


def test_build_stage_result_records_incremental_and_canonical_artifact(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    artifact_manifest = {
        "artifact_name": "features_daily",
        "build_scope": {"rows_written_or_replaced": 10, "partitions_changed": 1},
        "canonical_artifact": {"row_count": 1000, "partition_count": 12},
    }
    atomic_write_json(processed / "features_daily" / "_manifest.json", artifact_manifest)

    result = stage_result(
        next(
            stage for stage in daily_pipeline_stages("20240105") if stage.name == "features_build"
        ),
        0,
        "20240105",
        processed,
    )

    assert result["incremental_build"] == artifact_manifest["build_scope"]
    assert result["canonical_artifact"] == artifact_manifest["canonical_artifact"]


def test_successful_build_stages_publish_new_resulting_provenance(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    previous_universe = {"artifact_name": "universe_daily", "row_count": 10}
    previous_features = {"artifact_name": "features_daily", "row_count": 10}
    current_universe = {"artifact_name": "universe_daily", "row_count": 11}
    current_features = {"artifact_name": "features_daily", "row_count": 11}
    atomic_write_json(processed / "universe_daily" / "_manifest.json", previous_universe)
    atomic_write_json(processed / "features_daily" / "_manifest.json", previous_features)

    def executor(arguments: tuple[str, ...]) -> int:
        if arguments[:2] == ("universe", "build"):
            atomic_write_json(processed / "universe_daily" / "_manifest.json", current_universe)
        if arguments[:2] == ("features", "build"):
            atomic_write_json(processed / "features_daily" / "_manifest.json", current_features)
        return 0

    result = make_orchestrator(tmp_path, executor=executor, processed_root=processed).run(
        "20240105"
    )
    provenance = read_manifest(result.run.manifest_path)["source_provenance"]

    assert provenance["input_manifests"]["universe_daily"] == previous_universe
    assert provenance["input_manifests"]["features_daily"] == previous_features
    assert provenance["upstream_manifests"]["universe_daily"] == current_universe
    assert provenance["upstream_manifests"]["features_daily"] == current_features
    assert provenance["resulting_manifests"]["universe_daily"] == current_universe
    assert provenance["resulting_manifests"]["features_daily"] == current_features


def test_default_as_of_is_resolved_after_data_update(tmp_path: Path) -> None:
    events: list[str] = []

    def executor(arguments: tuple[str, ...]) -> int:
        events.append(" ".join(arguments))
        return 0

    def resolver(requested: str | None) -> str:
        assert requested is None
        assert events == ["data update --repair-gaps"]
        return "20240105"

    config = tmp_path / "config.yaml"
    config.write_text("project_name: pipeline-test\n", encoding="utf-8")
    orchestrator = DailyPipelineOrchestrator(
        executor=executor,
        as_of_resolver=resolver,
        config_path=config,
        processed_root=tmp_path / "processed",
        readiness_executor=successful_gate,
        runs_root=tmp_path / "runs",
        lock_path=tmp_path / "runs" / ".production.lock",
    )

    result = orchestrator.run()

    assert result.status == "success"
    assert result.as_of == "20240105"
    assert events[0] == "data update --repair-gaps"
    assert events[1] == "data validate"
    assert events[2] == "universe build --start-date 20240105 --end-date 20240105"


def test_latest_completed_trading_date_respects_market_close(tmp_path: Path) -> None:
    store = ParquetDataStore(tmp_path / "raw")
    store.write(
        get_dataset_spec("trade_cal"),
        pd.DataFrame(
            {
                "exchange": ["SSE"] * 3,
                "cal_date": ["20240103", "20240104", "20240105"],
                "is_open": [1, 1, 1],
            }
        ),
    )
    timezone = ZoneInfo("Asia/Shanghai")

    before_close = resolve_completed_trading_date(
        store,
        None,
        now=datetime(2024, 1, 5, 14, 59, tzinfo=timezone),
    )
    after_close = resolve_completed_trading_date(
        store,
        None,
        now=datetime(2024, 1, 5, 15, 1, tzinfo=timezone),
    )

    assert before_close == "20240104"
    assert after_close == "20240105"


def test_pipeline_daily_cli_dispatches_requested_as_of(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, str | None] = {}
    run = ProductionRun("cli-run", tmp_path / "runs" / "20240105" / "cli-run")

    class FakeOrchestrator:
        def __init__(self, **kwargs) -> None:
            observed["configured"] = "yes"

        def run(self, requested_as_of: str | None) -> DailyPipelineResult:
            observed["as_of"] = requested_as_of
            return DailyPipelineResult(run, "success", 0, requested_as_of)

    monkeypatch.setattr("ashare_quant.cli.DailyPipelineOrchestrator", FakeOrchestrator)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "pipeline",
            "daily",
            "--as-of",
            "20240105",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert observed == {"configured": "yes", "as_of": "20240105"}
    assert "pipeline_daily: status=success run_id=cli-run as_of=20240105" in captured.out


@pytest.mark.parametrize(("ready", "expected_exit"), [(True, 0), (False, 1)])
def test_pipeline_readiness_cli_returns_readiness_exit_code(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    ready: bool,
    expected_exit: int,
) -> None:
    class FakeFreshnessService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def check_all(self, as_of: str) -> tuple[GateResult, ...]:
            failures = () if ready else ("daily stale",)
            return (GateResult("raw_freshness_gate", as_of, failures, (), {}),)

    monkeypatch.setattr("ashare_quant.cli.FreshnessService", FakeFreshnessService)
    monkeypatch.setattr(
        "ashare_quant.cli.resolve_completed_trading_date",
        lambda store, requested: requested,
    )

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "pipeline",
            "readiness",
            "--as-of",
            "20240105",
        ]
    )

    assert exit_code == expected_exit
    assert ("READY" if ready else "NOT_READY") in capsys.readouterr().out


def make_orchestrator(
    tmp_path: Path,
    *,
    executor,
    processed_root: Path | None = None,
    readiness_executor=None,
) -> DailyPipelineOrchestrator:
    config = tmp_path / "config.yaml"
    config.write_text("project_name: pipeline-test\n", encoding="utf-8")
    return DailyPipelineOrchestrator(
        executor=executor,
        as_of_resolver=lambda requested: requested or "20240105",
        config_path=config,
        processed_root=processed_root or tmp_path / "processed",
        readiness_executor=readiness_executor or successful_gate,
        runs_root=tmp_path / "runs",
        lock_path=tmp_path / "runs" / ".production.lock",
    )


def successful_gate(gate_name: str, as_of: str) -> GateResult:
    return GateResult(gate_name, as_of, (), (), {"row_counts": {}})


def read_manifest(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded
