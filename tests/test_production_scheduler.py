from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.datasets import ALL_DATASETS, get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.orchestration.production import ProductionPipelineResult
from ashare_quant.orchestration.publication import REQUIRED_REPORT_ARTIFACTS
from ashare_quant.orchestration.run_manifest import (
    create_run,
    record_failure,
    update_run_status,
)
from ashare_quant.orchestration.scheduler import (
    FullDataUpdateScheduler,
    ProductionScheduler,
    resolve_automatic_production_date,
)
from ashare_quant.utils.manifest import atomic_write_json


def test_automatic_date_resolves_today_after_ready_time(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    settings = AppSettings()

    decision = resolve_automatic_production_date(
        store,
        settings,
        now=datetime(2026, 7, 24, 11, 0, tzinfo=UTC),  # 19:00 Asia/Shanghai
    )

    assert decision.status == "run"
    assert decision.resolved_as_of == "20260724"


def test_automatic_date_does_not_run_before_market_data_ready(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))

    decision = resolve_automatic_production_date(
        store,
        AppSettings(),
        now=datetime(2026, 7, 24, 10, 29, tzinfo=UTC),  # 18:29 Asia/Shanghai
    )

    assert decision.status == "skipped"
    assert decision.resolved_as_of == "20260724"
    assert decision.reason == "market_data_not_ready"


def test_non_trading_day_skips_without_falling_back(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    pipeline = FakePipeline(tmp_path)
    scheduler = make_scheduler(
        tmp_path,
        store,
        pipeline,
        now=datetime(2026, 7, 25, 12, 0, tzinfo=UTC),
    )

    result = scheduler.run(None)

    assert result.status == "skipped"
    assert result.resolved_as_of is None
    assert result.skipped_reason == "non_trading_day"
    assert pipeline.calls == []
    invocation = read_json(result.invocation_manifest)
    assert invocation["skipped"] is True
    assert invocation["pipeline_run_id"] is None


def test_explicit_as_of_preserves_requested_date_without_auto_replacement(
    tmp_path: Path,
) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    pipeline = FakePipeline(tmp_path)
    scheduler = make_scheduler(
        tmp_path,
        store,
        pipeline,
        now=datetime(2026, 7, 25, 1, 0, tzinfo=UTC),
    )

    result = scheduler.run("20260724", dry_run=True)

    assert result.status == "success"
    assert result.resolved_as_of == "20260724"
    assert pipeline.calls == [("20260724", True)]


def test_existing_valid_publication_is_skipped(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    publish_completed_run(tmp_path, "20260724", "existing-run", status="success")
    pipeline = FakePipeline(tmp_path)
    scheduler = make_scheduler(
        tmp_path,
        store,
        pipeline,
        now=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
    )

    result = scheduler.run(None)

    assert result.status == "skipped"
    assert result.skipped_reason == "already_successful"
    assert pipeline.calls == []


def test_failed_summary_is_not_treated_as_success(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    publish_completed_run(tmp_path, "20260724", "failed-run", status="failed")
    pipeline = FakePipeline(tmp_path)
    scheduler = make_scheduler(
        tmp_path,
        store,
        pipeline,
        now=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
    )

    result = scheduler.run(None)

    assert result.status == "success"
    assert pipeline.calls == [("20260724", False)]


def test_scheduler_manifest_records_systemd_context(tmp_path: Path, monkeypatch) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    pipeline = FakePipeline(tmp_path)
    scheduler = make_scheduler(
        tmp_path,
        store,
        pipeline,
        now=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
    )
    monkeypatch.setenv("INVOCATION_ID", "systemd-execution-id")

    result = scheduler.run(None, invocation_source="systemd_timer")
    invocation = read_json(result.invocation_manifest)
    run_manifest = read_json(pipeline.last_run.manifest_path)

    assert invocation["invocation_source"] == "systemd_timer"
    assert invocation["resolved_as_of"] == "20260724"
    assert invocation["service_execution_id"] == "systemd-execution-id"
    assert invocation["pipeline_run_id"] == pipeline.last_run.run_id
    assert run_manifest["invocation_source"] == "systemd_timer"
    assert run_manifest["scheduler_invocation_id"] == invocation["invocation_id"]
    assert run_manifest["resolved_as_of"] == "20260724"
    assert run_manifest["timezone"] == "Asia/Shanghai"


def test_failed_pipeline_returns_nonzero_and_preserves_failure_manifest(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    pipeline = FakePipeline(tmp_path, fail=True)
    scheduler = make_scheduler(
        tmp_path,
        store,
        pipeline,
        now=datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
    )

    result = scheduler.run("20260724")

    assert result.status == "failed"
    assert result.exit_code != 0
    assert read_json(result.invocation_manifest)["status"] == "failed"
    assert not (tmp_path / "reports" / "20260724" / "production_summary.json").exists()


def test_transient_pipeline_failure_has_bounded_audited_retries(tmp_path: Path) -> None:
    store = calendar_store(tmp_path, ("20260724",))
    pipeline = FakePipeline(tmp_path, fail=True)
    settings = AppSettings.model_validate(
        {
            "production": {
                "scheduler": {
                    "max_pipeline_attempts": 2,
                    "retry_backoff_seconds": 0,
                }
            }
        }
    )
    scheduler = ProductionScheduler(
        settings=settings,
        raw_store=store,
        pipeline=pipeline,
        reports_root=tmp_path / "reports",
        runs_root=tmp_path / "runs",
        now=lambda: datetime(2026, 7, 24, 12, 0, tzinfo=UTC),
        sleeper=lambda _: None,
    )

    result = scheduler.run("20260724")
    invocation = read_json(result.invocation_manifest)

    assert result.status == "failed"
    assert len(pipeline.calls) == 2
    assert len(invocation["attempts"]) == 2
    run_ids = [attempt["pipeline_run_id"] for attempt in invocation["attempts"]]
    assert len(set(run_ids)) == 2


def test_weekly_full_update_resolves_trade_cal_and_uses_all_update_controls(
    tmp_path: Path, monkeypatch
) -> None:
    store = calendar_store(tmp_path, ("20260727", "20260728", "20260729"))
    calls: list[tuple[tuple[str, ...], str, bool, bool]] = []

    class FakeIngestion:
        def __init__(self, settings, raw_store) -> None:
            pass

        def update(
            self,
            datasets: tuple[str, ...],
            end_date: str,
            refresh_snapshots: bool,
            repair_gaps: bool,
        ) -> list[object]:
            calls.append((datasets, end_date, refresh_snapshots, repair_gaps))
            return []

    class FakeValidator:
        def __init__(self, raw_store) -> None:
            pass

        def validate_all(self, datasets: tuple[str, ...]) -> list[object]:
            assert datasets == ALL_DATASETS
            return []

    monkeypatch.setattr(
        "ashare_quant.orchestration.scheduler.DataIngestionService",
        FakeIngestion,
    )
    monkeypatch.setattr(
        "ashare_quant.orchestration.scheduler.DataValidator",
        FakeValidator,
    )
    config = tmp_path / "config.yaml"
    config.write_text("project_name: scheduler-test\n", encoding="utf-8")
    updater = FullDataUpdateScheduler(
        settings=AppSettings(),
        config_path=config,
        raw_store=store,
        runs_root=tmp_path / "runs",
        lock_path=tmp_path / "runs" / ".production.lock",
        now=lambda: datetime(2026, 7, 29, 4, 0, tzinfo=UTC),  # Wednesday noon CST
    )

    result = updater.run(invocation_source="systemd_timer")

    assert result.status == "success"
    assert result.resolved_as_of == "20260728"
    assert calls == [(ALL_DATASETS, "20260728", True, True)]


def test_systemd_units_do_not_construct_dates_or_invoke_forbidden_workflows() -> None:
    production_service = Path("deploy/systemd/ashare-quant-production.service").read_text(
        encoding="utf-8"
    )
    production_timer = Path("deploy/systemd/ashare-quant-production.timer").read_text(
        encoding="utf-8"
    )
    full_update_service = Path("deploy/systemd/ashare-quant-full-update.service").read_text(
        encoding="utf-8"
    )
    full_update_timer = Path("deploy/systemd/ashare-quant-full-update.timer").read_text(
        encoding="utf-8"
    )
    combined = production_service + full_update_service

    assert "Type=oneshot" in production_service
    assert "User=@RUN_USER@" in production_service
    assert "pipeline production" in production_service
    assert "pipeline full-update" in full_update_service
    assert "--as-of" not in production_service
    assert "$(date" not in combined
    assert " date " not in combined
    assert "train" not in combined.lower()
    assert "promote" not in combined.lower()
    assert "paper" not in combined.lower()
    assert "order" not in combined.lower()
    assert "Mon..Fri *-*-* 19:30:00 Asia/Shanghai" in production_timer
    assert "Wed,Sun *-*-* 12:00:00 Asia/Shanghai" in full_update_timer


def test_scheduler_source_has_no_training_promotion_or_trading_calls() -> None:
    source = Path("src/ashare_quant/orchestration/scheduler.py").read_text(encoding="utf-8")

    assert ".train(" not in source
    assert ".promote_model(" not in source
    assert "paper_trading" not in source
    assert "create_order" not in source


def test_environment_example_contains_no_secret() -> None:
    content = Path("deploy/systemd/ashare-quant.env.example").read_text(encoding="utf-8")

    assert content == "TUSHARE_TOKEN=REPLACE_ME\nPYTHONUNBUFFERED=1\n"


def test_install_dry_run_does_not_touch_systemd(tmp_path: Path) -> None:
    env_file = tmp_path / "ashare-quant.env"
    env_file.write_text("TUSHARE_TOKEN=test-placeholder\n", encoding="utf-8")
    before = {
        path: path.stat().st_mtime_ns for path in Path("/etc/systemd/system").glob("ashare-quant-*")
    }

    result = subprocess.run(  # noqa: S603 - fixed repository script with test-owned arguments.
        [
            str(Path.cwd() / "scripts/install_production_timer.sh"),
            "--project-dir",
            str(Path.cwd()),
            "--user",
            os.environ.get("USER", str(os.getuid())),
            "--venv",
            str(Path.cwd() / ".venv"),
            "--env-file",
            str(env_file),
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    after = {
        path: path.stat().st_mtime_ns for path in Path("/etc/systemd/system").glob("ashare-quant-*")
    }

    assert result.returncode == 0, result.stderr
    assert "Dry run: no files or systemd state will be modified." in result.stdout
    assert before == after


class FakePipeline:
    def __init__(self, root: Path, *, fail: bool = False) -> None:
        self.root = root
        self.fail = fail
        self.calls: list[tuple[str, bool]] = []
        self.last_run = None

    def run(
        self,
        as_of: str,
        *,
        dry_run: bool = False,
        invocation_source: str = "manual_cli",
        scheduler_trigger_time: str | None = None,
        scheduler_invocation_id: str | None = None,
        service_execution_id: str | None = None,
        timezone: str | None = None,
    ) -> ProductionPipelineResult:
        self.calls.append((as_of, dry_run))
        run = create_run(
            "pipeline production",
            runs_root=self.root / "runs",
            pipeline_type="production_daily",
            as_of=as_of,
            invocation_source=invocation_source,
            resolved_as_of=as_of,
            timezone=timezone,
            scheduler_trigger_time=scheduler_trigger_time,
            scheduler_invocation_id=scheduler_invocation_id,
            service_execution_id=service_execution_id,
        )
        self.last_run = run
        if self.fail:
            record_failure(run, "network timeout")
            return ProductionPipelineResult(
                run,
                "failed",
                2,
                as_of,
                failed_stage="data_update",
                error_message="network timeout",
            )
        if dry_run:
            update_run_status(run, "success")
            return ProductionPipelineResult(run, "success", 0, as_of)
        publish_completed_run(
            self.root,
            as_of,
            run.run_id,
            status="success",
            existing_run=run,
        )
        return ProductionPipelineResult(
            run,
            "success",
            0,
            as_of,
            summary_path=self.root / "reports" / as_of / "production_summary.json",
        )


def make_scheduler(
    root: Path,
    store: ParquetDataStore,
    pipeline: FakePipeline,
    *,
    now: datetime,
) -> ProductionScheduler:
    return ProductionScheduler(
        settings=AppSettings(),
        raw_store=store,
        pipeline=pipeline,
        reports_root=root / "reports",
        runs_root=root / "runs",
        now=lambda: now,
        sleeper=lambda _: None,
    )


def calendar_store(root: Path, open_dates: tuple[str, ...]) -> ParquetDataStore:
    store = ParquetDataStore(root / "raw")
    store.write(
        get_dataset_spec("trade_cal"),
        pd.DataFrame(
            {
                "exchange": ["SSE"] * len(open_dates),
                "cal_date": list(open_dates),
                "is_open": [1] * len(open_dates),
            }
        ),
    )
    return store


def publish_completed_run(
    root: Path,
    as_of: str,
    run_id: str,
    *,
    status: str,
    existing_run=None,
) -> None:
    run = existing_run or create_run(
        "pipeline production",
        runs_root=root / "runs",
        pipeline_type="production_daily",
        as_of=as_of,
        run_id=run_id,
    )
    report_dir = root / "reports" / as_of
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_REPORT_ARTIFACTS:
        (report_dir / name).write_bytes(b"fixture\n")
    observation = root / "reports" / "production_observation" / f"{as_of}.json"
    observation.parent.mkdir(parents=True, exist_ok=True)
    observation.write_text("{}\n", encoding="utf-8")
    artifacts = [str(report_dir / name) for name in REQUIRED_REPORT_ARTIFACTS]
    artifacts.append(str(observation))
    atomic_write_json(
        report_dir / "production_summary.json",
        {
            "artifact_name": "production_daily_summary",
            "as_of": as_of,
            "run_id": run.run_id,
            "artifacts": artifacts,
            "observation_log_path": str(observation),
        },
    )
    if status == "success":
        update_run_status(run, "success")
    else:
        record_failure(run, "fixture failure")


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
