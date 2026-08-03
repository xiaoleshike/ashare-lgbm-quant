from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.governance.snapshot import GovernanceSnapshotResult
from ashare_quant.models.inference import InferenceResult
from ashare_quant.models.production_observation import ProductionObservationResult
from ashare_quant.models.shadow.schemas import ShadowPredictionResult
from ashare_quant.monitoring.schemas import MonitoringResult
from ashare_quant.orchestration.daily import DailyPipelineStages, StageResult
from ashare_quant.orchestration.freshness import GateResult
from ashare_quant.orchestration.lock import ProductionLockError, production_lock
from ashare_quant.orchestration.production import (
    PRODUCTION_STAGE_NAMES,
    ProductionPipeline,
)
from ashare_quant.orchestration.run_manifest import ProductionRun
from ashare_quant.orchestration.scheduler import SchedulerResult
from ashare_quant.paper_trading.service import (
    PaperTradingDailyResult,
    PaperTradingExecutionResult,
    PaperTradingRebalanceResult,
    PaperTradingReportResult,
)
from ashare_quant.research.agent.schemas import ResearchAgentResult
from ashare_quant.research.daily_report import DailyReportResult
from ashare_quant.research.decision_support import DecisionSupportResult
from ashare_quant.research.explainability.schemas import ExplainabilityResult
from ashare_quant.strategy.candidate_selector import CandidateSelectionResult


def test_production_pipeline_uses_one_outer_lock_and_records_stage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_calls = 0
    real_lock = production_lock

    @contextmanager
    def counted_lock(path: Path, *, command: str | None = None):
        nonlocal lock_calls
        lock_calls += 1
        with real_lock(path, command=command) as acquired:
            yield acquired

    monkeypatch.setattr("ashare_quant.orchestration.production.production_lock", counted_lock)
    pipeline, calls = make_pipeline(tmp_path)

    result = pipeline.run("20240105")
    manifest = load_json(result.run.manifest_path)

    assert result.status == "success"
    assert lock_calls == 1
    assert [stage["name"] for stage in manifest["stages"]] == list(PRODUCTION_STAGE_NAMES)
    assert [stage["status"] for stage in manifest["stages"]] == ["success"] * len(
        PRODUCTION_STAGE_NAMES
    )
    assert calls == [
        "model_predict",
        "strategy_candidates",
        "research_report",
        "research_explain",
        "research_decision",
        "production_observation",
    ]
    assert manifest["pipeline_type"] == "production_daily"
    assert manifest["as_of"] == "20240105"
    assert manifest["model_id"] == "champion-1"


@pytest.mark.parametrize(
    ("failed_stage", "failed_service"),
    [("shadow_prediction", "shadow"), ("monitoring", "monitoring")],
)
def test_closed_loop_component_failure_isolated_from_champion_output(
    tmp_path: Path,
    failed_stage: str,
    failed_service: str,
) -> None:
    pipeline, calls = make_pipeline(tmp_path, include_paper_trading=True)
    pipeline.shadow_prediction = FakeShadowService(
        tmp_path / "reports", fail=failed_service == "shadow"
    )
    pipeline.monitoring = FakeMonitoringService(
        tmp_path / "reports", fail=failed_service == "monitoring"
    )
    pipeline.research_agent = FakeResearchAgentService(tmp_path / "reports")
    pipeline.governance_snapshot = FakeGovernanceSnapshotService(tmp_path / "reports")

    result = pipeline.run("20240105")
    run_manifest = load_json(result.run.manifest_path)
    closed_loop = load_json(tmp_path / "reports/20240105/closed_loop_manifest.json")

    assert result.status == "success"
    assert (tmp_path / "reports/20240105/production_summary.json").is_file()
    assert calls[-1] == "paper_trading_daily"
    stage = next(item for item in closed_loop["stages"] if item["name"] == failed_stage)
    assert stage["status"] == "warning"
    assert run_manifest["status"] == "success"
    assert any(failed_stage in warning for warning in run_manifest["warnings"])


def test_closed_loop_records_component_ids_and_research_fallback(tmp_path: Path) -> None:
    pipeline, _ = make_pipeline(tmp_path)
    pipeline.shadow_prediction = FakeShadowService(tmp_path / "reports")
    pipeline.monitoring = FakeMonitoringService(tmp_path / "reports")
    pipeline.research_agent = FakeResearchAgentService(
        tmp_path / "reports", generation_mode="deterministic_fallback"
    )
    pipeline.governance_snapshot = FakeGovernanceSnapshotService(tmp_path / "reports")

    result = pipeline.run("20240105")
    run_manifest = load_json(result.run.manifest_path)
    closed_loop = load_json(tmp_path / "reports/20240105/closed_loop_manifest.json")

    assert result.status == "success"
    assert run_manifest["production_run_id"] == result.run.run_id
    assert run_manifest["shadow_run_id"] == "shadow-20240105"
    assert run_manifest["monitor_run_id"] == "monitor-20240105"
    assert run_manifest["research_run_id"] == "research-20240105"
    assert run_manifest["governance_snapshot_id"] == "governance-20240105"
    assert closed_loop["shadow_run_id"] == "shadow-20240105"
    closed_names = [item["name"] for item in closed_loop["stages"]]
    assert closed_names[: len(PRODUCTION_STAGE_NAMES)] == list(PRODUCTION_STAGE_NAMES)
    assert closed_names[-4:] == [
        "shadow_prediction",
        "monitoring",
        "research_agent",
        "governance_snapshot",
    ]
    assert all("duration_seconds" in item for item in closed_loop["stages"])
    assert all("artifact_hashes" in item for item in closed_loop["stages"])
    research = next(item for item in closed_loop["stages"] if item["name"] == "research_agent")
    assert research["status"] == "success"
    assert "deterministic fallback" in research["warnings"][0]


def test_production_pipeline_stops_after_failure_and_does_not_publish_summary(
    tmp_path: Path,
) -> None:
    pipeline, calls = make_pipeline(tmp_path, fail_stage="research_explain")

    result = pipeline.run("20240105")
    manifest = load_json(result.run.manifest_path)

    assert result.status == "failed"
    assert result.failed_stage == "research_explain"
    assert calls == [
        "model_predict",
        "strategy_candidates",
        "research_report",
        "research_explain",
    ]
    assert manifest["status"] == "failed"
    assert manifest["stages"][12]["status"] == "failed"
    assert manifest["stages"][13]["status"] == "pending"
    assert not (tmp_path / "reports" / "20240105" / "production_summary.json").exists()


@pytest.mark.parametrize(
    "failed_stage",
    (
        "model_predict",
        "strategy_candidates",
        "research_report",
        "research_explain",
        "research_decision",
        "production_observation",
    ),
)
def test_each_downstream_failure_is_terminal(tmp_path: Path, failed_stage: str) -> None:
    pipeline, calls = make_pipeline(tmp_path, fail_stage=failed_stage)

    result = pipeline.run("20240105")

    assert result.status == "failed"
    assert result.failed_stage == failed_stage
    assert calls[-1] == failed_stage
    assert "publish_production_summary" not in calls
    assert not (tmp_path / "reports" / "20240105" / "production_summary.json").exists()


def test_daily_stage_failure_stops_before_prediction(tmp_path: Path) -> None:
    pipeline, calls = make_pipeline(tmp_path)

    def fail_validation(arguments: tuple[str, ...]) -> StageResult:
        if arguments[:2] == ("data", "validate"):
            return StageResult("failed", error_message="raw validation failed")
        return StageResult("success")

    pipeline.daily_executor = fail_validation  # type: ignore[assignment]

    result = pipeline.run("20240105")

    assert result.status == "failed"
    assert result.failed_stage == "data_validate"
    assert calls == []
    assert not (tmp_path / "reports" / "20240105" / "production_summary.json").exists()


def test_missing_declared_artifact_fails_stage(tmp_path: Path) -> None:
    pipeline, _ = make_pipeline(tmp_path, omit_prediction_artifact=True)

    result = pipeline.run("20240105")

    assert result.status == "failed"
    assert result.failed_stage == "model_predict"
    assert "required artifacts" in (result.error_message or "")
    assert not (tmp_path / "reports" / "20240105" / "production_summary.json").exists()


def test_success_publishes_atomic_summary_with_top_candidates(tmp_path: Path) -> None:
    pipeline, _ = make_pipeline(tmp_path)

    result = pipeline.run("20240105")
    summary = load_json(result.summary_path or Path("missing"))

    assert summary["run_id"] == result.run.run_id
    assert summary["as_of"] == "20240105"
    assert summary["model_id"] == "champion-1"
    assert summary["candidate_count"] == 2
    assert [item["ts_code"] for item in summary["top_candidates"]] == [
        "000001.SZ",
        "600000.SH",
    ]
    assert summary["observation_log_path"].endswith("20240105.json")


def test_production_pipeline_integrates_paper_trading_without_nested_cli(
    tmp_path: Path,
) -> None:
    pipeline, calls = make_pipeline(tmp_path, include_paper_trading=True)

    result = pipeline.run("20240105")
    manifest = load_json(result.run.manifest_path)

    assert result.status == "success"
    assert calls[-1] == "paper_trading_daily"
    assert [stage["name"] for stage in manifest["stages"]][-2:] == [
        "paper_trading_daily",
        "publish_production_summary",
    ]
    paper_stage = manifest["stages"][-2]
    assert paper_stage["result"]["metrics"]["portfolio_count"] == 4
    assert (tmp_path / "reports" / "paper_trading_daily" / "20240105" / "summary.json").is_file()
    summary = load_json(tmp_path / "reports" / "20240105" / "production_summary.json")
    assert any("paper_trading_daily/20240105/summary.json" in path for path in summary["artifacts"])
    assert not list((tmp_path / "reports" / "20240105").glob(".*production_summary*.pending.json"))


def test_paper_trading_failure_does_not_publish_formal_summary(tmp_path: Path) -> None:
    pipeline, _ = make_pipeline(
        tmp_path,
        include_paper_trading=True,
        fail_paper_trading=True,
    )

    result = pipeline.run("20240105")

    assert result.status == "failed"
    assert result.failed_stage == "paper_trading_daily"
    assert not (tmp_path / "reports" / "20240105" / "production_summary.json").exists()
    assert not list((tmp_path / "reports" / "20240105").glob(".*production_summary*.pending.json"))


def test_production_summary_keeps_all_fifty_selected_candidates(tmp_path: Path) -> None:
    pipeline, _ = make_pipeline(tmp_path)
    report_dir = tmp_path / "reports" / "20240105"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact = report_dir / "candidates.csv"
    artifact.write_text("fixture\n", encoding="utf-8")
    observation = tmp_path / "reports" / "production_observation" / "20240105.json"
    observation.parent.mkdir(parents=True, exist_ok=True)
    observation.write_text("{}\n", encoding="utf-8")
    candidates = pd.DataFrame(
        {
            "rank": range(1, 51),
            "ts_code": [f"{code:06d}.SZ" for code in range(1, 51)],
            "prediction_score": [1.0 - code / 100 for code in range(50)],
        }
    )
    candidate_result = CandidateSelectionResult(
        "20240105",
        "champion-1",
        100,
        50,
        {},
        artifact,
        candidates,
    )
    state = {
        "candidate_result": candidate_result,
        "artifacts": [str(artifact), str(observation)],
        "observation_path": str(observation),
    }
    run = ProductionRun("summary-run", tmp_path / "runs" / "20240105" / "summary-run")

    pipeline._publish_summary(run, "20240105", state)
    summary = load_json(report_dir / "production_summary.json")

    assert summary["candidate_count"] == 50
    assert len(summary["top_candidates"]) == 50
    assert summary["top_candidates"][-1]["rank"] == 50


def test_repeated_date_creates_distinct_run_ids(tmp_path: Path) -> None:
    first, _ = make_pipeline(tmp_path)
    second, _ = make_pipeline(tmp_path)

    first_result = first.run("20240105")
    second_result = second.run("20240105")

    assert first_result.run.run_id != second_result.run.run_id


def test_dry_run_only_validates_and_plans(tmp_path: Path) -> None:
    pipeline, calls = make_pipeline(tmp_path)

    result = pipeline.run("20240105", dry_run=True)
    manifest = load_json(result.run.manifest_path)

    assert result.status == "success"
    assert calls == []
    assert [stage["name"] for stage in manifest["stages"]] == [
        "dry_run_validation",
        "manifest_planning",
    ]
    assert manifest["artifact_paths"] == []
    assert manifest["stages"][1]["result"]["metrics"]["planned_artifact_paths"]
    assert not (tmp_path / "reports" / "20240105").exists()


def test_existing_lock_blocks_production_before_stages(tmp_path: Path) -> None:
    pipeline, calls = make_pipeline(tmp_path)
    lock_path = tmp_path / "runs" / ".production.lock"

    with production_lock(lock_path, command="other run"):
        with pytest.raises(ProductionLockError, match="other run"):
            pipeline.run("20240105")

    assert calls == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [("success", 0), ("failed", 2), ("skipped", 0)],
)
def test_production_cli_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    expected: int,
) -> None:
    run = ProductionRun("cli-run", tmp_path / "runs" / "20240105" / "cli-run")

    class FakePipeline:
        def __init__(self, **kwargs) -> None:
            pass

    class FakeScheduler:
        def __init__(self, **kwargs) -> None:
            pass

        def run(self, as_of: str, *, dry_run: bool = False) -> SchedulerResult:
            return SchedulerResult(
                status,
                2 if status == "failed" else 0,
                "invocation-id",
                tmp_path / "invocation.json",
                as_of,
                run.run_id,
                skipped_reason="already_successful" if status == "skipped" else None,
                error_message="model_predict failed" if status == "failed" else None,
            )

    monkeypatch.setattr("ashare_quant.cli.ProductionPipeline", FakePipeline)
    monkeypatch.setattr("ashare_quant.cli.ProductionScheduler", FakeScheduler)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "pipeline",
            "production",
            "--as-of",
            "20240105",
            "--dry-run",
        ]
    )

    assert exit_code == expected


def test_production_orchestrator_has_no_training_promotion_or_trading_dependency() -> None:
    source = Path("src/ashare_quant/orchestration/production.py").read_text(encoding="utf-8")

    assert ".train(" not in source
    assert ".promote_model(" not in source
    assert "Backtest" not in source
    assert "ashare_quant.backtest" not in source
    assert "ashare_quant.trading" not in source
    assert "broker" not in source.lower()


def make_pipeline(
    tmp_path: Path,
    *,
    fail_stage: str | None = None,
    omit_prediction_artifact: bool = False,
    include_paper_trading: bool = False,
    fail_paper_trading: bool = False,
) -> tuple[ProductionPipeline, list[str]]:
    config = tmp_path / "config.yaml"
    config.write_text("project_name: production-test\n", encoding="utf-8")
    reports = tmp_path / "reports"
    calls: list[str] = []

    def daily_executor(arguments: tuple[str, ...]) -> StageResult:
        return StageResult("success", metrics={"command": list(arguments)})

    services = FakeServices(
        reports,
        calls,
        fail_stage=fail_stage,
        omit_prediction_artifact=omit_prediction_artifact,
    )
    readiness = FakeReadiness()
    pipeline = ProductionPipeline(
        config_path=config,
        processed_root=tmp_path / "processed",
        reports_root=reports,
        daily_executor=daily_executor,  # type: ignore[arg-type]
        daily_stages=DailyPipelineStages(),
        readiness=readiness,
        readiness_executor=lambda gate, as_of: GateResult(gate, as_of, (), (), {}),
        as_of_resolver=lambda requested: requested or "20240105",
        inference=services,  # type: ignore[arg-type]
        candidates=services,  # type: ignore[arg-type]
        research_report=services,  # type: ignore[arg-type]
        explainability=services,  # type: ignore[arg-type]
        decision_support=services,  # type: ignore[arg-type]
        observation=services,  # type: ignore[arg-type]
        paper_trading=FakePaperTrading(
            reports,
            calls,
            fail=fail_paper_trading,
        )
        if include_paper_trading
        else None,
        runs_root=tmp_path / "runs",
        lock_path=tmp_path / "runs" / ".production.lock",
    )
    return pipeline, calls


class FakeReadiness:
    def check_all(self, as_of: str) -> tuple[GateResult, ...]:
        return (GateResult("all", as_of, (), (), {}),)


class FakeServices:
    def __init__(
        self,
        reports_root: Path,
        calls: list[str],
        *,
        fail_stage: str | None,
        omit_prediction_artifact: bool,
    ) -> None:
        self.reports_root = reports_root
        self.calls = calls
        self.fail_stage = fail_stage
        self.omit_prediction_artifact = omit_prediction_artifact

    def _call(self, name: str) -> None:
        self.calls.append(name)
        if self.fail_stage == name:
            raise DataError(f"forced {name} failure")

    def predict(self, as_of: str) -> InferenceResult:
        self._call("model_predict")
        output = self.reports_root / as_of
        output.mkdir(parents=True, exist_ok=True)
        predictions = pd.DataFrame(
            {
                "trade_date": [as_of, as_of],
                "ts_code": ["000001.SZ", "600000.SH"],
                "prediction_score": [0.8, 0.7],
                "model_id": ["champion-1", "champion-1"],
            }
        )
        predictions.to_parquet(output / "predictions.parquet", index=False)
        (output / "ranking.csv").write_text("rank,ts_code,prediction_score\n", encoding="utf-8")
        (output / "summary.json").write_text("{}\n", encoding="utf-8")
        if not self.omit_prediction_artifact:
            (output / "manifest.json").write_text("{}\n", encoding="utf-8")
        return InferenceResult(as_of, "champion-1", 2, 2, 2, output, predictions)

    def select(self, as_of: str) -> CandidateSelectionResult:
        self._call("strategy_candidates")
        output = self.reports_root / as_of / "candidates.csv"
        candidates = pd.DataFrame(
            {
                "rank": [1, 2],
                "ts_code": ["000001.SZ", "600000.SH"],
                "prediction_score": [0.8, 0.7],
                "selection_reason": ["eligible", "eligible"],
                "trade_date": [as_of, as_of],
                "model_id": ["champion-1", "champion-1"],
            }
        )
        candidates.to_csv(output, index=False)
        (output.parent / "candidates_manifest.json").write_text("{}\n", encoding="utf-8")
        return CandidateSelectionResult(as_of, "champion-1", 2, 2, {}, output, candidates)

    def generate(self, as_of: str) -> DailyReportResult | DecisionSupportResult:
        if "research_report" not in self.calls:
            self._call("research_report")
            report = self.reports_root / as_of / "daily_report.md"
            summary = report.parent / "research_summary.json"
            report.write_text("# Report\n", encoding="utf-8")
            summary.write_text("{}\n", encoding="utf-8")
            return DailyReportResult(as_of, "champion-1", 2, report, summary, ())
        self._call("research_decision")
        json_path = self.reports_root / as_of / "decision.json"
        markdown = json_path.parent / "decision_report.md"
        json_path.write_text("{}\n", encoding="utf-8")
        markdown.write_text("# Decision\n", encoding="utf-8")
        return DecisionSupportResult(as_of, "champion-1", 2, json_path, markdown)

    def explain(self, as_of: str) -> ExplainabilityResult:
        self._call("research_explain")
        json_path = self.reports_root / as_of / "explanations.json"
        markdown = json_path.parent / "explanations.md"
        json_path.write_text("{}\n", encoding="utf-8")
        markdown.write_text("# Explanations\n", encoding="utf-8")
        return ExplainabilityResult(
            as_of, "champion-1", 2, "lightgbm_pred_contrib", str(json_path), str(markdown)
        )

    def record(self, as_of: str) -> ProductionObservationResult:
        self._call("production_observation")
        output = self.reports_root / "production_observation" / f"{as_of}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n", encoding="utf-8")
        return ProductionObservationResult(as_of, "champion-1", 2, output)


class FakePaperTrading:
    def __init__(
        self,
        reports_root: Path,
        calls: list[str],
        *,
        fail: bool = False,
    ) -> None:
        self.reports_root = reports_root
        self.calls = calls
        self.fail = fail

    def run_daily(
        self,
        as_of: str,
        *,
        production_summary_path: Path | None = None,
    ) -> PaperTradingDailyResult:
        assert production_summary_path is not None and production_summary_path.is_file()
        assert production_summary_path.name.startswith(".production_summary.")
        assert not (self.reports_root / as_of / "production_summary.json").exists()
        self.calls.append("paper_trading_daily")
        if self.fail:
            raise DataError("forced paper trading failure")
        output = self.reports_root / "paper_trading_daily" / as_of
        output.mkdir(parents=True, exist_ok=True)
        report = output / "report.md"
        summary = output / "summary.json"
        report.write_text("# Paper\n", encoding="utf-8")
        summary.write_text("{}\n", encoding="utf-8")
        return PaperTradingDailyResult(
            as_of,
            PaperTradingRebalanceResult(as_of, "next_open", 4, 80, output),
            PaperTradingExecutionResult(as_of, 4, 60, 4, output),
            PaperTradingReportResult(as_of, report, summary, 4),
        )


class FakeShadowService:
    def __init__(self, reports_root: Path, *, fail: bool = False) -> None:
        self.reports_root = reports_root
        self.fail = fail

    def predict(
        self,
        as_of: str,
        *,
        expected_production_run_id: str | None = None,
    ) -> ShadowPredictionResult:
        assert expected_production_run_id is not None
        if self.fail:
            raise DataError("forced shadow failure")
        output = self.reports_root / "shadow_predictions" / as_of
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": [as_of], "ts_code": ["000001.SZ"]}).to_parquet(
            output / "predictions.parquet", index=False
        )
        (output / "manifest.json").write_text("{}\n", encoding="utf-8")
        return ShadowPredictionResult(
            as_of,
            expected_production_run_id,
            f"shadow-{as_of}",
            1,
            6,
            output,
        )


class FakeMonitoringService:
    def __init__(self, reports_root: Path, *, fail: bool = False) -> None:
        self.reports_root = reports_root
        self.fail = fail

    def run(self, as_of: str) -> MonitoringResult:
        if self.fail:
            raise DataError("forced monitoring failure")
        output = self.reports_root / "model_monitor" / as_of
        output.mkdir(parents=True, exist_ok=True)
        for name in ("health.json", "monitor_summary.json", "manifest.json"):
            (output / name).write_text("{}\n", encoding="utf-8")
        return MonitoringResult(as_of, f"monitor-{as_of}", output, 4, 2, 0, 0)


class FakeResearchAgentService:
    def __init__(self, reports_root: Path, *, generation_mode: str = "llm") -> None:
        self.reports_root = reports_root
        self.generation_mode = generation_mode

    def generate(self, as_of: str) -> ResearchAgentResult:
        output = self.reports_root / "research_agent" / as_of
        output.mkdir(parents=True, exist_ok=True)
        for name in ("daily_research.md", "research_summary.json", "manifest.json"):
            (output / name).write_text("{}\n", encoding="utf-8")
        return ResearchAgentResult(
            as_of,
            output,
            self.generation_mode,
            f"research-{as_of}",
        )


class FakeGovernanceSnapshotService:
    def __init__(self, reports_root: Path) -> None:
        self.reports_root = reports_root

    def publish_daily(
        self,
        as_of: str,
        *,
        production_run_id: str,
    ) -> GovernanceSnapshotResult:
        assert production_run_id
        output = self.reports_root / "governance" / as_of
        output.mkdir(parents=True, exist_ok=True)
        paths = tuple(
            output / name
            for name in (
                "status.json",
                "validation.json",
                "recovery.json",
                "promotion_status.json",
                "manifest.json",
            )
        )
        for path in paths:
            path.write_text("{}\n", encoding="utf-8")
        return GovernanceSnapshotResult(f"governance-{as_of}", paths, ())


class DataError(ValueError):
    pass


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
