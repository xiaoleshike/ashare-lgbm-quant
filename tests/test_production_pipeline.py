from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.models.inference import InferenceResult
from ashare_quant.models.production_observation import ProductionObservationResult
from ashare_quant.orchestration.daily import DailyPipelineStages, StageResult
from ashare_quant.orchestration.freshness import GateResult
from ashare_quant.orchestration.lock import ProductionLockError, production_lock
from ashare_quant.orchestration.production import (
    PRODUCTION_STAGE_NAMES,
    ProductionPipeline,
    ProductionPipelineResult,
)
from ashare_quant.orchestration.run_manifest import ProductionRun
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


@pytest.mark.parametrize(("status", "expected"), [("success", 0), ("failed", 2)])
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

        def run(self, as_of: str, *, dry_run: bool = False) -> ProductionPipelineResult:
            return ProductionPipelineResult(
                run,
                status,
                0 if status == "success" else 2,
                as_of,
                failed_stage=None if status == "success" else "model_predict",
            )

    monkeypatch.setattr("ashare_quant.cli.ProductionPipeline", FakePipeline)

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


class DataError(ValueError):
    pass


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
