"""Operational governance status and recovery tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ashare_quant.cli import main
from ashare_quant.config.settings import (
    AppSettings,
    PaperPortfolioSettings,
    PaperTradingSettings,
    PathSettings,
)
from ashare_quant.governance.recovery import registry_recovery_preview
from ashare_quant.governance.service import GovernanceService
from ashare_quant.governance.snapshot import DailyGovernanceSnapshotService
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.utils.manifest import atomic_write_json


def _settings(tmp_path: Path) -> tuple[AppSettings, Path]:
    paths = PathSettings(
        raw_data=tmp_path / "data/raw",
        processed_data=tmp_path / "data/processed",
        parquet_store=tmp_path / "data/parquet",
        duckdb_path=tmp_path / "data/test.duckdb",
        reports=tmp_path / "reports",
        models=tmp_path / "models",
        backtests=tmp_path / "backtests",
        paper_trading=tmp_path / "paper_trading",
        data_quality_logs=tmp_path / "logs",
    )
    settings = AppSettings(paths=paths)
    config_path = tmp_path / "config" / "default.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n", encoding="utf-8")
    return settings, config_path


def _model_registry(settings: AppSettings, *, champion: bool = True) -> Path:
    artifact = settings.paths.models / "artifact"
    artifact.mkdir(parents=True)
    features = ["ret_5d", "amount_ratio_20d"]
    atomic_write_json(artifact / "feature_list.json", {"features": features})
    atomic_write_json(artifact / "manifest.json", {"artifact_name": "ranker", "horizon": 5})
    atomic_write_json(artifact / "metrics.json", {"test": {"rank_ic": 0.01}})
    (artifact / "model.txt").write_text("immutable model\n", encoding="utf-8")
    registry = settings.paths.models / "registry.json"
    atomic_write_json(
        registry,
        {
            "schema_version": 1,
            "updated_at": "2026-08-01T12:00:00+00:00",
            "models": [
                {
                    "model_id": "model_a",
                    "experiment_id": "experiment_a",
                    "model_type": "lightgbm_ranker",
                    "feature_hash": feature_list_hash(tuple(features)),
                    "feature_count": len(features),
                    "training_date_range": {"start": "20100101", "end": "20260701"},
                    "validation_metrics": {"rank_ic": 0.01},
                    "test_metrics": {"rank_ic": 0.01},
                    "git_commit": "abc",
                    "config_hash": "cfg",
                    "creation_time": "2026-07-01T00:00:00+00:00",
                    "artifact_path": str(artifact),
                    "status": "champion" if champion else "candidate",
                }
            ],
        },
    )
    return registry


def _production(settings: AppSettings, tmp_path: Path, *, missing: bool = False) -> None:
    report_dir = settings.paths.reports / "20260731"
    report_dir.mkdir(parents=True)
    artifact = report_dir / "predictions.parquet"
    if not missing:
        pd.DataFrame(
            {
                "trade_date": ["20260731"],
                "ts_code": ["000001.SZ"],
                "prediction_score": [0.2],
                "model_id": ["model_a"],
            }
        ).to_parquet(artifact, index=False)
    candidates = report_dir / "candidates.csv"
    pd.DataFrame(
        {
            "trade_date": ["20260731"],
            "ts_code": ["000001.SZ"],
            "prediction_score": [0.2],
            "model_id": ["model_a"],
        }
    ).to_csv(candidates, index=False)
    prediction_manifest = {
        "schema_version": 1,
        "artifact_name": "production_predictions",
        "as_of": "20260731",
        "model_id": "model_a",
        "feature_hash": "feature_hash",
        "prediction_count": 1,
        "config_hash": "config_hash",
    }
    atomic_write_json(report_dir / "manifest.json", prediction_manifest)
    atomic_write_json(
        report_dir / "candidates_manifest.json",
        {
            "schema_version": 1,
            "artifact_name": "production_candidates",
            "as_of": "20260731",
            "model_id": "model_a",
            "feature_hash": "feature_hash",
            "candidate_count": 1,
            "config_hash": "config_hash",
            "prediction_manifest": prediction_manifest,
        },
    )
    atomic_write_json(
        report_dir / "production_summary.json",
        {
            "schema_version": 1,
            "artifact_name": "production_daily_summary",
            "as_of": "20260731",
            "run_id": "run_1",
            "completed_time": "2026-07-31T12:00:00+00:00",
            "model_id": "model_a",
            "candidate_count": 1,
            "top_candidates": ["000001.SZ"],
            "artifacts": [
                str(artifact),
                str(candidates),
                str(report_dir / "manifest.json"),
                str(report_dir / "candidates_manifest.json"),
            ],
        },
    )
    run = tmp_path / "runs" / "20260731" / "run_1"
    run.mkdir(parents=True)
    atomic_write_json(
        run / "manifest.json",
        {"schema_version": 1, "run_id": "run_1", "status": "success"},
    )


def _monitor(settings: AppSettings, *, corrupt: bool = False) -> None:
    root = settings.paths.reports / "model_monitor" / "20260731"
    (root / "performance").mkdir(parents=True)
    atomic_write_json(root / "health.json", {"score_std": 1.0})
    atomic_write_json(root / "performance" / "manifest.json", {"status": "success"})
    pd.DataFrame({"model_id": ["model_a"]}).to_parquet(
        root / "performance" / "performance_metrics.parquet", index=False
    )
    pd.DataFrame({"portfolio_id": ["paper_a"]}).to_parquet(
        root / "portfolio_metrics.parquet", index=False
    )
    atomic_write_json(
        root / "monitor_summary.json",
        {
            "artifact_name": "production_monitor_summary",
            "alerts": {"alerts": []},
        },
    )
    hashes = {
        "health": file_sha256(root / "health.json"),
        "performance_manifest": file_sha256(root / "performance" / "manifest.json"),
        "performance_metrics": file_sha256(root / "performance" / "performance_metrics.parquet"),
        "portfolio_metrics": file_sha256(root / "portfolio_metrics.parquet"),
    }
    if corrupt:
        hashes["health"] = "0" * 64
    atomic_write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "artifact_name": "production_monitor_manifest",
            "monitor_metric_file_hashes": hashes,
        },
    )


def test_status_handles_missing_and_valid_registry(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    missing = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).status()
    assert missing.report.summary["champion"]["model_id"] is None
    _model_registry(settings)
    valid = GovernanceService(settings=settings, config_path=config, project_root=tmp_path).status()
    assert valid.report.summary["champion"]["model_id"] == "model_a"
    assert valid.report_path == settings.paths.reports / "governance" / "status.json"
    assert (settings.paths.reports / "governance" / "status.manifest.json").is_file()
    snapshots = list((settings.paths.reports / "governance" / "history" / "status").iterdir())
    assert len(snapshots) == 2


def test_status_reports_invalid_monitor_summary_without_crashing(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    root = settings.paths.reports / "model_monitor" / "20260731"
    root.mkdir(parents=True)
    (root / "monitor_summary.json").write_text("{broken", encoding="utf-8")
    result = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).status()
    assert any(
        item.name == "monitor.latest" and item.status == "FAIL" for item in result.report.checks
    )


def test_production_validation_detects_missing_artifact_and_invalid_champion(
    tmp_path: Path,
) -> None:
    settings, config = _settings(tmp_path)
    _model_registry(settings, champion=False)
    _production(settings, tmp_path, missing=True)
    result = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).validate_production()
    assert result.report.status == "FAIL"
    failed = {item.name for item in result.report.checks if item.status == "FAIL"}
    assert "pipeline.artifacts" in failed
    assert "model.champion" in failed


def test_production_validation_detects_monitor_hash_mismatch(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    _model_registry(settings)
    _production(settings, tmp_path)
    _monitor(settings, corrupt=True)
    result = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).validate_production()
    assert result.report.status == "FAIL"
    assert any(
        item.name == "monitor.manifest" and item.status == "FAIL" for item in result.report.checks
    )


def test_valid_production_contract_has_no_hard_failure(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    _model_registry(settings)
    _production(settings, tmp_path)
    _monitor(settings)
    result = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).validate_production()
    assert result.report.status == "WARNING"
    assert not [item for item in result.report.checks if item.status == "FAIL"]


def test_recovery_reports_corruption_staging_and_interrupted_apply(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    registry = _model_registry(settings)
    registry.write_text("{broken", encoding="utf-8")
    staging = settings.paths.models / "registry_versions" / ".staging"
    staging.mkdir(parents=True)
    journal = (
        settings.paths.models / "promotion_requests/request_a/apply/apply_a/apply_pending.json"
    )
    journal.parent.mkdir(parents=True)
    atomic_write_json(journal, {"status": "APPLY_PENDING"})
    result = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).validate_recovery()
    assert result.report.status == "FAIL"
    assert str(journal) in result.report.summary["interrupted_transactions"]
    assert str(staging) in result.report.summary["incomplete_publications"]


def test_production_validation_rejects_duplicate_paper_ledger_key(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    settings = settings.model_copy(
        update={
            "paper_trading": PaperTradingSettings(
                portfolios=(
                    PaperPortfolioSettings(
                        portfolio_id="paper_a",
                        signal_type="champion",
                        model_id="champion",
                    ),
                )
            )
        }
    )
    _model_registry(settings)
    _production(settings, tmp_path)
    ledger = settings.paths.paper_trading / "paper_a" / "orders.parquet"
    ledger.parent.mkdir(parents=True)
    pd.DataFrame({"order_id": ["same", "same"]}).to_parquet(ledger, index=False)
    result = GovernanceService(
        settings=settings, config_path=config, project_root=tmp_path
    ).validate_production()
    assert any(
        item.name == "paper.paper_a.orders.parquet" and item.status == "FAIL"
        for item in result.report.checks
    )


def test_governance_snapshot_tampering_is_not_overwritten(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    service = GovernanceService(settings=settings, config_path=config, project_root=tmp_path)
    result = service.status()
    history = settings.paths.reports / "governance" / "history" / "status"
    snapshot = next(history.iterdir())
    (snapshot / "status.json").write_text("{}\n", encoding="utf-8")
    from ashare_quant.data.exceptions import DataValidationError

    try:
        service.status()
    except DataValidationError:
        pass
    else:
        raise AssertionError("tampered immutable snapshot was silently overwritten")
    assert result.manifest_path.is_file()


def test_governance_is_read_only_and_does_not_call_stateful_services(
    tmp_path: Path, monkeypatch
) -> None:
    from ashare_quant.models.challenger import ChallengerTrainer
    from ashare_quant.models.inference import ProductionInferenceEngine
    from ashare_quant.models.promotion.apply import PromotionApplyService
    from ashare_quant.paper_trading.service import PaperTradingService

    def forbidden(*args, **kwargs):
        raise AssertionError("stateful service was called")

    monkeypatch.setattr(ProductionInferenceEngine, "predict", forbidden)
    monkeypatch.setattr(ChallengerTrainer, "train", forbidden)
    monkeypatch.setattr(PaperTradingService, "execute", forbidden)
    monkeypatch.setattr(PromotionApplyService, "apply", forbidden)
    settings, config = _settings(tmp_path)
    registry = _model_registry(settings)
    before = registry.read_bytes()
    GovernanceService(settings=settings, config_path=config, project_root=tmp_path).status()
    assert registry.read_bytes() == before


def test_governance_cli_returns_nonzero_on_hard_failure(tmp_path: Path) -> None:
    settings, _ = _settings(tmp_path)
    config = tmp_path / "config" / "cli.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "reports": str(settings.paths.reports),
                    "models": str(settings.paths.models),
                    "paper_trading": str(settings.paths.paper_trading),
                }
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(config), "governance", "validate-production"]) == 1
    assert main(["--config", str(config), "governance", "status"]) == 0


def test_registry_recovery_preview_selects_latest_valid_version_without_mutation(
    tmp_path: Path,
) -> None:
    settings, _ = _settings(tmp_path)
    registry = _model_registry(settings)
    before = registry.read_bytes()
    payload = json.loads(before)
    payload["updated_at"] = "2026-08-02T12:00:00+00:00"
    payload["registry_version_id"] = "registry_v2"
    payload["promotion_request_id"] = "request_v2"
    payload["approval_event_id"] = "approval_v2"
    versions = settings.paths.models / "registry_versions"
    version = versions / "registry_v2.json"
    atomic_write_json(version, payload)
    (versions / "corrupt.json").write_text("{broken", encoding="utf-8")
    orphan = dict(payload)
    orphan["updated_at"] = "2026-08-03T12:00:00+00:00"
    orphan["registry_version_id"] = "registry_orphan"
    orphan["promotion_request_id"] = "request_orphan"
    atomic_write_json(versions / "registry_orphan.json", orphan)
    assignment = settings.paths.models / "champion_history/assignment_v2.json"
    atomic_write_json(
        assignment,
        {
            "champion_assignment_id": "assignment_v2",
            "model_id": "model_a",
            "registry_version_id": "registry_v2",
            "activated_at": "2026-08-02T12:00:00+00:00",
        },
    )
    apply_manifest = (
        settings.paths.models / "promotion_requests/request_v2/apply/apply_v2/manifest.json"
    )
    atomic_write_json(
        apply_manifest,
        {
            "registry_version_id": "registry_v2",
            "registry_file_hash": file_sha256(version),
            "champion_history_hash": file_sha256(assignment),
        },
    )

    preview = registry_recovery_preview(settings.paths.models)

    assert preview.latest_valid_registry == versions / "registry_v2.json"
    assert preview.champion_model_id == "model_a"
    assert preview.champion_assignment_id == "assignment_v2"
    assert preview.transition_manifest == apply_manifest
    assert preview.corrupted_versions == ("corrupt.json", "registry_orphan.json")
    assert registry.read_bytes() == before


def test_daily_governance_snapshot_is_dated_atomic_and_idempotent(tmp_path: Path) -> None:
    settings, config = _settings(tmp_path)
    _model_registry(settings)
    _production(settings, tmp_path)
    _monitor(settings)
    service = DailyGovernanceSnapshotService(
        settings=settings,
        config_path=config,
        project_root=tmp_path,
    )

    first = service.publish_daily("20260731", production_run_id="run_1")
    second = service.publish_daily("20260731", production_run_id="run_1")

    assert first.snapshot_id == second.snapshot_id
    assert [path.name for path in first.artifact_paths] == [
        "status.json",
        "validation.json",
        "recovery.json",
        "promotion_status.json",
        "manifest.json",
    ]
    assert all(path.is_file() for path in first.artifact_paths)
    history = settings.paths.reports / "governance/20260731/history"
    assert [path.name for path in history.iterdir()] == [first.snapshot_id]
