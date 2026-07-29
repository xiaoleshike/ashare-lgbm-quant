from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import (
    AppSettings,
    PaperPortfolioSettings,
    PaperTradingSettings,
    PathSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.performance_observation.schemas import OBSERVATION_COLUMNS
from ashare_quant.monitoring.performance_observation.storage import (
    logical_observation_hash,
    publish_observation_artifact,
)
from ashare_quant.monitoring.schemas import MonitoringResult
from ashare_quant.monitoring.service import MonitoringService
from ashare_quant.utils.manifest import config_hash

AS_OF = "20240103"
MODEL_ID = "champion-fixture"
FEATURE_HASH = "f" * 64


def test_monitoring_is_read_only_deterministic_and_keeps_portfolios_isolated(
    tmp_path: Path,
) -> None:
    service, input_paths = monitoring_fixture(tmp_path)
    source_bytes = {path: path.read_bytes() for path in input_paths}

    first = service.run(AS_OF)
    first_outputs = {
        path.name: path.read_bytes() for path in first.output_dir.iterdir() if path.is_file()
    }
    second = service.run(AS_OF)

    assert first.run_id == second.run_id
    assert {
        path.name: path.read_bytes() for path in second.output_dir.iterdir() if path.is_file()
    } == first_outputs
    assert all(path.read_bytes() == source_bytes[path] for path in input_paths)
    assert {path.name for path in first.output_dir.iterdir()} == {
        "health.json",
        "performance",
        "portfolio_metrics.parquet",
        "monitor_summary.json",
        "monitor_report.md",
        "manifest.json",
    }
    metrics = pd.read_parquet(first.output_dir / "portfolio_metrics.parquet").set_index(
        "portfolio_id"
    )
    assert set(metrics.index) == {"alpha", "beta"}
    assert metrics.loc["alpha", "drawdown"] == pytest.approx(-0.1)
    assert metrics.loc["alpha", "position_count"] == 2
    assert metrics.loc["alpha", "max_position_weight"] == pytest.approx(0.5)
    assert metrics.loc["alpha", "top5_concentration"] == pytest.approx(0.8)
    assert metrics.loc["alpha", "cash_ratio"] == pytest.approx(0.2)
    assert metrics.loc["beta", "nav"] == pytest.approx(1.1)
    health = _json(first.output_dir / "health.json")
    assert health["universe_size"] == 3
    assert health["model_universe_size"] == 2
    assert health["prediction_count"] == 2
    assert health["feature_missing_ratios"] == {"ret_1d": 0.0, "sparse": 0.5}
    assert health["duplicate_score_ratio"] == 0.0
    summary = _json(first.output_dir / "monitor_summary.json")
    assert summary["scope"]["labels_read"] is False
    assert summary["scope"]["trading_state_modified"] is False
    assert len(summary["performance"]["models"]) == 1


def test_monitoring_ignores_ledger_rows_after_as_of(tmp_path: Path) -> None:
    service, _ = monitoring_fixture(tmp_path)
    equity_path = tmp_path / "paper" / "alpha" / "equity_curve.parquet"
    equity = pd.read_parquet(equity_path)
    future = equity.iloc[[-1]].assign(
        equity_id="future",
        as_of="20240104",
        equity=10.0,
        nav=0.1,
        daily_return=-0.888,
        drawdown=-0.9,
    )
    pd.concat([equity, future], ignore_index=True).to_parquet(equity_path, index=False)

    result = service.run(AS_OF)

    metrics = pd.read_parquet(result.output_dir / "portfolio_metrics.parquet").set_index(
        "portfolio_id"
    )
    assert metrics.loc["alpha", "nav"] == pytest.approx(0.9)
    assert metrics.loc["alpha", "drawdown"] == pytest.approx(-0.1)


def test_monitoring_accepts_portfolio_before_first_execution(tmp_path: Path) -> None:
    service, _ = monitoring_fixture(tmp_path)
    portfolio_root = tmp_path / "paper" / "alpha"
    for name in ("trades.parquet", "positions.parquet", "equity_curve.parquet"):
        (portfolio_root / name).unlink()

    result = service.run(AS_OF)

    metrics = pd.read_parquet(result.output_dir / "portfolio_metrics.parquet").set_index(
        "portfolio_id"
    )
    assert metrics.loc["alpha", "nav"] == pytest.approx(1.0)
    assert metrics.loc["alpha", "daily_return"] == pytest.approx(0.0)
    assert metrics.loc["alpha", "drawdown"] == pytest.approx(0.0)
    assert metrics.loc["alpha", "position_count"] == 0
    assert metrics.loc["alpha", "cash_ratio"] == pytest.approx(1.0)


def test_monitoring_rejects_feature_hash_mismatch(tmp_path: Path) -> None:
    service, _ = monitoring_fixture(tmp_path)
    path = tmp_path / "reports" / AS_OF / "candidates_manifest.json"
    payload = _json(path)
    payload["feature_hash"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataValidationError, match="feature hash mismatch"):
        service.run(AS_OF)


def test_failed_monitoring_publication_keeps_previous_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = monitoring_fixture(tmp_path)
    result = service.run(AS_OF)
    manifest_path = result.output_dir / "manifest.json"
    before = manifest_path.read_bytes()

    def fail_to_parquet(self: pd.DataFrame, path: Path, **kwargs: object) -> None:
        raise OSError("simulated monitoring publication failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)
    with pytest.raises(OSError, match="simulated"):
        service.run(AS_OF)

    assert manifest_path.read_bytes() == before


def test_performance_failure_keeps_previous_complete_monitor_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = monitoring_fixture(tmp_path)
    result = service.run(AS_OF)
    before = {
        path.relative_to(result.output_dir): path.read_bytes()
        for path in result.output_dir.glob("**/*")
        if path.is_file()
    }

    def fail_performance(as_of: str) -> None:
        raise DataValidationError(f"performance failure: {as_of}")

    monkeypatch.setattr(service.performance_service, "build", fail_performance)
    with pytest.raises(DataValidationError, match="performance failure"):
        service.run(AS_OF)

    assert before == {
        path.relative_to(result.output_dir): path.read_bytes()
        for path in result.output_dir.glob("**/*")
        if path.is_file()
    }


def test_monitoring_does_not_read_labels_or_call_trading_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = monitoring_fixture(tmp_path)
    original_read_parquet = pd.read_parquet

    def guarded_read_parquet(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        assert "labels" not in str(path)
        return original_read_parquet(path, *args, **kwargs)

    def forbidden_execution(*args: object, **kwargs: object) -> None:
        raise AssertionError("paper trading execution must not be called")

    monkeypatch.setattr(pd, "read_parquet", guarded_read_parquet)
    monkeypatch.setattr(
        "ashare_quant.paper_trading.service.PaperTradingService.execute",
        forbidden_execution,
    )

    service.run(AS_OF)


def test_monitor_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class SuccessfulService:
        def __init__(self, **kwargs: object) -> None:
            pass

        def run(self, as_of: str) -> MonitoringResult:
            return MonitoringResult(as_of, "monitor-run", tmp_path / "out", 2, 100)

    monkeypatch.setattr("ashare_quant.cli.MonitoringService", SuccessfulService)
    assert main(["--config", "config/default.yaml", "monitor", "run", "--as-of", AS_OF]) == 0
    assert "monitor_run: as_of=20240103" in capsys.readouterr().out

    class FailingService(SuccessfulService):
        def run(self, as_of: str) -> MonitoringResult:
            raise DataValidationError("source hash mismatch")

    monkeypatch.setattr("ashare_quant.cli.MonitoringService", FailingService)
    assert main(["--config", "config/default.yaml", "monitor", "run", "--as-of", AS_OF]) == 2
    assert "source hash mismatch" in capsys.readouterr().err


def monitoring_fixture(tmp_path: Path) -> tuple[MonitoringService, tuple[Path, ...]]:
    reports = tmp_path / "reports"
    paper = tmp_path / "paper"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: fixture\n", encoding="utf-8")
    settings = AppSettings(
        paths=PathSettings(reports=reports, paper_trading=paper),
        paper_trading=PaperTradingSettings(
            portfolios=(
                PaperPortfolioSettings(
                    portfolio_id="alpha",
                    signal_type="model",
                    model_id="alpha-model",
                ),
                PaperPortfolioSettings(
                    portfolio_id="beta",
                    signal_type="model",
                    model_id="beta-model",
                ),
            )
        ),
    )
    report_dir = reports / AS_OF
    report_dir.mkdir(parents=True)
    predictions = pd.DataFrame(
        {
            "trade_date": [AS_OF, AS_OF],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "prediction_score": [0.8, 0.2],
            "model_id": [MODEL_ID, MODEL_ID],
        }
    )
    candidates = predictions.assign(rank=[1, 2], selection_reason=["selected", "selected"])
    predictions.to_parquet(report_dir / "predictions.parquet", index=False)
    candidates.to_csv(report_dir / "candidates.csv", index=False)
    prediction_manifest = {
        "schema_version": 1,
        "artifact_name": "production_predictions",
        "as_of": AS_OF,
        "model_id": MODEL_ID,
        "feature_hash": FEATURE_HASH,
        "prediction_count": 2,
        "config_hash": config_hash(config_path),
        "readiness": [
            {
                "gate": "universe_readiness_gate",
                "row_counts": {
                    "rows": 3,
                    "in_base_universe": 3,
                    "in_model_universe": 2,
                },
            },
            {
                "gate": "features_readiness_gate",
                "row_counts": {
                    "features": 3,
                    "universe": 3,
                    "eligible_universe": 2,
                    "eligible_after_hard_features": 2,
                },
                "missingness_summary": {"ret_1d": 0.0, "sparse": 0.5},
            },
        ],
    }
    candidate_manifest = {
        "schema_version": 1,
        "artifact_name": "production_candidates",
        "as_of": AS_OF,
        "model_id": MODEL_ID,
        "feature_hash": FEATURE_HASH,
        "prediction_count": 2,
        "candidate_count": 2,
        "config_hash": config_hash(config_path),
        "prediction_manifest": prediction_manifest,
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(prediction_manifest),
        encoding="utf-8",
    )
    (report_dir / "candidates_manifest.json").write_text(
        json.dumps(candidate_manifest),
        encoding="utf-8",
    )
    artifact_paths = [
        report_dir / "predictions.parquet",
        report_dir / "candidates.csv",
        report_dir / "manifest.json",
        report_dir / "candidates_manifest.json",
    ]
    production_summary = {
        "schema_version": 1,
        "artifact_name": "production_daily_summary",
        "run_id": "production-run",
        "as_of": AS_OF,
        "model_id": MODEL_ID,
        "candidate_count": 2,
        "artifacts": [str(path) for path in artifact_paths],
        "completed_time": "2024-01-03T10:00:00+00:00",
    }
    (report_dir / "production_summary.json").write_text(
        json.dumps(production_summary),
        encoding="utf-8",
    )
    artifact_paths.append(report_dir / "production_summary.json")

    observation_dir = reports / "performance_observation" / AS_OF
    observation = pd.DataFrame(
        {
            "observation_id": ["observation-1", "observation-2"],
            "signal_date": ["20231220", "20231220"],
            "observation_as_of": [AS_OF, AS_OF],
            "model_id": [MODEL_ID, MODEL_ID],
            "model_role": ["champion", "champion"],
            "horizon": [5, 5],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "prediction_score": [0.8, 0.2],
            "rank": [1, 2],
            "score_percentile": [1.0, 0.5],
            "future_excess_ret": [0.02, -0.01],
            "entry_date": ["20231221", "20231221"],
            "exit_date": ["20231228", "20231228"],
            "label_status": ["available", "available"],
            "feature_hash": [FEATURE_HASH, FEATURE_HASH],
            "universe_hash": ["universe-hash", "universe-hash"],
            "prediction_hash": ["prediction-hash", "prediction-hash"],
            "production_run_id": ["production-run", "production-run"],
            "shadow_run_id": ["shadow-run", "shadow-run"],
        },
        columns=list(OBSERVATION_COLUMNS),
    )
    observation_manifest = {
        "schema_version": 1,
        "artifact_name": "performance_observation",
        "observation_as_of": AS_OF,
        "observation_hash": logical_observation_hash(observation),
        "source_identity_hash": "observation-source",
        "row_count": 2,
        "available_rows": 2,
        "access_policy": "prospective_production",
        "model_lineage": [
            {
                "model_id": MODEL_ID,
                "model_role": "champion",
                "feature_hash": FEATURE_HASH,
                "universe_hash": "universe-hash",
                "source_models": [],
                "fusion_method": None,
            }
        ],
        "contracts": {
            "labels_used_only_after_maturity": True,
            "historical_predictions_used": False,
            "inference_called": False,
            "backtest_called": False,
            "paper_trading_called": False,
            "registry_modified": False,
        },
    }
    publish_observation_artifact(
        output_dir=observation_dir,
        observations=observation,
        metrics={"available_rows": 2},
        manifest=observation_manifest,
    )
    artifact_paths.extend(
        [
            observation_dir / "observation.parquet",
            observation_dir / "metrics.json",
            observation_dir / "manifest.json",
        ]
    )

    for portfolio_id, equity_values in (("alpha", (100.0, 90.0)), ("beta", (100.0, 110.0))):
        root = paper / portfolio_id
        root.mkdir(parents=True)
        account = {
            "portfolio_id": portfolio_id,
            "initial_cash": 100.0,
            "broker_connected": False,
        }
        (root / "account.json").write_text(json.dumps(account), encoding="utf-8")
        artifact_paths.append(root / "account.json")
        orders = pd.DataFrame(
            {
                "order_id": [f"{portfolio_id}-order"],
                "as_of": [AS_OF],
                "portfolio_id": [portfolio_id],
            }
        )
        trades = pd.DataFrame(
            {
                "trade_id": [f"{portfolio_id}-trade"],
                "as_of": [AS_OF],
                "portfolio_id": [portfolio_id],
                "status": ["filled"],
                "gross_value": [20.0],
                "cost": [1.0],
            }
        )
        positions = pd.DataFrame(
            {
                "event_id": [f"{portfolio_id}-p1", f"{portfolio_id}-p2"],
                "as_of": [AS_OF, AS_OF],
                "portfolio_id": [portfolio_id, portfolio_id],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "shares": [100, 100],
                "market_value": [45.0, 27.0] if portfolio_id == "alpha" else [40.0, 20.0],
            }
        )
        equity = pd.DataFrame(
            {
                "equity_id": [f"{portfolio_id}-e1", f"{portfolio_id}-e2"],
                "as_of": ["20240102", AS_OF],
                "portfolio_id": [portfolio_id, portfolio_id],
                "cash": [20.0, 18.0] if portfolio_id == "alpha" else [50.0, 50.0],
                "equity": list(equity_values),
                "nav": [value / 100.0 for value in equity_values],
                "daily_return": [0.0, equity_values[1] / equity_values[0] - 1.0],
                "drawdown": [0.0, min(equity_values[1] / equity_values[0] - 1.0, 0.0)],
            }
        )
        for name, frame in (
            ("orders", orders),
            ("trades", trades),
            ("positions", positions),
            ("equity_curve", equity),
        ):
            path = root / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            artifact_paths.append(path)
    return (
        MonitoringService(
            settings=settings,
            config_path=config_path,
            reports_root=reports,
            paper_root=paper,
        ),
        tuple(artifact_paths),
    )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
