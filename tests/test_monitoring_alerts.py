from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config.settings import AppSettings, MonitoringAlertSettings, PathSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.monitoring.alerts.evaluator import (
    deterministic_alert_id,
    evaluate_alerts,
)
from ashare_quant.monitoring.alerts.lifecycle import append_history, apply_lifecycle
from ashare_quant.monitoring.alerts.rules import configured_rules
from ashare_quant.monitoring.alerts.schemas import (
    AlertEvaluationResult,
    AlertMonitorResult,
    AlertSeverity,
    AlertState,
    AlertValidationResult,
)
from ashare_quant.monitoring.alerts.service import AlertService
from ashare_quant.monitoring.alerts.storage import (
    ALERT_HISTORY_COLUMNS,
    validate_alert_payload,
)
from ashare_quant.utils.manifest import atomic_write_json

AS_OF = "20240209"
MODEL_ID = "model-h5"
PORTFOLIO_ID = "paper-top20"


def test_alert_identity_is_deterministic_and_environment_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = deterministic_alert_id("alpha_decay", MODEL_ID, None, "alpha_decay_ratio_h5")
    monkeypatch.setenv("HOSTNAME", "different-host")
    monkeypatch.setenv("PYTHONHASHSEED", "999")
    second = deterministic_alert_id("alpha_decay", MODEL_ID, None, "alpha_decay_ratio_h5")

    assert first == second
    assert len(first) == 64


def test_alert_schema_and_enum_validation() -> None:
    payload = {
        "schema_version": 1,
        "artifact_name": "monitoring_alerts",
        "as_of": AS_OF,
        "alerts": [],
        "warnings": [],
    }
    validate_alert_payload(payload)
    assert AlertSeverity("WARNING") is AlertSeverity.WARNING
    with pytest.raises(ValueError):
        AlertSeverity("INVALID")

    invalid = {
        **payload,
        "alerts": [{"alert_id": "invalid"}],
    }
    with pytest.raises(DataValidationError, match="lacks fields"):
        validate_alert_payload(invalid)


def test_required_alert_rules_warning_and_critical_detection() -> None:
    health, performance, portfolios = metric_frames()

    result = evaluate_alerts(
        health=health,
        performance=performance,
        portfolios=portfolios,
        rules=configured_rules(MonitoringAlertSettings()),
        source_artifact_hash="source-hash",
    )
    by_metric = {candidate.metric_name: candidate for candidate in result.candidates}

    assert by_metric["alpha_decay_ratio_h5"].severity is AlertSeverity.WARNING
    assert by_metric["rank_ic_delta_h5"].severity is AlertSeverity.CRITICAL
    assert by_metric["score_std"].severity is AlertSeverity.CRITICAL
    assert by_metric["unique_score_ratio"].severity is AlertSeverity.CRITICAL
    assert by_metric["maximum_feature_psi"].severity is AlertSeverity.CRITICAL
    assert by_metric["maximum_feature_ks"].severity is AlertSeverity.CRITICAL
    assert by_metric["maximum_missing_ratio_drift"].severity is AlertSeverity.CRITICAL
    assert by_metric["prediction_coverage"].severity is AlertSeverity.CRITICAL
    assert by_metric["current_drawdown"].severity is AlertSeverity.WARNING
    assert by_metric["max_drawdown"].severity is AlertSeverity.CRITICAL
    assert by_metric["max_position_weight"].severity is AlertSeverity.WARNING
    assert by_metric["top5_concentration"].severity is AlertSeverity.CRITICAL
    assert by_metric["industry_concentration"].severity is AlertSeverity.WARNING
    assert by_metric["rejected_order_ratio"].severity is AlertSeverity.WARNING
    assert by_metric["failed_execution_ratio"].severity is AlertSeverity.CRITICAL


def test_optional_metrics_warn_without_false_recovery() -> None:
    health, performance, portfolios = metric_frames()
    health["drift_reference"] = None
    portfolios["industry_concentration"] = None

    result = evaluate_alerts(
        health=health,
        performance=performance,
        portfolios=portfolios,
        rules=configured_rules(MonitoringAlertSettings()),
        source_artifact_hash="source-hash",
    )

    assert any("maximum_feature_psi" in warning for warning in result.warnings)
    industry_id = deterministic_alert_id(
        "concentration_risk",
        None,
        PORTFOLIO_ID,
        "industry_concentration",
    )
    assert industry_id not in result.evaluated_alert_ids


def test_alert_lifecycle_new_active_recovered_and_no_duplicates() -> None:
    alert_id = deterministic_alert_id("model_alpha_decay", MODEL_ID, None, "alpha_decay_ratio_h5")
    candidate = _candidate_evaluation(alert_id)
    first = apply_lifecycle(candidate, _empty_history(), "20240201")
    history = append_history(_empty_history(), first)
    second = apply_lifecycle(candidate, history, "20240202")
    history = append_history(history, second)
    recovered_evaluation = AlertEvaluationResult((), frozenset({alert_id}), ())
    recovered = apply_lifecycle(recovered_evaluation, history, "20240205")
    history = append_history(history, recovered)

    assert first[0].status is AlertState.NEW
    assert second[0].status is AlertState.ACTIVE
    assert second[0].first_seen == "20240201"
    assert recovered[0].status is AlertState.RECOVERED
    assert recovered[0].severity is AlertSeverity.INFO
    assert len(history) == 3
    assert not history.duplicated(["alert_id", "last_seen"]).any()


def test_alert_service_atomic_deterministic_append_only_and_manifest_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = alert_fixture(tmp_path, AS_OF)
    from ashare_quant.monitoring.alerts import storage as storage_module

    writes: list[str] = []
    original_write = storage_module.atomic_write_json

    def tracked_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(path.name)
        original_write(path, payload)

    monkeypatch.setattr(storage_module, "atomic_write_json", tracked_write)
    first = service.run(AS_OF)
    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir() if path.is_file()}
    second = service.run(AS_OF)

    assert writes[-1] == "manifest.json"
    assert second.idempotent
    assert before == {
        path.name: path.read_bytes() for path in second.output_dir.iterdir() if path.is_file()
    }
    history = pd.read_parquet(service.history_path)
    assert tuple(history.columns) == ALERT_HISTORY_COLUMNS
    assert not history.duplicated(["alert_id", "last_seen"]).any()
    manifest = _json(first.output_dir / "manifest.json")
    assert manifest["labels_read"] is False
    assert manifest["models_modified"] is False
    assert manifest["trading_modified"] is False


def test_alert_source_hash_mismatch_and_failed_publication_keep_prior_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = alert_fixture(tmp_path, AS_OF)
    result = service.run(AS_OF)
    before_manifest = (result.output_dir / "manifest.json").read_bytes()
    performance_path = (
        tmp_path
        / "reports"
        / "model_monitor"
        / AS_OF
        / "performance"
        / "performance_metrics.parquet"
    )
    changed = pd.read_parquet(performance_path)
    changed.loc[0, "alpha_decay_ratio"] = 0.01
    changed.to_parquet(performance_path, index=False)

    with pytest.raises(DataValidationError, match="source hash mismatch"):
        service.run(AS_OF)
    assert (result.output_dir / "manifest.json").read_bytes() == before_manifest

    failed_root = tmp_path / "failed"
    failed_service = alert_fixture(failed_root, AS_OF)

    def fail_replace(*args: object, **kwargs: object) -> None:
        raise OSError("publication failure")

    monkeypatch.setattr(
        "ashare_quant.monitoring.alerts.storage.replace_targets_atomically",
        fail_replace,
    )
    with pytest.raises(OSError, match="publication failure"):
        failed_service.run(AS_OF)
    assert not failed_service.output_dir(AS_OF).exists()
    assert not failed_service.history_path.exists()


def test_alert_engine_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = alert_fixture(tmp_path, AS_OF)
    original_read = pd.read_parquet

    def guarded_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        text = str(path)
        assert "labels" not in text
        assert "features" not in text
        assert "backtest" not in text
        assert "challenger_evaluation" not in text
        return original_read(path, *args, **kwargs)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("prohibited service called")

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    monkeypatch.setattr(
        "ashare_quant.models.inference.ProductionInferenceEngine.predict",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.backtest.historical.HistoricalBacktestEngine.run",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.paper_trading.service.PaperTradingService.execute",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.paper_trading.service.PaperTradingService.rebalance",
        forbidden,
    )
    monkeypatch.setattr("ashare_quant.models.registry.ModelRegistry.promote_model", forbidden)

    service.run(AS_OF)


def test_alert_cli_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Service:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, as_of: str) -> AlertMonitorResult:
            return AlertMonitorResult(as_of, tmp_path / "alerts", 3, 1)

        def validate(self, as_of: str) -> AlertValidationResult:
            return AlertValidationResult(as_of, False, False, 0, error="invalid")

        def status(self, as_of: str) -> AlertValidationResult:
            return AlertValidationResult(as_of, True, True, 3)

    monkeypatch.setattr("ashare_quant.cli.AlertService", Service)
    prefix = ["--config", "config/default.yaml", "monitor"]
    assert main([*prefix, "alerts", "--as-of", AS_OF]) == 0
    assert "monitor_alerts: as_of=20240209" in capsys.readouterr().out
    assert main([*prefix, "alerts-status", "--as-of", AS_OF]) == 0
    assert "valid=True" in capsys.readouterr().out
    assert main([*prefix, "alerts-validate", "--as-of", AS_OF]) == 2
    assert "error=invalid" in capsys.readouterr().err


def alert_fixture(tmp_path: Path, as_of: str) -> AlertService:
    reports = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path.write_text("project_name: alerts\n", encoding="utf-8")
    health, performance, portfolios = metric_frames()
    health["as_of"] = as_of
    root = reports / "model_monitor" / as_of
    performance_root = root / "performance"
    performance_root.mkdir(parents=True)
    atomic_write_json(root / "health.json", health)
    performance.to_parquet(performance_root / "performance_metrics.parquet", index=False)
    portfolios.to_parquet(root / "portfolio_metrics.parquet", index=False)
    performance_manifest = {
        "schema_version": 1,
        "artifact_name": "performance_monitor",
        "as_of": as_of,
        "status": "success",
        "metrics_file_sha256": file_sha256(performance_root / "performance_metrics.parquet"),
    }
    atomic_write_json(performance_root / "manifest.json", performance_manifest)
    monitor_manifest = {
        "schema_version": 1,
        "artifact_name": "production_monitor_manifest",
        "as_of": as_of,
        "monitor_metric_file_hashes": {
            "health": file_sha256(root / "health.json"),
            "performance_metrics": file_sha256(performance_root / "performance_metrics.parquet"),
            "performance_manifest": file_sha256(performance_root / "manifest.json"),
            "portfolio_metrics": file_sha256(root / "portfolio_metrics.parquet"),
        },
    }
    atomic_write_json(root / "manifest.json", monitor_manifest)
    settings = AppSettings(paths=PathSettings(reports=reports))
    return AlertService(settings=settings, config_path=config_path, reports_root=reports)


def metric_frames() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    health = {
        "as_of": AS_OF,
        "model_id": MODEL_ID,
        "universe_size": 100,
        "model_universe_size": 100,
        "prediction_count": 50,
        "score_std": 0.0001,
        "unique_score_ratio": 0.40,
        "drift_reference": {
            "metrics": {
                "maximum_feature_psi": 0.30,
                "maximum_feature_ks": 0.25,
                "maximum_missing_ratio_drift": 0.30,
            }
        },
    }
    performance = pd.DataFrame(
        {
            "model_id": [MODEL_ID],
            "model_role": ["challenger_h5"],
            "horizon": [5],
            "rank_ic": [0.10],
            "rolling_20_ic_mean": [0.0],
            "alpha_decay_ratio": [0.60],
        }
    )
    portfolios = pd.DataFrame(
        {
            "portfolio_id": [PORTFOLIO_ID],
            "drawdown": [-0.15],
            "max_drawdown": [-0.25],
            "max_position_weight": [0.15],
            "top5_concentration": [0.70],
            "industry_concentration": [0.40],
            "rejected_order_ratio": [0.15],
            "failed_execution_ratio": [0.20],
        }
    )
    return health, performance, portfolios


def _candidate_evaluation(alert_id: str) -> AlertEvaluationResult:
    from ashare_quant.monitoring.alerts.schemas import AlertCandidate

    candidate = AlertCandidate(
        alert_id=alert_id,
        alert_type="model_alpha_decay",
        severity=AlertSeverity.CRITICAL,
        model_id=MODEL_ID,
        portfolio_id=None,
        metric_name="alpha_decay_ratio_h5",
        metric_value=0.40,
        threshold=0.50,
        source_artifact_hash="source-hash",
    )
    return AlertEvaluationResult((candidate,), frozenset({alert_id}), ())


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=list(ALERT_HISTORY_COLUMNS))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
