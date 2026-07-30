from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.monitoring.performance.aggregation import aggregate_performance
from ashare_quant.monitoring.performance.schemas import (
    PERFORMANCE_METRIC_COLUMNS,
    PerformanceMonitorResult,
    PerformanceValidationResult,
)
from ashare_quant.monitoring.performance.service import PerformanceMonitoringService
from ashare_quant.monitoring.performance.validation import validate_observation_frame
from ashare_quant.monitoring.performance_observation.schemas import OBSERVATION_COLUMNS
from ashare_quant.monitoring.performance_observation.storage import (
    logical_observation_hash,
    publish_observation_artifact,
)

AS_OF = "20240209"
MODEL_ID = "challenger-h5"
FEATURE_HASH = "f" * 64
UNIVERSE_HASH = "u" * 64


def test_performance_metrics_ic_topn_decay_and_deciles() -> None:
    observations = observation_rows()
    lineage = model_lineage()

    metrics, details, warnings = aggregate_performance(observations, lineage)

    assert tuple(metrics.columns) == PERFORMANCE_METRIC_COLUMNS
    row = metrics.iloc[0]
    assert row["pearson_ic"] == pytest.approx(0.6)
    assert row["rank_ic"] == pytest.approx(0.6)
    expected_std = pd.Series([1.0] * 20 + [-1.0] * 5).std(ddof=1)
    assert row["icir"] == pytest.approx(0.6 / expected_std)
    assert row["positive_ic_ratio"] == pytest.approx(0.8)
    assert row["rolling_20_ic_mean"] == pytest.approx(0.5)
    assert row["alpha_decay_ratio"] == pytest.approx(0.5 / 0.6)
    assert row["top10_average_excess_ret"] == pytest.approx(0.333)
    assert row["top10_hit_rate"] == pytest.approx(0.8)
    assert row["top10_decay_ratio"] == pytest.approx(0.2775 / 0.333)
    assert row["decile_monotonicity"] == pytest.approx(1.0)
    assert details["daily_ic_rows"] == 25
    assert any("window=60" in warning for warning in warnings)


def test_performance_validation_rejects_schema_role_immaturity_and_duplicates() -> None:
    base = observation_rows().head(2)
    missing = base.drop(columns=["prediction_hash"])
    with pytest.raises(DataValidationError, match="required columns"):
        validate_observation_frame(missing, AS_OF)

    bad_role = base.assign(model_role="unknown")
    with pytest.raises(DataValidationError, match="unsupported model_role"):
        validate_observation_frame(bad_role, AS_OF)

    immature = base.assign(label_status="immature", future_excess_ret=None)
    with pytest.raises(DataValidationError, match="immature"):
        validate_observation_frame(immature, AS_OF)

    duplicated = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicated"):
        validate_observation_frame(duplicated, AS_OF)


def test_empty_observation_batch_is_valid_with_insufficient_history_warning(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: performance-monitor\n", encoding="utf-8")
    rows = pd.DataFrame(columns=list(OBSERVATION_COLUMNS))
    output_dir = reports / "performance_observation" / AS_OF
    manifest = {
        "schema_version": 1,
        "artifact_name": "performance_observation",
        "observation_as_of": AS_OF,
        "observation_hash": logical_observation_hash(rows),
        "source_identity_hash": f"source-{AS_OF}",
        "row_count": 0,
        "available_rows": 0,
        "access_policy": "prospective_production",
        "model_lineage": [],
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
        output_dir=output_dir,
        observations=rows,
        metrics={"available_rows": 0},
        manifest=manifest,
    )
    service = PerformanceMonitoringService(reports_root=reports, config_path=config)

    built = service.build(AS_OF)

    assert built.metrics.empty
    assert tuple(built.metrics.columns) == PERFORMANCE_METRIC_COLUMNS
    assert built.manifest["models"] == []
    assert built.manifest["row_counts"]["observations"] == 0
    assert built.summary["warnings"] == [
        "insufficient observations: no mature performance observations"
    ]


def test_nonempty_observation_batch_still_requires_model_lineage(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: performance-monitor\n", encoding="utf-8")
    rows = observation_rows().head(2)
    output_dir = reports / "performance_observation" / AS_OF
    manifest = {
        "schema_version": 1,
        "artifact_name": "performance_observation",
        "observation_as_of": AS_OF,
        "observation_hash": logical_observation_hash(rows),
        "source_identity_hash": f"source-{AS_OF}",
        "row_count": len(rows),
        "available_rows": len(rows),
        "access_policy": "prospective_production",
        "model_lineage": [],
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
        output_dir=output_dir,
        observations=rows,
        metrics={"available_rows": len(rows)},
        manifest=manifest,
    )
    service = PerformanceMonitoringService(reports_root=reports, config_path=config)

    with pytest.raises(DataValidationError, match="non-empty.*model_lineage"):
        service.build(AS_OF)


def test_performance_service_deterministic_manifest_last_and_status(tmp_path: Path) -> None:
    service = performance_fixture(tmp_path)

    first = service.run(AS_OF)
    before = {path.name: path.read_bytes() for path in first.output_dir.iterdir() if path.is_file()}
    second = service.run(AS_OF)
    status = service.status(AS_OF)

    assert second.idempotent
    assert status.valid
    assert status.exists
    assert before == {
        path.name: path.read_bytes() for path in second.output_dir.iterdir() if path.is_file()
    }
    manifest = read_json(first.output_dir / "manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["artifact_name"] == "performance_monitor"
    assert manifest["labels_read"] is False
    assert manifest["models"][0]["model_role"] == "challenger_h5"
    assert manifest["models"][0]["horizon"] == 5


def test_performance_source_hash_mismatch_and_duplicate_history_fail(tmp_path: Path) -> None:
    service = performance_fixture(tmp_path)
    source = tmp_path / "reports" / "performance_observation" / AS_OF
    (source / "metrics.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(DataValidationError, match="metrics hash mismatch"):
        service.build(AS_OF)

    tmp_path_2 = tmp_path / "duplicate"
    service = performance_fixture(tmp_path_2)
    duplicate_date = "20240210"
    publish_source(
        service.reports_root / "performance_observation" / duplicate_date,
        observation_rows(),
        duplicate_date,
    )
    with pytest.raises(DataValidationError, match="identities are duplicated"):
        service.build(duplicate_date)


def test_failed_performance_publication_does_not_overwrite_previous(
    tmp_path: Path,
) -> None:
    service = performance_fixture(tmp_path)
    result = service.run(AS_OF)
    before = (result.output_dir / "manifest.json").read_bytes()
    source = tmp_path / "reports" / "performance_observation" / AS_OF / "metrics.json"
    source.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(DataValidationError, match="metrics hash mismatch"):
        service.run(AS_OF)
    assert (result.output_dir / "manifest.json").read_bytes() == before


def test_performance_manifest_is_written_last_and_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = performance_fixture(tmp_path)
    writes: list[str] = []
    from ashare_quant.monitoring.performance import service as service_module

    original_write = service_module.atomic_write_json

    def tracked_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(path.name)
        original_write(path, payload)

    monkeypatch.setattr(service_module, "atomic_write_json", tracked_write)
    result = service.run(AS_OF)
    assert writes[-1] == "manifest.json"
    assert result.output_dir.is_dir()

    failed_root = tmp_path / "failed"
    failed_service = performance_fixture(failed_root)

    def fail_manifest(path: Path, payload: dict[str, Any]) -> None:
        if path.name == "manifest.json":
            raise OSError("manifest failure")
        original_write(path, payload)

    monkeypatch.setattr(service_module, "atomic_write_json", fail_manifest)
    with pytest.raises(OSError, match="manifest failure"):
        failed_service.run(AS_OF)
    assert not failed_service.output_dir(AS_OF).exists()
    assert not list(failed_service.output_dir(AS_OF).parent.glob(".performance-*"))


def test_performance_monitor_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = performance_fixture(tmp_path)
    original_read = pd.read_parquet

    def guarded_read(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        text = str(path)
        assert "labels" not in text
        assert "features" not in text
        assert "daily" not in text
        assert "backtest" not in text
        return original_read(path, *args, **kwargs)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("prohibited service called")

    monkeypatch.setattr(pd, "read_parquet", guarded_read)
    monkeypatch.setattr(
        "ashare_quant.models.inference.ProductionInferenceEngine.predict",
        forbidden,
    )
    monkeypatch.setattr("ashare_quant.models.inference.score_registered_model_range", forbidden)
    monkeypatch.setattr("ashare_quant.backtest.engine.simulate_portfolio", forbidden)
    monkeypatch.setattr(
        "ashare_quant.backtest.historical.HistoricalBacktestEngine.run",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.strategy.candidate_selector.CandidateSelector.select",
        forbidden,
    )
    monkeypatch.setattr(
        "ashare_quant.paper_trading.service.PaperTradingService.rebalance",
        forbidden,
    )
    monkeypatch.setattr("ashare_quant.paper_trading.service.PaperTradingService.execute", forbidden)
    monkeypatch.setattr("ashare_quant.models.registry.ModelRegistry.promote_model", forbidden)

    service.run(AS_OF)


def test_performance_cli_success_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Service:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, as_of: str) -> PerformanceMonitorResult:
            return PerformanceMonitorResult(as_of, tmp_path / "out", 2, 100)

        def validate(self, as_of: str) -> PerformanceValidationResult:
            return PerformanceValidationResult(as_of, False, False, 0, 0, error="invalid")

        def status(self, as_of: str) -> PerformanceValidationResult:
            return PerformanceValidationResult(as_of, True, True, 2, 100)

    monkeypatch.setattr("ashare_quant.cli.PerformanceMonitoringService", Service)
    prefix = ["--config", "config/default.yaml", "monitor"]
    assert main([*prefix, "performance", "--as-of", AS_OF]) == 0
    assert "performance_monitor: as_of=20240209" in capsys.readouterr().out
    assert main([*prefix, "performance-status", "--as-of", AS_OF]) == 0
    assert "valid=True" in capsys.readouterr().out
    assert main([*prefix, "performance-validate", "--as-of", AS_OF]) == 2
    assert "error=invalid" in capsys.readouterr().err


def performance_fixture(tmp_path: Path) -> PerformanceMonitoringService:
    tmp_path.mkdir(parents=True, exist_ok=True)
    reports = tmp_path / "reports"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: performance-monitor\n", encoding="utf-8")
    publish_source(reports / "performance_observation" / AS_OF, observation_rows(), AS_OF)
    return PerformanceMonitoringService(reports_root=reports, config_path=config)


def publish_source(output_dir: Path, rows: pd.DataFrame, as_of: str) -> None:
    manifest = {
        "schema_version": 1,
        "artifact_name": "performance_observation",
        "observation_as_of": as_of,
        "observation_hash": logical_observation_hash(rows),
        "source_identity_hash": f"source-{as_of}",
        "row_count": len(rows),
        "available_rows": int(rows["label_status"].eq("available").sum()),
        "access_policy": "prospective_production",
        "model_lineage": list(model_lineage().values()),
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
        output_dir=output_dir,
        observations=rows,
        metrics={"available_rows": len(rows)},
        manifest=manifest,
    )


def model_lineage() -> dict[str, dict[str, Any]]:
    return {
        MODEL_ID: {
            "model_id": MODEL_ID,
            "model_role": "challenger_h5",
            "feature_hash": FEATURE_HASH,
            "universe_hash": UNIVERSE_HASH,
            "source_models": [],
            "fusion_method": None,
        }
    }


def observation_rows() -> pd.DataFrame:
    dates = pd.bdate_range("20240101", periods=25).strftime("%Y%m%d")
    rows: list[dict[str, object]] = []
    for day_index, signal_date in enumerate(dates):
        direction = 1.0 if day_index < 20 else -1.0
        for value in range(1, 61):
            code = f"{value:06d}.SZ"
            rows.append(
                {
                    "observation_id": f"{MODEL_ID}-{signal_date}-{code}",
                    "signal_date": signal_date,
                    "observation_as_of": AS_OF,
                    "model_id": MODEL_ID,
                    "model_role": "challenger_h5",
                    "horizon": 5,
                    "ts_code": code,
                    "prediction_score": float(value),
                    "rank": 61 - value,
                    "score_percentile": value / 60.0,
                    "future_excess_ret": direction * value / 100.0,
                    "entry_date": signal_date,
                    "exit_date": signal_date,
                    "label_status": "available",
                    "feature_hash": FEATURE_HASH,
                    "universe_hash": UNIVERSE_HASH,
                    "prediction_hash": f"prediction-{signal_date}",
                    "production_run_id": f"production-{signal_date}",
                    "shadow_run_id": f"shadow-{signal_date}",
                }
            )
    return pd.DataFrame.from_records(rows, columns=list(OBSERVATION_COLUMNS))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
