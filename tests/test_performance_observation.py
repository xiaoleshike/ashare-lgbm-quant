from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.labels.storage import LABEL_COLUMNS
from ashare_quant.monitoring.performance_observation.maturity import (
    maturity_dates,
    open_sessions,
    require_observation_session,
)
from ashare_quant.monitoring.performance_observation.schemas import (
    OBSERVATION_COLUMNS,
    OBSERVATION_KEY,
    PerformanceObservationResult,
)
from ashare_quant.monitoring.performance_observation.service import (
    PerformanceObservationService,
)
from ashare_quant.monitoring.performance_observation.storage import (
    logical_observation_hash,
    read_observation_artifact,
)
from ashare_quant.monitoring.performance_observation.validation import (
    _validate_shadow_manifest,
)

SIGNAL_DATE = "20240105"
FEATURE_HASH = "feature-hash"
UNIVERSE_HASH = "universe-hash"
PREDICTION_HASH = "prediction-hash"
PRODUCTION_RUN_ID = "production-run"
SHADOW_RUN_ID = "shadow-run"


def test_maturity_uses_open_sessions_across_weekend_and_holidays() -> None:
    calendar = pd.DataFrame(
        {
            "cal_date": [
                "20240105",
                "20240106",
                "20240107",
                "20240108",
                "20240109",
                "20240110",
                "20240111",
                "20240112",
                "20240115",
            ],
            "is_open": [1, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )
    sessions = open_sessions(calendar)

    entry, exit_date = maturity_dates(sessions, SIGNAL_DATE, 5)

    assert entry == "20240108"
    assert exit_date == "20240115"
    with pytest.raises(DataValidationError, match="not an open"):
        require_observation_session(sessions, "20240107")


def test_calendar_without_future_horizon_sessions_skips_immature_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    calendar_path = next((tmp_path / "raw" / "trade_cal").glob("**/*.parquet"))
    calendar = pd.read_parquet(calendar_path)
    calendar = calendar.loc[calendar["cal_date"].astype(str) <= sessions[6]]
    calendar.to_parquet(calendar_path, index=False)

    result = service.run(sessions[6])
    observations = pd.read_parquet(result.output_dir / "observation.parquet")

    assert set(observations["horizon"]) == {5}


def test_immature_horizons_do_not_read_labels_or_create_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, shadow = observation_fixture(tmp_path, monkeypatch)
    requested_horizons: list[int | None] = []
    original_read = service.label_store.read

    def tracked_read(
        start_date: str | None = None,
        end_date: str | None = None,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        requested_horizons.append(horizon)
        return original_read(start_date, end_date, horizon)

    monkeypatch.setattr(service.label_store, "read", tracked_read)
    as_of = sessions[6]
    result = service.run(as_of)
    observations = pd.read_parquet(result.output_dir / "observation.parquet")

    assert requested_horizons == [5]
    assert set(observations["horizon"]) == {5}
    expected_models = {
        "champion-model",
        "challenger-h5",
        "ensemble-model",
    }
    assert set(observations["model_id"]) == expected_models
    assert len(observations) == 6
    assert shadow["trade_date"].max() == SIGNAL_DATE


def test_observation_identity_schema_metrics_and_maturity_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)

    result = service.run(sessions[61])

    observations = pd.read_parquet(result.output_dir / "observation.parquet")
    assert tuple(observations.columns) == OBSERVATION_COLUMNS
    assert not observations.duplicated(list(OBSERVATION_KEY)).any()
    assert not observations["observation_id"].duplicated().any()
    assert len(observations) == 18
    assert result.available_rows == 18
    assert set(observations["horizon"]) == {5, 10, 20, 60}
    assert (observations["exit_date"] <= sessions[61]).all()
    assert set(observations["feature_hash"]) == {FEATURE_HASH}
    assert set(observations["universe_hash"]) == {UNIVERSE_HASH}
    assert set(observations["production_run_id"]) == {PRODUCTION_RUN_ID}
    assert set(observations["shadow_run_id"]) == {SHADOW_RUN_ID}
    metrics = _json(result.output_dir / "metrics.json")
    assert metrics["available_rows"] == 18
    assert len(metrics["models"]) == 9
    assert all("rank_ic" in item for item in metrics["models"])
    assert all("rolling" in item for item in metrics["models"])
    manifest = _json(result.output_dir / "manifest.json")
    assert manifest["access_policy"] == "prospective_production"
    assert manifest["contracts"]["labels_used_only_after_maturity"] is True
    assert manifest["contracts"]["inference_called"] is False
    assert manifest["contracts"]["backtest_called"] is False
    assert manifest["contracts"]["paper_trading_called"] is False
    lineage = {item["model_id"]: item for item in manifest["model_lineage"]}
    assert lineage["ensemble-model"]["source_models"] == [
        "challenger-h10",
        "challenger-h20",
        "challenger-h5",
        "challenger-h60",
    ]
    assert lineage["ensemble-model"]["fusion_method"] == "percentile_mean"


def test_unavailable_label_is_preserved_without_zero_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    mask = labels["ts_code"].eq("000002.SZ") & labels["horizon"].eq(5)
    labels.loc[mask, "future_excess_ret"] = None
    labels.loc[mask, "stock_forward_ret"] = None
    labels.loc[mask, "benchmark_forward_ret"] = None
    labels.loc[mask, "is_label_available"] = False
    labels.loc[mask, "label_unavailable_reason"] = "entry_not_buyable"
    labels.to_parquet(label_path, index=False)

    result = service.run(sessions[6])
    observations = pd.read_parquet(result.output_dir / "observation.parquet")
    unavailable = observations.loc[observations["ts_code"].eq("000002.SZ")]

    assert set(unavailable["label_status"]) == {"entry_not_buyable"}
    assert unavailable["future_excess_ret"].isna().all()


def test_append_only_increment_and_deterministic_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    first = service.run(sessions[6])
    first_bytes = {
        path.name: path.read_bytes() for path in first.output_dir.iterdir() if path.is_file()
    }

    rerun = service.run(sessions[6])
    second = service.run(sessions[11])
    second_rows = pd.read_parquet(second.output_dir / "observation.parquet")

    assert rerun.idempotent
    assert {
        path.name: path.read_bytes() for path in rerun.output_dir.iterdir() if path.is_file()
    } == first_bytes
    assert set(second_rows["horizon"]) == {10}
    assert len(second_rows) == 4
    history_keys = pd.concat(
        [
            pd.read_parquet(first.output_dir / "observation.parquet"),
            second_rows,
        ],
        ignore_index=True,
    )
    assert not history_keys.duplicated(list(OBSERVATION_KEY)).any()


def test_later_calendar_and_immature_label_changes_do_not_change_old_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    first = service.run(sessions[6])
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    labels.loc[labels["horizon"].eq(60), "future_excess_ret"] = 7.0
    labels.to_parquet(label_path, index=False)

    rerun = service.run(sessions[6])

    assert rerun.idempotent
    assert rerun.manifest_path.read_bytes() == first.manifest_path.read_bytes()


def test_same_identity_with_changed_label_is_hard_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    result = service.run(sessions[6])
    before = (result.output_dir / "manifest.json").read_bytes()
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    mask = labels["ts_code"].eq("000001.SZ") & labels["horizon"].eq(5)
    labels.loc[mask, "future_excess_ret"] = 9.0
    labels.to_parquet(label_path, index=False)

    with pytest.raises(DataValidationError, match="different immutable content"):
        service.run(sessions[6])

    assert (result.output_dir / "manifest.json").read_bytes() == before


def test_missing_mature_label_is_failure_not_zero_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    label_path = next((tmp_path / "processed" / "labels_forward").glob("**/*.parquet"))
    labels = pd.read_parquet(label_path)
    labels = labels.loc[~(labels["ts_code"].eq("000001.SZ") & labels["horizon"].eq(5))]
    labels.to_parquet(label_path, index=False)

    with pytest.raises(DataValidationError, match="labels are missing"):
        service.run(sessions[6])

    assert not (tmp_path / "reports" / "performance_observation" / sessions[6]).exists()


def test_forbidden_sources_and_access_policies_are_rejected() -> None:
    base = {
        "schema_version": 1,
        "artifact_name": "shadow_prediction_bundle",
        "models": [
            {
                "model_id": "model",
                "access_policy": "frozen_oos_evaluation",
            }
        ],
    }
    with pytest.raises(DataValidationError, match="non-prospective"):
        _validate_shadow_manifest(base, SIGNAL_DATE)
    historical = {
        **base,
        "models": [
            {
                "model_id": "model",
                "access_policy": "prospective_production",
                "source": "reports/ensemble_evaluation/run",
            }
        ],
    }
    with pytest.raises(DataValidationError, match="historical evaluation"):
        _validate_shadow_manifest(historical, SIGNAL_DATE)


def test_observation_never_calls_inference_backtest_or_paper_trading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("prohibited service was called")

    monkeypatch.setattr("ashare_quant.models.inference.score_registered_model_range", forbidden)
    monkeypatch.setattr("ashare_quant.backtest.engine.simulate_portfolio", forbidden)
    monkeypatch.setattr("ashare_quant.paper_trading.service.PaperTradingService.execute", forbidden)

    service.run(sessions[6])


def test_atomic_failure_does_not_publish_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "ashare_quant.monitoring.performance_observation.storage.atomic_write_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest failure")),
    )

    with pytest.raises(OSError, match="manifest failure"):
        service.run(sessions[6])

    output = tmp_path / "reports" / "performance_observation" / sessions[6]
    assert not output.exists()
    assert not list(output.parent.glob(f".{sessions[6]}.*"))


def test_observation_artifact_hash_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, sessions, _ = observation_fixture(tmp_path, monkeypatch)
    result = service.run(sessions[6])
    artifact = read_observation_artifact(result.output_dir)
    assert artifact is not None
    frame, manifest = cast(tuple[pd.DataFrame, dict[str, Any]], artifact)
    assert logical_observation_hash(frame) == manifest["observation_hash"]
    frame.loc[0, "prediction_score"] = 999.0
    frame.to_parquet(result.output_dir / "observation.parquet", index=False)
    with pytest.raises(DataValidationError, match="Parquet hash mismatch"):
        read_observation_artifact(result.output_dir)


def test_observation_cli_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SuccessfulService:
        def __init__(self, **_: object) -> None:
            pass

        def run(self, as_of: str) -> PerformanceObservationResult:
            output = tmp_path / "reports" / "performance_observation" / as_of
            return PerformanceObservationResult(as_of, 10, 8, output, output / "manifest.json")

    monkeypatch.setattr("ashare_quant.cli.PerformanceObservationService", SuccessfulService)
    args = [
        "--config",
        "config/default.yaml",
        "monitor",
        "observe",
        "--as-of",
        "20240115",
    ]
    assert main(args) == 0
    assert "performance_observation: as_of=20240115 rows=10" in capsys.readouterr().out

    class FailingService(SuccessfulService):
        def run(self, as_of: str) -> PerformanceObservationResult:
            raise DataValidationError("immature")

    monkeypatch.setattr("ashare_quant.cli.PerformanceObservationService", FailingService)
    assert main(args) == 2
    assert "performance observation failed: immature" in capsys.readouterr().err


def observation_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[PerformanceObservationService, tuple[str, ...], pd.DataFrame]:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    reports_root = tmp_path / "reports"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project_name: observation-fixture\n", encoding="utf-8")
    business_dates = pd.bdate_range(SIGNAL_DATE, periods=90)
    sessions = tuple(date.strftime("%Y%m%d") for date in business_dates)
    calendar_path = raw_root / "trade_cal" / "year=2024" / "month=01" / "data.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "exchange": ["SSE"] * len(sessions),
            "cal_date": sessions,
            "is_open": [1] * len(sessions),
        }
    ).to_parquet(calendar_path, index=False)
    labels = _labels(sessions)
    label_path = processed_root / "labels_forward" / "year=2024" / "month=01" / "data.parquet"
    label_path.parent.mkdir(parents=True)
    labels.to_parquet(label_path, index=False)
    shadow = _shadow_predictions()
    shadow_manifest = {
        "schema_version": 1,
        "artifact_name": "shadow_prediction_bundle",
        "production_run_id": PRODUCTION_RUN_ID,
        "shadow_run_id": SHADOW_RUN_ID,
        "prediction_hash": PREDICTION_HASH,
        "feature_hash": FEATURE_HASH,
        "universe_hash": UNIVERSE_HASH,
        "models": [
            {
                "model_id": model_id,
                "model_role": role,
                "access_policy": "prospective_production",
            }
            for model_id, role in shadow[["model_id", "model_role"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ],
    }
    ensemble = next(
        item for item in shadow_manifest["models"] if item["model_id"] == "ensemble-model"
    )
    ensemble["source_models"] = [
        "challenger-h5",
        "challenger-h10",
        "challenger-h20",
        "challenger-h60",
    ]
    ensemble["fusion_method"] = "percentile_mean"
    monkeypatch.setattr(
        "ashare_quant.monitoring.performance_observation.service.load_shadow_sources",
        lambda **kwargs: (
            shadow.copy(),
            [{**shadow_manifest, "source_signal_date": SIGNAL_DATE}],
            {SIGNAL_DATE: "shadow-file-hash"},
        ),
    )
    service = PerformanceObservationService(
        raw_root=raw_root,
        processed_root=processed_root,
        reports_root=reports_root,
        config_path=config_path,
        runs_root=tmp_path / "runs",
    )
    return service, sessions, shadow


def _shadow_predictions() -> pd.DataFrame:
    models = (
        ("champion-model", "champion", 5),
        ("challenger-h5", "challenger_h5", 5),
        ("challenger-h10", "challenger_h10", 10),
        ("challenger-h20", "challenger_h20", 20),
        ("challenger-h60", "challenger_h60", 60),
        ("ensemble-model", "multi_horizon_ensemble", None),
    )
    rows: list[dict[str, object]] = []
    for model_id, role, horizon in models:
        for rank, (code, score) in enumerate((("000001.SZ", 0.8), ("000002.SZ", 0.2)), start=1):
            rows.append(
                {
                    "trade_date": SIGNAL_DATE,
                    "ts_code": code,
                    "model_id": model_id,
                    "model_role": role,
                    "native_horizon": horizon,
                    "prediction_score": score,
                    "rank": rank,
                    "score_percentile": 1.0 if rank == 1 else 0.5,
                    "production_run_id": PRODUCTION_RUN_ID,
                    "shadow_run_id": SHADOW_RUN_ID,
                    "prediction_hash": PREDICTION_HASH,
                    "feature_hash": FEATURE_HASH,
                    "universe_hash": UNIVERSE_HASH,
                    "access_policy": "prospective_production",
                }
            )
    return pd.DataFrame.from_records(rows)


def _labels(sessions: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in (5, 10, 20, 60):
        entry_date, exit_date = maturity_dates(sessions, SIGNAL_DATE, horizon)
        for code, value in (("000001.SZ", horizon / 100.0), ("000002.SZ", -0.01)):
            rows.append(
                {
                    "trade_date": SIGNAL_DATE,
                    "ts_code": code,
                    "horizon": horizon,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": 10.0,
                    "exit_price": 11.0,
                    "stock_forward_ret": value,
                    "benchmark_forward_ret": 0.0,
                    "future_excess_ret": value,
                    "future_rank_pct": 1.0 if value > 0 else 0.5,
                    "future_quantile": 4 if value > 0 else 0,
                    "is_label_available": True,
                    "label_unavailable_reason": "",
                }
            )
    return pd.DataFrame.from_records(rows, columns=list(LABEL_COLUMNS))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
