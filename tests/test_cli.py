from ashare_quant.cli import main


def test_config_check_cli_reports_non_secret_status(capsys, monkeypatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")

    exit_code = main(["--config", "config/default.yaml", "config-check"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "project=ashare-lgbm-quant" in captured.out
    assert "tushare_token_set=True" in captured.out
    assert "hidden-token" not in captured.out


def test_doctor_cli_succeeds_without_token(monkeypatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    assert main(["--config", "config/default.yaml", "doctor"]) == 0


def test_data_validate_cli_returns_zero_for_valid_required_dataset(tmp_path, capsys) -> None:
    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore

    store = ParquetDataStore(tmp_path)
    store.write(
        get_dataset_spec("stock_basic"),
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],
                "name": ["Ping An"],
                "list_date": ["19910403"],
            }
        ),
    )

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "validate",
            "--dataset",
            "stock_basic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stock_basic: ok=True status=valid" in captured.out


def test_data_status_cli_reports_snapshot_freshness(tmp_path, capsys) -> None:
    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore

    store = ParquetDataStore(tmp_path)
    store.write(
        get_dataset_spec("stock_basic"),
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],
                "name": ["Ping An"],
                "list_date": ["19910403"],
            }
        ),
    )

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "status",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stock_basic: exists=True" in captured.out
    assert "snapshot_updated_at=" in captured.out
    assert "snapshot_age_days=" in captured.out


def test_data_validate_cli_returns_nonzero_for_missing_required_dataset(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "validate",
            "--dataset",
            "stock_basic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "stock_basic: ok=False status=missing" in captured.out
    assert "required dataset is not downloaded" in captured.out


def test_data_validate_cli_returns_nonzero_for_empty_required_dataset(tmp_path, capsys) -> None:
    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec

    spec = get_dataset_spec("stock_basic")
    path = tmp_path / "stock_basic" / "snapshot=latest"
    path.mkdir(parents=True)
    pd.DataFrame(columns=list(spec.required_columns)).to_parquet(path / "data.parquet", index=False)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "validate",
            "--dataset",
            "stock_basic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "stock_basic: ok=False status=empty" in captured.out
    assert "required dataset is empty" in captured.out


def test_data_validate_cli_skips_missing_optional_dataset(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "validate",
            "--dataset",
            "fund_basic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "fund_basic: ok=True status=skipped_optional" in captured.out
    assert "optional dataset is not downloaded" in captured.out


def test_data_validate_cli_returns_nonzero_when_one_dataset_fails(tmp_path, capsys) -> None:
    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore

    store = ParquetDataStore(tmp_path)
    store.write(
        get_dataset_spec("stock_basic"),
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "symbol": ["000001"],
                "name": ["Ping An"],
                "list_date": ["19910403"],
            }
        ),
    )

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "validate",
            "--dataset",
            "stock_basic",
            "--dataset",
            "daily",
            "--dataset",
            "fund_basic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "stock_basic: ok=True status=valid" in captured.out
    assert "daily: ok=False status=missing" in captured.out
    assert "fund_basic: ok=True status=skipped_optional" in captured.out


def test_data_gaps_cli_reports_missing_trading_days(tmp_path, capsys) -> None:
    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore

    store = ParquetDataStore(tmp_path)
    store.write(
        get_dataset_spec("trade_cal"),
        pd.DataFrame(
            {
                "exchange": ["SSE", "SSE", "SSE"],
                "cal_date": ["20240102", "20240103", "20240104"],
                "is_open": [1, 1, 1],
            }
        ),
    )
    store.write(
        get_dataset_spec("daily"),
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "trade_date": ["20240102", "20240104"],
                "open": [10.0, 10.0],
                "high": [11.0, 11.0],
                "low": [9.0, 9.0],
                "close": [10.5, 10.5],
                "vol": [100.0, 100.0],
            }
        ),
    )

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path),
            "gaps",
            "--dataset",
            "daily",
            "--start-date",
            "20240102",
            "--end-date",
            "20240104",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "daily: gaps=True" in captured.out
    assert "first=20240103" in captured.out


def test_gap_status_distinguishes_repairable_and_pre_inception_dates(capsys) -> None:
    from ashare_quant.cli import print_gap_reports
    from ashare_quant.data.ingestion import GapReport

    print_gap_reports(
        [
            GapReport(
                dataset="index_daily",
                start_date="20100104",
                end_date="20100601",
                expected_dates=2,
                missing_dates=("20100601",),
                missing_by_entity={"399006.SZ": ("20100601",)},
                excluded_before_inception_by_entity={"399006.SZ": ("20100104", "20100528")},
            )
        ]
    )

    output = capsys.readouterr().out
    assert "missing=1 first=20100601" in output
    assert "excluded_before_inception=2 first=20100104,20100528" in output


def test_universe_build_cli_with_fixture_raw_store(tmp_path, monkeypatch, capsys) -> None:
    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore
    from test_universe import fixture_inputs

    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    raw_store = ParquetDataStore(raw_root)
    for name, frame in fixture_inputs().items():
        raw_store.write(get_dataset_spec(name), frame)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "universe",
            "--storage-root",
            str(raw_root),
            "--output-root",
            str(output_root),
            "build",
            "--start-date",
            "20240105",
            "--end-date",
            "20240105",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "universe_daily: rows_built=8 rows_written=8" in captured.out
    stored = pd.read_parquet(
        output_root / "universe_daily" / "year=2024" / "month=01" / "data.parquet"
    )
    assert len(stored) == 8


def test_universe_incremental_manifest_describes_complete_canonical_store(
    tmp_path, monkeypatch
) -> None:
    import json

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore
    from test_universe import fixture_inputs

    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    raw_store = ParquetDataStore(raw_root)
    for name, frame in fixture_inputs().items():
        raw_store.write(get_dataset_spec(name), frame)

    for trade_date in ("20240104", "20240105", "20240105"):
        exit_code = main(
            [
                "--config",
                "config/default.yaml",
                "universe",
                "--storage-root",
                str(raw_root),
                "--output-root",
                str(output_root),
                "build",
                "--start-date",
                trade_date,
                "--end-date",
                trade_date,
            ]
        )
        assert exit_code == 0

    manifest = json.loads(
        (output_root / "universe_daily" / "_manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["row_count"] == 16
    assert manifest["canonical_artifact"] == {
        "row_count": 16,
        "partition_count": 1,
        "min_date": "20240104",
        "max_date": "20240105",
    }
    assert manifest["build_scope"] == {
        "build_start_date": "20240105",
        "build_end_date": "20240105",
        "rows_written_or_replaced": 8,
        "partitions_changed": 1,
    }


def test_labels_build_cli_with_fixture_data(tmp_path, monkeypatch, capsys) -> None:
    import json

    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore
    from ashare_quant.universe import UniverseStore
    from test_labels import label_fixture_inputs

    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_store = ParquetDataStore(raw_root)
    inputs = label_fixture_inputs()
    for name in ("trade_cal", "daily", "adj_factor", "index_daily"):
        raw_store.write(get_dataset_spec(name), inputs[name])
    UniverseStore(processed_root).write(inputs["universe"])

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "labels",
            "--storage-root",
            str(raw_root),
            "--processed-root",
            str(processed_root),
            "build",
            "--start-date",
            "20240102",
            "--end-date",
            "20240102",
            "--horizons",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "labels_forward: rows_built=7 rows_written=7" in captured.out
    stored = pd.read_parquet(
        processed_root / "labels_forward" / "year=2024" / "month=01" / "data.parquet"
    )
    assert len(stored) == 7
    manifest = json.loads(
        (processed_root / "labels_forward" / "_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["artifact_name"] == "labels_forward"
    assert manifest["label_horizons"] == [3]
    assert manifest["row_count"] == 7


def test_labels_validate_cli_returns_nonzero_for_invalid_labels(
    tmp_path, monkeypatch, capsys
) -> None:
    import pandas as pd

    from ashare_quant.labels import build_label_frame
    from test_labels import default_label_settings, label_fixture_inputs

    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    processed_root = tmp_path / "processed"
    labels = build_label_frame(
        label_fixture_inputs(), default_label_settings(), "20240102", "20240102", (3,)
    )
    invalid = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    label_path = processed_root / "labels_forward" / "year=2024" / "month=01"
    label_path.mkdir(parents=True)
    invalid.to_parquet(label_path / "data.parquet", index=False)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "labels",
            "--processed-root",
            str(processed_root),
            "validate",
            "--start-date",
            "20240102",
            "--end-date",
            "20240102",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "duplicate label rows" in captured.out


def test_labels_validate_cli_returns_nonzero_for_empty_labels(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    processed_root = tmp_path / "processed"

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "labels",
            "--processed-root",
            str(processed_root),
            "validate",
            "--start-date",
            "20240102",
            "--end-date",
            "20240102",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "labels are empty or not built" in captured.out


def test_labels_validate_cli_returns_nonzero_for_missing_configured_horizon(
    tmp_path, monkeypatch, capsys
) -> None:
    from ashare_quant.labels import build_label_frame
    from test_labels import default_label_settings, label_fixture_inputs

    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    processed_root = tmp_path / "processed"
    labels = build_label_frame(
        label_fixture_inputs(), default_label_settings(), "20240102", "20240102", (3, 5)
    )
    label_path = processed_root / "labels_forward" / "year=2024" / "month=01"
    label_path.mkdir(parents=True)
    labels.to_parquet(label_path / "data.parquet", index=False)

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "labels",
            "--processed-root",
            str(processed_root),
            "validate",
            "--start-date",
            "20240102",
            "--end-date",
            "20240102",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "configured label horizons are missing: [10]" in captured.out


def test_failed_labels_build_does_not_replace_existing_manifest(
    tmp_path, monkeypatch, capsys
) -> None:
    import json

    import pandas as pd

    from ashare_quant.data.datasets import get_dataset_spec
    from ashare_quant.data.storage import ParquetDataStore
    from ashare_quant.utils.manifest import atomic_write_json

    monkeypatch.setenv("TUSHARE_TOKEN", "hidden-token")
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    ParquetDataStore(raw_root).write(
        get_dataset_spec("trade_cal"),
        pd.DataFrame({"exchange": ["SSE"], "cal_date": ["20240102"], "is_open": [1]}),
    )
    manifest_path = processed_root / "labels_forward" / "_manifest.json"
    atomic_write_json(manifest_path, {"artifact_name": "labels_forward", "sentinel": "old"})

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "labels",
            "--storage-root",
            str(raw_root),
            "--processed-root",
            str(processed_root),
            "build",
            "--start-date",
            "20240102",
            "--end-date",
            "20240102",
            "--horizons",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code != 0
    assert captured.out == "" or "validation: ok=False" in captured.out
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["sentinel"] == "old"


def test_features_status_reports_missing_manifest(tmp_path, capsys) -> None:
    processed_root = tmp_path / "processed"

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "features",
            "--processed-root",
            str(processed_root),
            "status",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "features_daily_manifest: exists=False stale=True" in captured.out


def test_features_registry_cli_reports_count(capsys) -> None:
    exit_code = main(["--config", "config/default.yaml", "features", "registry"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "feature_count=153" in captured.out
