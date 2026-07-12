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


def test_labels_build_cli_with_fixture_data(tmp_path, monkeypatch, capsys) -> None:
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


def test_features_registry_cli_reports_count(capsys) -> None:
    exit_code = main(["--config", "config/default.yaml", "features", "registry"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "feature_count=204" in captured.out
