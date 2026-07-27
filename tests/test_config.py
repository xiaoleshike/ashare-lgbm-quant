from pathlib import Path

import pytest
from pydantic import ValidationError

from ashare_quant.config import load_settings


def test_load_settings_reads_yaml_and_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUSHARE_TOKEN", "test-token")

    settings = load_settings(Path("config/default.yaml"))

    assert settings.project_name == "ashare-lgbm-quant"
    assert settings.has_tushare_token is True
    assert settings.data.provider == "tushare"
    assert settings.data.run_baostock_post_ingestion_check is False
    assert settings.data.endpoint_rate_limits_per_minute["cyq_chips"] == 200
    assert settings.data.snapshot_refresh_policy == "manual"
    assert settings.data.snapshot_refresh_ttl_days == 7
    assert settings.data.index_first_available_dates["399006.SZ"] == "20100531"
    assert settings.production.freshness.baseline_sessions == 20
    assert settings.production.freshness.git_dirty_policy == "warning"
    assert settings.production.freshness.hard_required_features == ()
    assert settings.production.timezone == "Asia/Shanghai"
    assert settings.production.market_data_ready_time == "18:30"
    assert settings.production.scheduler.enabled is True
    assert settings.production.scheduler.skip_if_already_successful is True


def test_load_settings_does_not_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    settings = load_settings(Path("config/default.yaml"))

    assert settings.has_tushare_token is False


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("data:\n  retry_attempts: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_file)


def test_invalid_freshness_threshold_order_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid_freshness.yaml"
    config_file.write_text(
        "production:\n  freshness:\n"
        "    severe_count_ratio_low: 0.9\n"
        "    moderate_count_ratio_low: 0.8\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="count-ratio thresholds"):
        load_settings(config_file)


def test_invalid_production_scheduler_clock_is_rejected(tmp_path: Path) -> None:
    invalid_timezone = tmp_path / "invalid_timezone.yaml"
    invalid_timezone.write_text(
        "production:\n  timezone: Invalid/Timezone\n",
        encoding="utf-8",
    )
    invalid_time = tmp_path / "invalid_time.yaml"
    invalid_time.write_text(
        "production:\n  market_data_ready_time: '18:30:00'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="timezone is unknown"):
        load_settings(invalid_timezone)
    with pytest.raises(ValidationError, match="must use HH:MM"):
        load_settings(invalid_time)


def test_index_inception_boundary_requires_configured_code_and_valid_date(
    tmp_path: Path,
) -> None:
    unknown_code = tmp_path / "unknown_index.yaml"
    unknown_code.write_text(
        "data:\n  index_codes: [000300.SH]\n"
        "  index_first_available_dates:\n    399006.SZ: '20100531'\n",
        encoding="utf-8",
    )
    invalid_date = tmp_path / "invalid_date.yaml"
    invalid_date.write_text(
        "data:\n  index_codes: [399006.SZ]\n"
        "  index_first_available_dates:\n    399006.SZ: '201005'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="not present in data.index_codes"):
        load_settings(unknown_code)
    with pytest.raises(ValidationError, match="must be YYYYMMDD"):
        load_settings(invalid_date)


def test_universe_settings_are_loaded_from_default_config() -> None:
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")

    assert settings.universe.min_list_trading_days == 180
    assert settings.universe.liquidity_window_days == 20
    assert settings.universe.mark_limit_up_not_buyable is True


def test_label_settings_are_loaded_from_default_config() -> None:
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")

    assert settings.labels.horizons == (5, 10, 20, 60)
    assert settings.labels.benchmark_index_code == "000300.SH"
    assert settings.labels.benchmark_index_code in settings.data.index_codes
    assert settings.labels.quantile_buckets == 5
    assert settings.labels.skip_unbuyable_entry is True


def test_feature_settings_are_loaded_from_default_config() -> None:
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")

    assert settings.features.return_windows == (1, 3, 5, 10, 20, 60, 120)
    assert settings.features.benchmark_index_code == "000300.SH"
    assert settings.features.benchmark_index_code in settings.data.index_codes
    assert settings.features.include_fundamentals is True
    assert settings.features.enable_industry_features is False
    assert settings.features.enable_unsafe_fina_indicator_features is False


def test_config_rejects_industry_features_without_pit_source(tmp_path: Path) -> None:
    config_file = tmp_path / "unsafe_industry_features.yaml"
    config_file.write_text("features:\n  enable_industry_features: true\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="no verified point-in-time industry source"):
        load_settings(config_file)


def test_config_rejects_unsafe_fina_indicator_features(tmp_path: Path) -> None:
    config_file = tmp_path / "unsafe_fina_indicator_features.yaml"
    config_file.write_text(
        "features:\n  enable_unsafe_fina_indicator_features: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="fina_indicator data lacks f_ann_date"):
        load_settings(config_file)


def test_default_config_downloads_configured_benchmark_index() -> None:
    settings = load_settings("config/default.yaml")

    assert "000300.SH" in settings.data.index_codes
    assert settings.labels.benchmark_index_code == settings.features.benchmark_index_code
    assert settings.labels.benchmark_index_code in settings.data.index_codes


def test_config_rejects_benchmark_missing_from_index_downloads(tmp_path: Path) -> None:
    config_file = tmp_path / "missing_benchmark.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data:",
                "  index_codes:",
                "    - 000001.SH",
                "labels:",
                "  benchmark_index_code: 000300.SH",
                "features:",
                "  benchmark_index_code: 000300.SH",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="included in data.index_codes"):
        load_settings(config_file)


def test_config_rejects_different_label_and_feature_benchmarks(tmp_path: Path) -> None:
    config_file = tmp_path / "different_benchmarks.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data:",
                "  index_codes:",
                "    - 000300.SH",
                "    - 000905.SH",
                "labels:",
                "  benchmark_index_code: 000300.SH",
                "features:",
                "  benchmark_index_code: 000905.SH",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="must match"):
        load_settings(config_file)
