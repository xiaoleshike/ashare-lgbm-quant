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
    assert settings.data.endpoint_rate_limits_per_minute["cyq_chips"] == 200
    assert settings.data.snapshot_refresh_policy == "manual"
    assert settings.data.snapshot_refresh_ttl_days == 7


def test_load_settings_does_not_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    settings = load_settings(Path("config/default.yaml"))

    assert settings.has_tushare_token is False


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("data:\n  retry_attempts: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_file)


def test_universe_settings_are_loaded_from_default_config() -> None:
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")

    assert settings.universe.min_list_trading_days == 180
    assert settings.universe.liquidity_window_days == 20
    assert settings.universe.mark_limit_up_not_buyable is True


def test_label_settings_are_loaded_from_default_config() -> None:
    from ashare_quant.config import load_settings

    settings = load_settings("config/default.yaml")

    assert settings.labels.horizons == (3, 5, 10)
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
