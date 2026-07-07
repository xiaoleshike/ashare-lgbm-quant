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


def test_load_settings_does_not_require_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TUSHARE_TOKEN", raising=False)

    settings = load_settings(Path("config/default.yaml"))

    assert settings.has_tushare_token is False


def test_invalid_config_is_rejected(tmp_path: Path) -> None:
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("data:\n  retry_attempts: 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_file)
