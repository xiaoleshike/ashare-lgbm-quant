from ashare_quant.cli import main


def test_config_check_cli_reports_non_secret_status(
    capsys, monkeypatch
) -> None:
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
