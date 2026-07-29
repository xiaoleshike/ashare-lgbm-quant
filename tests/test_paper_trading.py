from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.cli import main
from ashare_quant.config import load_settings
from ashare_quant.config.settings import (
    AppSettings,
    PaperPortfolioSettings,
    PaperTradingSettings,
    PathSettings,
)
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.paper_trading import service as paper_service_module
from ashare_quant.paper_trading.service import PaperTradingService
from ashare_quant.paper_trading.signals import PaperSignal, _combine_percentile_rankings

SIGNAL_DATE = "20240102"
ENTRY_DATE = "20240103"
NEXT_DATE = "20240104"


class FixedSignals:
    def __init__(self, rankings: dict[tuple[str, str], list[tuple[str, float]]]) -> None:
        self.rankings = rankings

    def load(
        self,
        portfolio: PaperPortfolioSettings,
        as_of: str,
        production_summary_path: Path,
    ) -> PaperSignal:
        del production_summary_path
        rows = self.rankings[(portfolio.portfolio_id, as_of)]
        ranking = pd.DataFrame(
            {
                "trade_date": [as_of] * len(rows),
                "ts_code": [item[0] for item in rows],
                "prediction_score": [item[1] for item in rows],
                "rank": range(1, len(rows) + 1),
            }
        )
        return PaperSignal(
            portfolio.portfolio_id,
            as_of,
            "fixture-model",
            "fixture-feature-hash",
            f"signal-{portfolio.portfolio_id}-{as_of}",
            ranking,
        )


def test_default_configuration_creates_four_isolated_accounts(tmp_path: Path) -> None:
    settings = AppSettings(
        paths=_paths(tmp_path),
        paper_trading=PaperTradingSettings(
            portfolios=(
                PaperPortfolioSettings(
                    portfolio_id="champion_top20",
                    signal_type="champion",
                    model_id="champion",
                ),
                PaperPortfolioSettings(
                    portfolio_id="h5_top20",
                    signal_type="model",
                    model_id="h5",
                ),
                PaperPortfolioSettings(
                    portfolio_id="h10_top20",
                    signal_type="model",
                    model_id="h10",
                ),
                PaperPortfolioSettings(
                    portfolio_id="ensemble_top20",
                    signal_type="ensemble",
                    component_model_ids=("h5", "h10"),
                ),
            )
        ),
    )
    service = _service(tmp_path, settings)

    first = service.init()
    second = service.init()

    assert first.account_count == 4
    assert first.created_count == 4
    assert second.created_count == 0
    for portfolio in settings.paper_trading.portfolios:
        account = _json(tmp_path / "paper" / portfolio.portfolio_id / "account.json")
        assert account["portfolio_id"] == portfolio.portfolio_id
        assert account["broker_connected"] is False
        assert account["real_orders_generated"] is False


def test_repository_config_declares_requested_paper_portfolios() -> None:
    settings = load_settings("config/default.yaml").paper_trading

    assert [item.portfolio_id for item in settings.portfolios] == [
        "champion_top20",
        "h5_top20",
        "h10_top20",
        "ensemble_top20",
    ]
    assert all(item.top_n == 20 for item in settings.portfolios)
    assert settings.execution == "next_open"
    assert settings.lot_size == 100


def test_rebalance_is_idempotent_and_orders_execute_at_next_open(tmp_path: Path) -> None:
    service = paper_fixture(tmp_path)
    service.signal_provider = FixedSignals({("alpha", SIGNAL_DATE): [("000001.SZ", 0.9)]})  # type: ignore[assignment]

    first = service.rebalance(SIGNAL_DATE)
    second = service.rebalance(SIGNAL_DATE)
    execution = service.execute(ENTRY_DATE)
    repeated_execution = service.execute(ENTRY_DATE)

    assert first.execution_rule == "next_open"
    assert first.orders_written == 1
    assert second.orders_written == 0
    assert execution.trades_written == 1
    assert repeated_execution.trades_written == 0
    order = pd.read_parquet(tmp_path / "paper" / "alpha" / "orders.parquet").iloc[0]
    trade = pd.read_parquet(tmp_path / "paper" / "alpha" / "trades.parquet").iloc[0]
    assert order["as_of"] == SIGNAL_DATE
    assert order["execution_rule"] == "next_open"
    assert trade["as_of"] == ENTRY_DATE
    assert trade["side"] == "buy"
    assert trade["status"] == "filled"
    assert int(trade["shares"]) % 100 == 0


def test_rebalance_does_not_require_future_trade_calendar(tmp_path: Path) -> None:
    service = paper_fixture(tmp_path)
    calendar_path = next((tmp_path / "raw" / "trade_cal").glob("**/*.parquet"))
    calendar = pd.read_parquet(calendar_path)
    calendar.loc[calendar["cal_date"].astype(str) <= SIGNAL_DATE].to_parquet(
        calendar_path, index=False
    )
    service.signal_provider = FixedSignals({("alpha", SIGNAL_DATE): [("000001.SZ", 0.9)]})  # type: ignore[assignment]

    result = service.rebalance(SIGNAL_DATE)

    assert result.execution_rule == "next_open"
    assert result.orders_written == 1


def test_rebalance_validates_every_portfolio_before_writing_orders(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
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
        ),
    )
    _write_market_fixture(tmp_path)
    service = _service(tmp_path, settings)
    service.signal_provider = FixedSignals({("alpha", SIGNAL_DATE): [("000001.SZ", 0.9)]})  # type: ignore[assignment]

    with pytest.raises(KeyError):
        service.rebalance(SIGNAL_DATE)

    assert not list((tmp_path / "paper").glob("*/orders.parquet"))


def test_ensemble_percentile_ranks_align_by_stock_key_not_score_order() -> None:
    first = pd.DataFrame(
        {
            "trade_date": [SIGNAL_DATE] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "prediction_score": [3.0, 2.0, 1.0],
        }
    )
    second = pd.DataFrame(
        {
            "trade_date": [SIGNAL_DATE] * 3,
            "ts_code": ["000003.SZ", "000001.SZ", "000002.SZ"],
            "prediction_score": [4.0, 3.0, 1.0],
        }
    )

    result = _combine_percentile_rankings([("h5", first), ("h10", second)])
    scores = result.set_index("ts_code")["prediction_score"]

    assert scores["000001.SZ"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert scores["000002.SZ"] == pytest.approx((2.0 / 3.0 + 1.0 / 3.0) / 2.0)
    assert scores["000003.SZ"] == pytest.approx((1.0 / 3.0 + 1.0) / 2.0)


def test_ensemble_rejects_actual_stock_key_mismatch() -> None:
    first = pd.DataFrame(
        {
            "trade_date": [SIGNAL_DATE],
            "ts_code": ["000001.SZ"],
            "prediction_score": [1.0],
        }
    )
    second = first.assign(ts_code="000002.SZ")

    with pytest.raises(DataValidationError, match="universe differs"):
        _combine_percentile_rankings([("h5", first), ("h10", second)])


def test_failed_trade_commit_can_be_replayed_without_duplicate_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = paper_fixture(tmp_path)
    service.signal_provider = FixedSignals({("alpha", SIGNAL_DATE): [("000001.SZ", 0.9)]})  # type: ignore[assignment]
    service.rebalance(SIGNAL_DATE)
    real_append = paper_service_module.append_ledger
    failed = False

    def fail_trade_once(
        path: Path,
        rows: pd.DataFrame,
        *,
        unique_columns: tuple[str, ...],
        sort_columns: tuple[str, ...],
    ) -> int:
        nonlocal failed
        if path.name == "trades.parquet" and not failed:
            failed = True
            raise OSError("forced trade commit failure")
        return real_append(
            path,
            rows,
            unique_columns=unique_columns,
            sort_columns=sort_columns,
        )

    monkeypatch.setattr(paper_service_module, "append_ledger", fail_trade_once)
    with pytest.raises(OSError, match="forced trade commit failure"):
        service.execute(ENTRY_DATE)

    replay = service.execute(ENTRY_DATE)

    assert replay.trades_written == 1
    assert replay.equity_rows_written == 0
    assert len(pd.read_parquet(tmp_path / "paper" / "alpha" / "positions.parquet")) == 1
    assert len(pd.read_parquet(tmp_path / "paper" / "alpha" / "equity_curve.parquet")) == 1
    assert len(pd.read_parquet(tmp_path / "paper" / "alpha" / "trades.parquet")) == 1


def test_execution_applies_limits_suspension_lots_and_costs(tmp_path: Path) -> None:
    service = paper_fixture(tmp_path, top_n=3)
    service.signal_provider = FixedSignals(
        {
            ("alpha", SIGNAL_DATE): [
                ("000001.SZ", 0.9),
                ("000002.SZ", 0.8),
                ("000003.SZ", 0.7),
            ]
        }
    )  # type: ignore[assignment]

    service.rebalance(SIGNAL_DATE)
    service.execute(ENTRY_DATE)
    trades = pd.read_parquet(tmp_path / "paper" / "alpha" / "trades.parquet")
    normal = trades.loc[trades["ts_code"].eq("000001.SZ")].iloc[0]
    limit_up = trades.loc[trades["ts_code"].eq("000002.SZ")].iloc[0]
    suspended = trades.loc[trades["ts_code"].eq("000003.SZ")].iloc[0]

    assert normal["status"] == "filled"
    assert int(normal["shares"]) % 100 == 0
    assert float(normal["commission"]) == pytest.approx(float(normal["gross_value"]) * 0.00025)
    assert float(normal["slippage"]) == pytest.approx(float(normal["gross_value"]) * 0.0005)
    assert float(normal["stamp_duty"]) == 0.0
    assert limit_up["status"] == "rejected"
    assert limit_up["reason"] == "not_buyable"
    assert suspended["status"] == "rejected"
    assert suspended["reason"] == "not_buyable"


def test_limit_down_stock_cannot_be_sold_and_stamp_duty_applies_to_sell(
    tmp_path: Path,
) -> None:
    service = paper_fixture(tmp_path)
    signals = FixedSignals(
        {
            ("alpha", SIGNAL_DATE): [("000001.SZ", 0.9)],
            ("alpha", ENTRY_DATE): [("000004.SZ", 0.9)],
            ("alpha", NEXT_DATE): [("000005.SZ", 0.9)],
        }
    )
    service.signal_provider = signals  # type: ignore[assignment]
    service.rebalance(SIGNAL_DATE)
    service.execute(ENTRY_DATE)
    service.rebalance(ENTRY_DATE)
    service.execute(NEXT_DATE)

    trades = pd.read_parquet(tmp_path / "paper" / "alpha" / "trades.parquet")
    blocked = trades.loc[trades["ts_code"].eq("000001.SZ") & trades["as_of"].eq(NEXT_DATE)].iloc[0]
    assert blocked["side"] == "sell"
    assert blocked["status"] == "rejected"
    assert blocked["reason"] == "not_sellable"

    service.rebalance(NEXT_DATE)
    service.execute("20240105")
    updated = pd.read_parquet(tmp_path / "paper" / "alpha" / "trades.parquet")
    sale = updated.loc[updated["ts_code"].eq("000001.SZ") & updated["as_of"].eq("20240105")].iloc[0]
    assert sale["status"] == "filled"
    assert float(sale["stamp_duty"]) == pytest.approx(float(sale["gross_value"]) * 0.001)


def test_portfolios_have_separate_cash_positions_and_ledgers(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        portfolios=(
            PaperPortfolioSettings(
                portfolio_id="alpha", signal_type="model", model_id="alpha-model", top_n=1
            ),
            PaperPortfolioSettings(
                portfolio_id="beta", signal_type="model", model_id="beta-model", top_n=1
            ),
        ),
    )
    _write_market_fixture(tmp_path)
    service = _service(tmp_path, settings)
    service.signal_provider = FixedSignals(
        {
            ("alpha", SIGNAL_DATE): [("000001.SZ", 0.9)],
            ("beta", SIGNAL_DATE): [("000004.SZ", 0.9)],
        }
    )  # type: ignore[assignment]

    service.rebalance(SIGNAL_DATE)
    service.execute(ENTRY_DATE)
    report = service.report(ENTRY_DATE)

    alpha = pd.read_parquet(tmp_path / "paper" / "alpha" / "positions.parquet")
    beta = pd.read_parquet(tmp_path / "paper" / "beta" / "positions.parquet")
    summary = _json(report.summary_path)
    assert set(alpha.loc[alpha["shares"].gt(0), "ts_code"]) == {"000001.SZ"}
    assert set(beta.loc[beta["shares"].gt(0), "ts_code"]) == {"000004.SZ"}
    assert summary["portfolio_count"] == 2
    assert summary["constraints"]["broker_connected"] is False
    assert summary["constraints"]["real_orders_generated"] is False


def test_paper_trading_init_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  raw_data: {tmp_path / 'raw'}",
                f"  processed_data: {tmp_path / 'processed'}",
                f"  parquet_store: {tmp_path / 'raw'}",
                f"  models: {tmp_path / 'models'}",
                f"  reports: {tmp_path / 'reports'}",
                f"  paper_trading: {tmp_path / 'paper'}",
                "paper_trading:",
                "  portfolios:",
                "    - portfolio_id: alpha",
                "      signal_type: model",
                "      model_id: alpha-model",
                "      top_n: 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["--config", str(config), "paper-trading", "init"]) == 0
    assert "paper_trading_init: accounts=1 created=1" in capsys.readouterr().out


def paper_fixture(tmp_path: Path, *, top_n: int = 1) -> PaperTradingService:
    _write_market_fixture(tmp_path)
    settings = _settings(
        tmp_path,
        portfolios=(
            PaperPortfolioSettings(
                portfolio_id="alpha",
                signal_type="model",
                model_id="fixture-model",
                top_n=top_n,
            ),
        ),
    )
    return _service(tmp_path, settings)


def _service(tmp_path: Path, settings: AppSettings) -> PaperTradingService:
    config = tmp_path / "config.yaml"
    if not config.exists():
        config.write_text("project_name: paper-test\n", encoding="utf-8")
    return PaperTradingService(
        settings=settings,
        config_path=config,
        registry=ModelRegistry(tmp_path / "models"),
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        reports_root=tmp_path / "reports",
        paper_root=tmp_path / "paper",
    )


def _settings(
    tmp_path: Path,
    *,
    portfolios: tuple[PaperPortfolioSettings, ...],
) -> AppSettings:
    return AppSettings(
        paths=_paths(tmp_path),
        paper_trading=PaperTradingSettings(
            initial_cash=100_000.0,
            portfolios=portfolios,
        ),
    )


def _paths(tmp_path: Path) -> PathSettings:
    return PathSettings(
        raw_data=tmp_path / "raw",
        processed_data=tmp_path / "processed",
        parquet_store=tmp_path / "raw",
        models=tmp_path / "models",
        reports=tmp_path / "reports",
        paper_trading=tmp_path / "paper",
    )


def _write_market_fixture(tmp_path: Path) -> None:
    codes = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ")
    dates = (SIGNAL_DATE, ENTRY_DATE, NEXT_DATE, "20240105")
    _write_dataset(
        tmp_path / "raw",
        "trade_cal",
        pd.DataFrame({"cal_date": dates, "is_open": [1] * len(dates)}),
    )
    daily_rows: list[dict[str, object]] = []
    limit_rows: list[dict[str, object]] = []
    universe_rows: list[dict[str, object]] = []
    for date in dates:
        for index, code in enumerate(codes):
            open_price = 10.0 + index
            if date == NEXT_DATE and code == "000001.SZ":
                open_price = 9.0
            daily_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "open": open_price,
                    "close": open_price,
                }
            )
            limit_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "up_limit": open_price if date == ENTRY_DATE and code == "000002.SZ" else 99.0,
                    "down_limit": 9.0 if date == NEXT_DATE and code == "000001.SZ" else 1.0,
                }
            )
            universe_rows.append(
                {
                    "trade_date": date,
                    "ts_code": code,
                    "is_suspended": date == ENTRY_DATE and code == "000003.SZ",
                    "is_st": False,
                }
            )
    _write_dataset(tmp_path / "raw", "daily", pd.DataFrame(daily_rows))
    _write_dataset(tmp_path / "raw", "stk_limit", pd.DataFrame(limit_rows))
    _write_dataset(
        tmp_path / "processed",
        "universe_daily",
        pd.DataFrame(universe_rows),
    )


def _write_dataset(root: Path, dataset: str, frame: pd.DataFrame) -> None:
    directory = root / dataset / "year=2024" / "month=01"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(directory / "data.parquet", index=False)


def _dataset_path(root: Path, dataset: str) -> Path:
    return next((root / dataset).glob("**/*.parquet"))


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload
