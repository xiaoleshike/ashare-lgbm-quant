from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ashare_quant.cli import main
from ashare_quant.features import FEATURE_REGISTRY, FeatureStore, FeatureValidator


def test_feature_validator_accepts_valid_registered_schema(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    store.write(feature_frame())

    result = FeatureValidator(store).validate("20240105", "20240105")

    assert result.ok
    assert result.rows == 2
    assert result.errors == ()


def test_feature_validator_rejects_infinite_values(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path)
    frame = feature_frame()
    frame.loc[0, FEATURE_REGISTRY[0].name] = np.inf
    store.write(frame)

    result = FeatureValidator(store).validate("20240105", "20240105")

    assert not result.ok
    assert "infinite values" in result.errors[0]


def test_features_validate_cli_returns_nonzero_for_empty_date(tmp_path: Path, capsys) -> None:
    store = FeatureStore(tmp_path)
    store.write(feature_frame())

    exit_code = main(
        [
            "--config",
            "config/default.yaml",
            "features",
            "--processed-root",
            str(tmp_path),
            "validate",
            "--start-date",
            "20240108",
            "--end-date",
            "20240108",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "validation: ok=False rows=0" in captured.out
    assert "date range is empty" in captured.out


def feature_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in ("000001.SZ", "000002.SZ"):
        row: dict[str, object] = {"trade_date": "20240105", "ts_code": code}
        row.update({spec.name: 1.0 for spec in FEATURE_REGISTRY})
        rows.append(row)
    return pd.DataFrame(rows)
