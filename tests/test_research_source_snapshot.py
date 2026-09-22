from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import ashare_quant.data.research_source_snapshot as snapshot_module
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.research_source_snapshot import (
    materialize_research_source_snapshot,
    snapshot_contract,
    validate_research_source_snapshot,
)
from ashare_quant.orchestration import (
    ProductionLockError,
    acquire_production_lock,
    production_lock_path,
    release_production_lock,
)


def test_physical_snapshot_survives_mutable_source_update(tmp_path: Path) -> None:
    source = tmp_path / "production_raw"
    _dataset(source, "daily", ["20240102", "20240103"])
    result = materialize_research_source_snapshot(
        source_root=source,
        snapshots_root=tmp_path / "research_snapshots",
        security_identity_mapping_hash="mapping-hash",
        lifecycle_evidence_hash="evidence-hash",
        writer_lock_path=tmp_path / "runs" / ".production.lock",
        datasets=("daily",),
    )
    frozen = result.output_dir / "datasets" / "daily" / "part.parquet"
    frozen_before = frozen.read_bytes()

    _dataset(source, "daily", ["20240102", "20240103", "20240104"])

    validate_research_source_snapshot(result.output_dir)
    assert frozen.read_bytes() == frozen_before
    assert pd.read_parquet(frozen)["trade_date"].tolist() == ["20240102", "20240103"]


def test_manifest_only_reference_is_not_a_reproducible_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _dataset(source, "daily", ["20240102"])
    result = materialize_research_source_snapshot(
        source_root=source,
        snapshots_root=tmp_path / "snapshots",
        security_identity_mapping_hash="mapping-hash",
        lifecycle_evidence_hash="evidence-hash",
        writer_lock_path=tmp_path / "runs" / ".production.lock",
        datasets=("daily",),
    )
    shutil.rmtree(result.output_dir / "datasets")

    with pytest.raises(DataValidationError, match="FROZEN_BYTES_MISSING"):
        validate_research_source_snapshot(result.output_dir)


def test_snapshot_identity_ignores_source_absolute_path(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _dataset(first, "daily", ["20240102"])
    _dataset(second, "daily", ["20240102"])

    one = materialize_research_source_snapshot(
        source_root=first,
        snapshots_root=tmp_path / "snapshots-one",
        security_identity_mapping_hash="mapping-hash",
        lifecycle_evidence_hash="evidence-hash",
        writer_lock_path=tmp_path / "runs-one" / ".production.lock",
        datasets=("daily",),
    )
    two = materialize_research_source_snapshot(
        source_root=second,
        snapshots_root=tmp_path / "snapshots-two",
        security_identity_mapping_hash="mapping-hash",
        lifecycle_evidence_hash="evidence-hash",
        writer_lock_path=tmp_path / "runs-two" / ".production.lock",
        datasets=("daily",),
    )

    assert one.snapshot_id == two.snapshot_id


def test_snapshot_uses_the_same_production_writer_lock(tmp_path: Path) -> None:
    production_state = tmp_path / "production" / "paper_trading"
    source = tmp_path / "production" / "raw"
    _dataset(source, "daily", ["20240102"])
    writer_lock = production_lock_path(production_state)
    lock = acquire_production_lock(writer_lock, command="production fixture")
    try:
        with pytest.raises(ProductionLockError, match="another production run is active"):
            materialize_research_source_snapshot(
                source_root=source,
                snapshots_root=tmp_path / "research" / "snapshots",
                security_identity_mapping_hash="mapping-hash",
                lifecycle_evidence_hash="evidence-hash",
                writer_lock_path=writer_lock,
                datasets=("daily",),
            )
    finally:
        release_production_lock(lock)


def test_snapshot_contract_declares_actual_rebuild_dependencies() -> None:
    contract = snapshot_contract()

    assert "forecast" not in contract["dataset_dependencies"]
    assert "express" not in contract["dataset_dependencies"]
    assert set(contract["dependency_owners"]["universe"]) >= {
        "daily",
        "suspend_d",
        "stock_basic",
    }
    assert set(contract["dependency_owners"]["features"]) >= {
        "daily",
        "fina_indicator",
        "income",
        "balancesheet",
        "cashflow",
    }
    assert (
        json.loads(
            Path("config/research_source_snapshot_contract.json").read_text(encoding="utf-8")
        )["contract_version"]
        == contract["contract_version"]
    )


def test_snapshot_rejects_mixed_source_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _dataset(source, "daily", ["20240102"])
    original = snapshot_module._source_inventory
    calls = 0

    def changing_inventory(root: Path, datasets: tuple[str, ...]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        inventory = original(root, datasets)
        if calls == 2:
            inventory["datasets"][0]["generation_fixture"] = "changed"  # type: ignore[index]
        return inventory

    monkeypatch.setattr(snapshot_module, "_source_inventory", changing_inventory)

    with pytest.raises(DataValidationError, match="MIXED_GENERATION"):
        materialize_research_source_snapshot(
            source_root=source,
            snapshots_root=tmp_path / "snapshots",
            security_identity_mapping_hash="mapping-hash",
            lifecycle_evidence_hash="evidence-hash",
            writer_lock_path=tmp_path / "runs" / ".production.lock",
            datasets=("daily",),
        )


def _dataset(root: Path, dataset: str, dates: list[str]) -> None:
    path = root / dataset
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * len(dates),
            "trade_date": dates,
            "close": [10.0] * len(dates),
        }
    ).to_parquet(path / "part.parquet", index=False)
