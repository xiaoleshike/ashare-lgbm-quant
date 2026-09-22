from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import ashare_quant.data.research_source_snapshot as snapshot_module
from ashare_quant.cli import main
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.research_source_snapshot import (
    ResearchSourceSnapshotResult,
    materialize_repair_derived_research_source_snapshot,
    materialize_research_source_snapshot,
    snapshot_contract,
    validate_research_source_snapshot,
)
from ashare_quant.data.security_lifecycle_audit import file_sha256
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


def test_snapshot_cli_delegates_to_validated_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs: object) -> ResearchSourceSnapshotResult:
        captured.update(kwargs)
        return ResearchSourceSnapshotResult(
            snapshot_id="research_source_snapshot_fixture",
            output_dir=tmp_path / "snapshots" / "research_source_snapshot_fixture",
            idempotent=False,
        )

    monkeypatch.setattr("ashare_quant.cli.materialize_research_source_snapshot", fake_materialize)
    source = tmp_path / "source"
    snapshots = tmp_path / "snapshots"
    lock = tmp_path / "runs" / ".production.lock"

    result = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "research-source-snapshot-create",
            "--source-root",
            str(source),
            "--snapshots-root",
            str(snapshots),
            "--lifecycle-evidence",
            "config/security_identity/security_lifecycle_events.json",
            "--writer-lock-path",
            str(lock),
        ]
    )

    assert result == 0
    assert captured["source_root"] == source
    assert captured["snapshots_root"] == snapshots
    assert captured["writer_lock_path"] == lock
    assert set(captured["datasets"]) == set(snapshot_contract()["dataset_dependencies"])
    assert "research_source_snapshot_fixture" in capsys.readouterr().out


def test_repair_snapshot_cli_delegates_to_validated_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    captured: dict[str, object] = {}

    def fake_materialize(**kwargs: object) -> ResearchSourceSnapshotResult:
        captured.update(kwargs)
        return ResearchSourceSnapshotResult(
            snapshot_id="research_source_snapshot_repaired",
            output_dir=tmp_path / "snapshots/research_source_snapshot_repaired",
            idempotent=False,
        )

    monkeypatch.setattr(
        "ashare_quant.cli.materialize_repair_derived_research_source_snapshot",
        fake_materialize,
    )
    parent = tmp_path / "parent"
    probe = tmp_path / "probe"
    snapshots = tmp_path / "snapshots"

    result = main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "research-source-snapshot-repair",
            "--parent-snapshot",
            str(parent),
            "--repair-source-artifact",
            str(probe),
            "--snapshots-root",
            str(snapshots),
            "--lifecycle-evidence",
            "config/security_identity/security_lifecycle_events.json",
        ]
    )

    assert result == 0
    assert captured["parent_snapshot"] == parent
    assert captured["repair_source_artifact"] == probe
    assert "research_source_snapshot_repaired" in capsys.readouterr().out


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


def test_repair_derived_snapshot_applies_only_frozen_provider_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    partition = source / "suspend_d" / "year=2024" / "month=01"
    partition.mkdir(parents=True)
    pd.DataFrame(
        {
            "ts_code": ["000002.SZ"],
            "trade_date": ["20240102"],
            "suspend_timing": [None],
            "suspend_type": ["S"],
            "month": ["01"],
            "year": [2024],
        }
    ).to_parquet(partition / "data.parquet", index=False)
    snapshots = tmp_path / "snapshots"
    parent = materialize_research_source_snapshot(
        source_root=source,
        snapshots_root=snapshots,
        security_identity_mapping_hash="mapping-hash",
        lifecycle_evidence_hash="old-evidence-hash",
        writer_lock_path=tmp_path / "runs" / ".production.lock",
        datasets=("suspend_d",),
    )
    probe = _probe_artifact(tmp_path / "probe")

    child = materialize_repair_derived_research_source_snapshot(
        parent_snapshot=parent.output_dir,
        repair_source_artifact=probe,
        snapshots_root=snapshots,
        lifecycle_evidence_hash="new-evidence-hash",
    )

    validate_research_source_snapshot(child.output_dir)
    parent_rows = pd.read_parquet(
        parent.output_dir / "datasets/suspend_d/year=2024/month=01/data.parquet"
    )
    child_rows = pd.read_parquet(
        child.output_dir / "datasets/suspend_d/year=2024/month=01/data.parquet"
    )
    assert len(parent_rows) == 1
    assert len(child_rows) == 2
    assert set(child_rows["ts_code"]) == {"000001.SZ", "000002.SZ"}


def _probe_artifact(path: Path) -> Path:
    path.mkdir(parents=True)
    missing = [
        {
            "canonical_ts_code": "000001.SZ",
            "source_ts_code": "000001.SZ",
            "trade_date": "20240103",
            "suspend_timing": None,
            "suspend_type": "S",
        }
    ]
    pd.DataFrame(
        [
            {
                "source_completeness_status": "LOCAL_SUSPEND_D_INCOMPLETE",
                "missing_suspend_rows": json.dumps(missing),
            }
        ]
    ).to_parquet(path / "comparison.parquet", index=False)
    pd.DataFrame(
        [
            {
                "request_id": "request-1",
                **missing[0],
            }
        ]
    ).to_parquet(path / "suspend_d_provider.parquet", index=False)
    pd.DataFrame(
        columns=["request_id", "source_ts_code", "canonical_ts_code", "trade_date"]
    ).to_parquet(path / "daily_provider.parquet", index=False)
    (path / "requests.json").write_text("[]", encoding="utf-8")
    (path / "summary.json").write_text("{}", encoding="utf-8")
    artifacts = {
        name: file_sha256(path / name)
        for name in (
            "requests.json",
            "suspend_d_provider.parquet",
            "daily_provider.parquet",
            "comparison.parquet",
            "summary.json",
        )
    }
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_name": "security_lifecycle_source_probe",
                "probe_id": path.name,
                "artifact_hashes": artifacts,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


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
