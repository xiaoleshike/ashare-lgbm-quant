from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ashare_quant.utils.manifest import (
    artifact_manifest_status,
    atomic_write_json,
    config_hash,
    current_git_info,
    manifest_path,
    parquet_artifact_statistics,
    write_build_manifest,
)


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("project_name: test\n", encoding="utf-8")

    status = artifact_manifest_status(tmp_path / "artifact", config_path=config)

    assert not status.exists
    assert status.stale
    assert status.reason == "manifest missing"


def test_config_change_produces_stale_manifest_status(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    config_a = tmp_path / "config_a.yaml"
    config_b = tmp_path / "config_b.yaml"
    config_a.write_text("project_name: a\n", encoding="utf-8")
    config_b.write_text("project_name: b\n", encoding="utf-8")
    atomic_write_json(
        manifest_path(artifact_dir),
        {
            "git_commit": current_git_info()["commit"],
            "config_hash": config_hash(config_a),
        },
    )

    status = artifact_manifest_status(artifact_dir, config_path=config_b)

    assert status.exists
    assert status.stale
    assert status.config_hash_match is False
    assert "config hash mismatch" in status.reason


def test_git_revision_mismatch_is_reported(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    config = tmp_path / "config.yaml"
    config.write_text("project_name: test\n", encoding="utf-8")
    atomic_write_json(
        manifest_path(artifact_dir),
        {
            "git_commit": "not-the-current-revision",
            "config_hash": config_hash(config),
        },
    )

    status = artifact_manifest_status(artifact_dir, config_path=config)

    assert status.exists
    assert status.stale
    assert status.artifact_git_revision == "not-the-current-revision"
    assert "git revision mismatch" in status.reason


def test_atomic_manifest_write_preserves_previous_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact" / "_manifest.json"
    atomic_write_json(path, {"version": "old"})

    original_replace = Path.replace

    def fail_replace(self: Path, target: str | Path) -> Path:
        if self.name != "_manifest.json":
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(path, {"version": "new"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": "old"}


def test_incremental_manifest_separates_build_scope_from_canonical_artifact(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "universe_daily"
    first_path = artifact_dir / "year=2024" / "month=01" / "data.parquet"
    first_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    ).to_parquet(first_path, index=False)
    initial_statistics = parquet_artifact_statistics(artifact_dir)

    initial = write_build_manifest(
        artifact_dir,
        artifact_name="universe_daily",
        build_started_at="2024-01-01T00:00:00+00:00",
        config_path=None,
        start_date="20240102",
        end_date="20240103",
        row_count=2,
        partitions_changed=1,
        canonical_statistics=initial_statistics,
        source_fingerprints={},
    )

    second_path = artifact_dir / "year=2024" / "month=02" / "data.parquet"
    second_path.parent.mkdir(parents=True)
    pd.DataFrame({"trade_date": ["20240201"], "ts_code": ["000001.SZ"]}).to_parquet(
        second_path, index=False
    )
    incremental_statistics = parquet_artifact_statistics(artifact_dir)
    incremental = write_build_manifest(
        artifact_dir,
        artifact_name="universe_daily",
        build_started_at="2024-02-01T00:00:00+00:00",
        config_path=None,
        start_date="20240201",
        end_date="20240201",
        row_count=1,
        partitions_changed=1,
        canonical_statistics=incremental_statistics,
        source_fingerprints={},
    )

    assert initial["row_count"] == 2
    assert initial["canonical_artifact"] == {
        "row_count": 2,
        "partition_count": 1,
        "min_date": "20240102",
        "max_date": "20240103",
    }
    assert incremental["row_count"] == 3
    assert incremental["requested_start_date"] == "20240201"
    assert incremental["build_scope"] == {
        "build_start_date": "20240201",
        "build_end_date": "20240201",
        "rows_written_or_replaced": 1,
        "partitions_changed": 1,
    }
    assert incremental["canonical_artifact"] == {
        "row_count": 3,
        "partition_count": 2,
        "min_date": "20240102",
        "max_date": "20240201",
    }


def test_repeated_same_date_manifest_keeps_canonical_identity(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "features_daily"
    path = artifact_dir / "year=2024" / "month=01" / "data.parquet"
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240103"],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ret_1d": [None, 0.01],
        }
    )
    frame.to_parquet(path, index=False)
    statistics = parquet_artifact_statistics(artifact_dir)

    for _ in range(2):
        manifest = write_build_manifest(
            artifact_dir,
            artifact_name="features_daily",
            build_started_at="2024-01-03T00:00:00+00:00",
            config_path=None,
            start_date="20240103",
            end_date="20240103",
            row_count=1,
            partitions_changed=1,
            canonical_statistics=statistics,
            source_fingerprints={},
            extra={"feature_count": 1},
        )

    assert manifest["row_count"] == 2
    assert manifest["canonical_artifact"]["feature_count"] == 1
    assert manifest["build_scope"]["rows_written_or_replaced"] == 1


def test_failed_incremental_manifest_write_preserves_previous_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_dir = tmp_path / "universe_daily"
    atomic_write_json(manifest_path(artifact_dir), {"sentinel": "previous-valid"})
    original_replace = Path.replace

    def fail_manifest_replace(self: Path, target: str | Path) -> Path:
        if Path(target) == manifest_path(artifact_dir):
            raise OSError("incremental manifest publication failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="incremental manifest publication failed"):
        write_build_manifest(
            artifact_dir,
            artifact_name="universe_daily",
            build_started_at="2024-01-03T00:00:00+00:00",
            config_path=None,
            start_date="20240103",
            end_date="20240103",
            row_count=1,
            partitions_changed=1,
            source_fingerprints={},
        )

    assert json.loads(manifest_path(artifact_dir).read_text(encoding="utf-8")) == {
        "sentinel": "previous-valid"
    }
