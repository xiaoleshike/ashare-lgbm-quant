from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_quant.utils.manifest import (
    artifact_manifest_status,
    atomic_write_json,
    config_hash,
    current_git_info,
    manifest_path,
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
