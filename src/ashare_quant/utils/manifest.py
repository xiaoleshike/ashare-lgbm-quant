"""Lightweight provenance manifests for processed artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.storage import ParquetDataStore

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ManifestStatus:
    """Summarize whether a processed artifact manifest matches the current run context."""

    exists: bool
    stale: bool
    artifact_git_revision: str | None = None
    current_git_revision: str | None = None
    config_hash_match: bool | None = None
    reason: str = "missing"


def utc_now_iso() -> str:
    """Return a UTC timestamp suitable for manifest fields."""

    return datetime.now(UTC).isoformat(timespec="seconds")


def write_build_manifest(
    artifact_dir: Path,
    *,
    artifact_name: str,
    build_started_at: str,
    config_path: str | Path | None,
    start_date: str,
    end_date: str,
    row_count: int,
    source_fingerprints: dict[str, dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write one processed artifact provenance manifest."""

    git_info = current_git_info()
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_name": artifact_name,
        "build_started_at": build_started_at,
        "build_completed_at": utc_now_iso(),
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "config_path": str(config_path) if config_path is not None else None,
        "config_hash": config_hash(config_path),
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "row_count": row_count,
        "output_path": str(artifact_dir),
        "source_fingerprints": source_fingerprints,
    }
    if extra:
        manifest.update(extra)
    atomic_write_json(manifest_path(artifact_dir), manifest)
    return manifest


def manifest_path(artifact_dir: Path) -> Path:
    """Return the manifest path for a processed artifact directory."""

    return artifact_dir / "_manifest.json"


def read_manifest(artifact_dir: Path) -> dict[str, Any] | None:
    """Read a manifest if present."""

    path = manifest_path(artifact_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    return loaded if isinstance(loaded, dict) else None


def artifact_manifest_status(
    artifact_dir: Path,
    *,
    config_path: str | Path | None,
) -> ManifestStatus:
    """Compare stored artifact manifest against current code/config identity."""

    manifest = read_manifest(artifact_dir)
    current_git = current_git_info()["commit"]
    if manifest is None:
        return ManifestStatus(
            exists=False,
            stale=True,
            current_git_revision=current_git,
            reason="manifest missing",
        )
    artifact_git = as_optional_str(manifest.get("git_commit"))
    artifact_hash = as_optional_str(manifest.get("config_hash"))
    current_hash = config_hash(config_path)
    git_matches = artifact_git == current_git
    config_matches = artifact_hash == current_hash
    reasons: list[str] = []
    if not git_matches:
        reasons.append("git revision mismatch")
    if not config_matches:
        reasons.append("config hash mismatch")
    return ManifestStatus(
        exists=True,
        stale=bool(reasons),
        artifact_git_revision=artifact_git,
        current_git_revision=current_git,
        config_hash_match=config_matches,
        reason=", ".join(reasons) if reasons else "current",
    )


def config_hash(config_path: str | Path | None) -> str | None:
    """Return a deterministic SHA256 hash of the config file bytes."""

    if config_path is None:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_git_info() -> dict[str, Any]:
    """Return current git commit and dirty state when git metadata is available."""

    commit = run_git_command(["git", "rev-parse", "HEAD"])
    dirty_text = run_git_command(["git", "status", "--porcelain"])
    return {
        "commit": commit,
        "dirty": bool(dirty_text),
    }


def run_git_command(args: list[str]) -> str | None:
    """Run a git command and return stripped stdout, or None outside a git repo."""

    try:
        completed = subprocess.run(  # noqa: S603
            args,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = completed.stdout.strip()
    return output or None


def raw_source_fingerprints(
    store: ParquetDataStore,
    dataset_names: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Return inexpensive fingerprints for raw source datasets."""

    fingerprints: dict[str, dict[str, Any]] = {}
    for name in dataset_names:
        status = store.status(get_dataset_spec(name))
        fingerprints[name] = {
            "exists": status.exists,
            "rows": status.rows,
            "partitions": status.partitions,
            "min_date": status.min_date,
            "max_date": status.max_date,
            "snapshot_updated_at": status.snapshot_updated_at,
        }
    return fingerprints


def processed_source_fingerprint(
    artifact_dir: Path,
    *,
    rows: int,
    partitions: int,
    min_date: str | None,
    max_date: str | None,
) -> dict[str, Any]:
    """Return an inexpensive fingerprint for a processed source artifact."""

    manifest = read_manifest(artifact_dir)
    return {
        "exists": artifact_dir.exists(),
        "rows": rows,
        "partitions": partitions,
        "min_date": min_date,
        "max_date": max_date,
        "manifest_git_commit": None if manifest is None else manifest.get("git_commit"),
        "manifest_config_hash": None if manifest is None else manifest.get("config_hash"),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON without replacing a previous valid manifest on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path: Path | None = None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temp_path = Path(file.name)
        try:
            file.write(text)
            file.write("\n")
            file.flush()
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    try:
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def as_optional_str(value: object) -> str | None:
    """Return a value as a string unless it is missing."""

    if value is None:
        return None
    return str(value)
