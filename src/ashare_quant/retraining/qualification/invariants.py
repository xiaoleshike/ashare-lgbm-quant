"""Protected-state snapshots for qualification safety verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256


def protected_state_inventory(
    *, project_root: Path, models_root: Path, reports_root: Path, paper_root: Path, as_of: str
) -> dict[str, dict[str, Any]]:
    """Hash governance, trading, candidate, Champion, and production-Shadow state."""

    paths: dict[str, Path] = {
        "registry": models_root / "registry.json",
        "champion_history": models_root / "champion_history",
        "promotion_state": models_root / "promotion_requests",
        "rollback_state": models_root / "rollback_requests",
        "paper_trading_state": paper_root,
        "production_candidates": reports_root / as_of / "candidates.csv",
        "production_shadow_manifest": reports_root / "shadow_predictions" / as_of / "manifest.json",
        "production_shadow_predictions": reports_root
        / "shadow_predictions"
        / as_of
        / "predictions.parquet",
    }
    champion = _champion_artifact(models_root)
    if champion is not None:
        paths["champion_model_artifact"] = champion
    return {
        name: {
            "path": str(path),
            "sha256": path_fingerprint(path, project_root=project_root),
        }
        for name, path in sorted(paths.items())
    }


def compare_protected_state(
    baseline: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]]
) -> tuple[str, ...]:
    changed = []
    for name in sorted(set(baseline) | set(current)):
        if baseline.get(name, {}).get("sha256") != current.get(name, {}).get("sha256"):
            changed.append(name)
    return tuple(changed)


def path_fingerprint(path: Path, *, project_root: Path) -> str | None:
    if path.is_file():
        return file_sha256(path)
    if not path.exists():
        return None
    records = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        try:
            name = str(child.resolve().relative_to(project_root.resolve()))
        except ValueError:
            name = str(child.resolve())
        records.append({"path": name, "sha256": file_sha256(child)})
    return canonical_payload_hash(records)


def _champion_artifact(models_root: Path) -> Path | None:
    registry = models_root / "registry.json"
    if not registry.is_file():
        return None
    import json

    try:
        payload = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return registry
    records = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return registry
    champions = [
        item for item in records if isinstance(item, dict) and item.get("status") == "champion"
    ]
    if len(champions) != 1:
        return registry
    artifact = champions[0].get("artifact_path")
    return Path(str(artifact)) if artifact else registry
