"""Validated feature-list loading for controlled model experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.features.registry import FEATURE_REGISTRY


def load_recommended_features(path: Path) -> tuple[str, ...]:
    """Load exactly the top-50 list from a diagnostics recommendation artifact."""

    resolved = resolve_recommended_path(path)
    payload = load_json_object(resolved)
    features = parse_feature_array(payload, "recommended_features", resolved)
    if payload.get("recommended_set") != "top_50" or len(features) != 50:
        raise DataValidationError(
            "Experiment A requires recommended_set=top_50 with exactly 50 features"
        )
    validate_feature_names(features, resolved)
    return features


def load_robust_features(path: Path) -> tuple[str, ...]:
    """Load the manually maintained robust subset used by Experiment B."""

    payload = load_json_object(path)
    features = parse_feature_array(payload, "features", path)
    if len(features) >= 50:
        raise DataValidationError("Experiment B robust feature list must contain fewer than 50")
    validate_feature_names(features, path)
    return features


def resolve_recommended_path(path: Path) -> Path:
    """Resolve diagnostics `latest.json` to its immutable recommendation file."""

    if path.name != "latest.json":
        return path
    payload = load_json_object(path)
    report_dir = payload.get("report_dir")
    if not isinstance(report_dir, str) or not report_dir:
        raise DataValidationError(f"diagnostics latest pointer lacks report_dir: {path}")
    return Path(report_dir) / "recommended_features.json"


def feature_list_hash(features: tuple[str, ...]) -> str:
    """Return a deterministic ordered feature-list SHA256 hash."""

    canonical = json.dumps(list(features), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json_object(path: Path) -> dict[str, object]:
    """Read one JSON object with a clear validation error."""

    if not path.exists():
        raise DataValidationError(f"feature list does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid feature-list JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"feature-list JSON must contain an object: {path}")
    return payload


def parse_feature_array(payload: dict[str, object], key: str, path: Path) -> tuple[str, ...]:
    """Parse a non-empty, unique string array from a feature-list object."""

    raw = payload.get(key)
    if not isinstance(raw, list) or not raw or not all(isinstance(item, str) for item in raw):
        raise DataValidationError(f"{path} must contain a non-empty string array `{key}`")
    features = tuple(str(item) for item in raw)
    if len(features) != len(set(features)):
        raise DataValidationError(f"feature list contains duplicate names: {path}")
    return features


def validate_feature_names(features: tuple[str, ...], path: Path) -> None:
    """Reject features outside the enabled production registry."""

    enabled = {spec.name for spec in FEATURE_REGISTRY}
    unknown = sorted(set(features) - enabled)
    if unknown:
        raise DataValidationError(f"feature list contains disabled or unknown features: {unknown}")
