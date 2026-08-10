"""Append-only review records for historical backtests that are unsafe as evidence."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.utils.manifest import atomic_write_json, current_git_info


@dataclass(frozen=True, slots=True)
class BacktestInvalidationResult:
    invalidation_id: str
    output_dir: Path
    idempotent: bool


class BacktestInvalidationService:
    """Publish an immutable invalidation without editing the referenced backtest."""

    def __init__(self, *, backtests_root: Path, reports_root: Path) -> None:
        self.backtests_root = backtests_root
        self.root = reports_root / "backtest_invalidations"

    def create(
        self,
        *,
        backtest_id: str,
        reason_codes: tuple[str, ...],
        reviewed_by: str,
        note: str = "",
    ) -> BacktestInvalidationResult:
        if not backtest_id or Path(backtest_id).name != backtest_id:
            raise DataValidationError("invalid backtest_id")
        if (
            not reviewed_by.strip()
            or not reason_codes
            or any(not item.strip() for item in reason_codes)
        ):
            raise DataValidationError("reviewed_by and non-empty reason codes are required")
        backtest_path = self.backtests_root / backtest_id
        manifest_path = backtest_path / "manifest.json"
        if not manifest_path.is_file():
            raise DataValidationError(f"backtest manifest does not exist: {manifest_path}")
        manifest_hash = _file_hash(manifest_path)
        logical = {
            "schema_version": 1,
            "backtest_id": backtest_id,
            "backtest_manifest_hash": manifest_hash,
            "reason_codes": sorted(set(reason_codes)),
            "reviewed_by": reviewed_by.strip(),
            "note": note,
        }
        invalidation_id = f"backtest_invalidation_{_payload_hash(logical)[:24]}"
        output = self.root / invalidation_id
        if output.exists():
            payload = _load_json(output / "invalidation.json")
            if payload.get("logical_identity_hash") != _payload_hash(logical):
                raise DataValidationError("immutable backtest invalidation identity differs")
            return BacktestInvalidationResult(invalidation_id, output, True)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            **logical,
            "artifact_name": "backtest_invalidation",
            "invalidation_id": invalidation_id,
            "backtest_path": str(backtest_path),
            "reviewed_at": datetime.now(UTC).isoformat(),
            "logical_identity_hash": _payload_hash(logical),
            "replacement_status": "NOT_REPLACED",
        }
        with tempfile.TemporaryDirectory(dir=self.root, prefix=".invalidation-") as temporary:
            staging = Path(temporary)
            atomic_write_json(staging / "invalidation.json", payload)
            (staging / "report.md").write_text(_report(payload), encoding="utf-8")
            git = current_git_info()
            manifest = {
                "schema_version": 1,
                "artifact_name": "backtest_invalidation_manifest",
                "invalidation_id": invalidation_id,
                "invalidation_sha256": _file_hash(staging / "invalidation.json"),
                "backtest_manifest_sha256": manifest_hash,
                "git_commit": git["commit"],
                "git_dirty": git["dirty"],
                "manifest_written_last": True,
            }
            atomic_write_json(staging / "manifest.json", manifest)
            if output.exists():
                raise DataValidationError("immutable backtest invalidation already exists")
            staging.rename(output)
        return BacktestInvalidationResult(invalidation_id, output, False)


def _report(payload: dict[str, object]) -> str:
    reasons = payload["reason_codes"]
    assert isinstance(reasons, list)
    return "\n".join(
        [
            "# Backtest Evidence Invalidation",
            "",
            f"- Backtest: `{payload['backtest_id']}`",
            f"- Reviewed by: `{payload['reviewed_by']}`",
            f"- Reasons: {', '.join(str(item) for item in reasons)}",
            "",
            "The original backtest remains immutable. This record only prevents its use as "
            "trusted evidence.",
            "",
        ]
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid backtest invalidation: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"backtest invalidation must be an object: {path}")
    return payload
