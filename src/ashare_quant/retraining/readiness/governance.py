"""Governance snapshot validation before retraining execution."""

from __future__ import annotations

from pathlib import Path

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.registry_versions import load_registry_records
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.readiness.schemas import ClosedLoopContext, GovernanceContext
from ashare_quant.retraining.readiness.validators import SourceTracker, require_string

_REPORTS = ("status.json", "validation.json", "recovery.json", "promotion_status.json")


def validate_governance(
    *,
    settings: AppSettings,
    reports_root: Path,
    as_of: str,
    closed_loop: ClosedLoopContext,
    tracker: SourceTracker,
) -> GovernanceContext:
    """Validate immutable governance, Registry, recovery, and promotion state."""

    root = reports_root / "governance" / as_of
    manifest = tracker.json(root / "manifest.json", "governance snapshot manifest")
    if (
        manifest.get("artifact_name") != "daily_governance_snapshot"
        or manifest.get("snapshot_id") != closed_loop.governance_snapshot_id
        or str(manifest.get("as_of")) != as_of
    ):
        raise DataValidationError("governance snapshot identity is invalid")
    hashes = manifest.get("artifact_hashes")
    if not isinstance(hashes, dict) or set(hashes) != set(_REPORTS):
        raise DataValidationError("governance snapshot has incomplete artifact hashes")
    history = root / "history" / closed_loop.governance_snapshot_id
    history_manifest = tracker.json(history / "manifest.json", "immutable governance manifest")
    if history_manifest != manifest:
        raise DataValidationError("governance projection differs from immutable manifest")
    reports: dict[str, dict[str, object]] = {}
    for name in _REPORTS:
        projected = tracker.json(root / name, f"governance {name}")
        immutable = tracker.json(history / name, f"immutable governance {name}")
        if projected != immutable or file_sha256(root / name) != hashes.get(name):
            raise DataValidationError(f"governance artifact hash mismatch: {name}")
        reports[name] = projected
    status = reports["status.json"]
    validation = reports["validation.json"]
    recovery = reports["recovery.json"]
    promotion = reports["promotion_status.json"]
    if validation.get("status") == "FAIL" or recovery.get("status") == "FAIL":
        raise DataValidationError("governance validation or recovery report has hard failures")
    recovery_summary = recovery.get("summary")
    if not isinstance(recovery_summary, dict) or any(
        recovery_summary.get(name)
        for name in ("interrupted_transactions", "incomplete_publications")
    ):
        raise DataValidationError("governance recovery has a pending transaction")
    promotion_summary = promotion.get("promotion")
    if not isinstance(promotion_summary, dict) or promotion_summary.get("invalid_requests"):
        raise DataValidationError("governance promotion state contains invalid requests")
    champion = status.get("summary")
    champion = champion.get("champion") if isinstance(champion, dict) else None
    if not isinstance(champion, dict) or not isinstance(champion.get("assignment_id"), str):
        raise DataValidationError("governance snapshot lacks a valid Champion assignment")
    source_hashes = status.get("source_hashes")
    registry = settings.paths.models / "registry.json"
    tracker.track(registry)
    try:
        records = load_registry_records(registry)
    except Exception as error:
        raise DataValidationError(f"current Registry is invalid: {error}") from error
    champions = [item for item in records if item.status == "champion"]
    if len(champions) != 1 or champions[0].model_id != champion.get("model_id"):
        raise DataValidationError("current Champion differs from governance snapshot")
    if not isinstance(source_hashes, dict):
        raise DataValidationError("governance status lacks source hashes")
    for source, expected_hash in source_hashes.items():
        source_path = Path(str(source))
        tracker.track(source_path)
        if file_sha256(source_path) != expected_hash:
            raise DataValidationError(f"governance source hash mismatch: {source_path}")
    recorded_registry_hashes = {
        str(value) for key, value in source_hashes.items() if str(key).endswith("/registry.json")
    }
    if recorded_registry_hashes != {file_sha256(registry)}:
        raise DataValidationError("current Registry differs from governance snapshot")
    assignment_id = str(champion["assignment_id"])
    assignments = list(settings.paths.models.glob("champion_history/*.json"))
    assignment_matches = [
        tracker.json(path, "Champion assignment")
        for path in assignments
        if path.stem == assignment_id
    ]
    if (
        len(assignment_matches) != 1
        or assignment_matches[0].get("model_id") != champions[0].model_id
    ):
        raise DataValidationError("Champion assignment history is invalid")
    assignment = assignment_matches[0]
    previous_hash: str | None = None
    previous_version: str | None = None
    promotion_request_id = assignment.get("promotion_request_id")
    if isinstance(promotion_request_id, str) and promotion_request_id:
        gate = tracker.json(
            reports_root / "promotion_gate" / promotion_request_id / "manifest.json",
            "Champion promotion gate manifest",
        )
        previous_hash = require_string(gate, "policy_hash", "promotion gate manifest")
        previous_version = require_string(gate, "policy_version", "promotion gate manifest")
    if promotion.get("production_run_id") != closed_loop.production_run_id:
        raise DataValidationError("governance snapshot is linked to another production run")
    return GovernanceContext(
        snapshot_hash=file_sha256(root / "manifest.json"),
        promotion_policy_hash=require_string(
            promotion, "promotion_policy_hash", "governance promotion status"
        ),
        promotion_policy_version=require_string(
            promotion, "promotion_policy_version", "governance promotion status"
        ),
        previous_promotion_policy_hash=previous_hash,
        previous_promotion_policy_version=previous_version,
    )
