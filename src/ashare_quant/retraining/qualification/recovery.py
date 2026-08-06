"""Read-only recovery inspection for qualification snapshots."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.shadow.storage import file_sha256
from ashare_quant.retraining.qualification.authorization_storage import (
    QualificationAuthorizationStorage,
)
from ashare_quant.retraining.qualification.invariants import compare_protected_state
from ashare_quant.retraining.qualification.schemas import (
    QualificationRecovery,
    QualificationSnapshot,
)
from ashare_quant.retraining.qualification.storage import QualificationStorage


def inspect_qualification_recovery(
    storage: QualificationStorage,
    run_id: str,
    *,
    authorization_storage: QualificationAuthorizationStorage | None = None,
    current_static_policy_hash: str | None = None,
    now: Callable[[], datetime] | None = None,
) -> QualificationRecovery:
    issues: list[str] = []
    actions: list[str] = []
    output = storage.output_dir(run_id)
    if not output.exists():
        return QualificationRecovery(run_id, "MISSING", ("qualification does not exist",), ())
    try:
        snapshot = storage.read(run_id)
    except (DataValidationError, OSError, ValueError) as error:
        issues.append(str(error))
        actions.append("inspect the incomplete snapshot and preserve it for forensic review")
        snapshot = None
    for stale_path in (
        sorted(storage.staging_root.glob("*")) if storage.staging_root.is_dir() else ()
    ):
        issues.append(f"stale qualification staging path: {stale_path}")
    backup = storage.staging_root / f".{run_id}.backup"
    if backup.exists():
        issues.append(f"stale qualification backup: {backup}")
    if snapshot is not None:
        for name, source in snapshot.source_inventory.items():
            source_path = source.get("path")
            digest = source.get("sha256")
            if not isinstance(source_path, str) or not isinstance(digest, str):
                issues.append(f"invalid source inventory entry: {name}")
                continue
            try:
                if file_sha256(Path(source_path)) != digest:
                    issues.append(f"qualification source changed: {name}")
            except DataValidationError:
                issues.append(f"qualification source missing: {name}")
        baseline = snapshot.invariant_results.get("baseline")
        current = snapshot.invariant_results.get("current")
        if isinstance(baseline, dict) and isinstance(current, dict):
            issues.extend(
                f"protected invariant changed: {name}"
                for name in compare_protected_state(baseline, current)
            )
        if snapshot.summary.current_state in {"TRAINING", "VALIDATING", "SHADOW_ENROLLING"}:
            issues.append(f"ambiguous interrupted state: {snapshot.summary.current_state}")
        if snapshot.summary.static_qualification_policy_hash is None:
            issues.append("LEGACY_AUTHORIZATION_MIGRATION_REQUIRED")
        elif (
            current_static_policy_hash is not None
            and current_static_policy_hash != snapshot.summary.static_qualification_policy_hash
        ):
            issues.append("qualification static-policy drift")
        if authorization_storage is not None:
            _inspect_authorizations(
                snapshot,
                authorization_storage,
                issues,
                now=(now or (lambda: datetime.now(UTC)))(),
            )
    if issues:
        actions.append("run qualification-status and inspect referenced immutable stage manifests")
    return QualificationRecovery(
        run_id,
        "ACTION_REQUIRED" if issues else "CLEAN",
        tuple(dict.fromkeys(issues)),
        tuple(dict.fromkeys(actions)),
    )


def _inspect_authorizations(
    snapshot: QualificationSnapshot,
    storage: QualificationAuthorizationStorage,
    issues: list[str],
    *,
    now: datetime,
) -> None:
    summary = snapshot.summary
    events = snapshot.events
    run_id = summary.qualification_run_id
    for stale in storage.staging_paths(run_id):
        issues.append(f"stale authorization staging path: {stale}")
    try:
        authorizations = storage.authorizations(run_id)
    except (DataValidationError, OSError, ValueError) as error:
        issues.append(str(error))
        return
    active_by_stage: dict[str, int] = {"training": 0, "shadow": 0}
    for authorization, _, digest in authorizations:
        if authorization.qualification_run_id != run_id:
            issues.append(
                f"authorization bound to wrong qualification: {authorization.authorization_id}"
            )
        matching = [
            event
            for event in events
            if event.details.get("authorization_id") == authorization.authorization_id
            and event.details.get("authorization_sha256") == digest
        ]
        if len(matching) != 1:
            issues.append(
                f"authorization audit binding is invalid: {authorization.authorization_id}"
            )
        try:
            revocations = storage.revocations(run_id, authorization.authorization_id)
            claims = storage.claims(run_id, authorization.authorization_id)
        except (DataValidationError, OSError, ValueError) as error:
            issues.append(str(error))
            continue
        if len(revocations) > 1:
            issues.append(f"duplicate authorization revocations: {authorization.authorization_id}")
        for revocation, _, _ in revocations:
            if revocation.authorization_sha256 != digest or revocation.stage != authorization.stage:
                issues.append(
                    f"invalid authorization revocation binding: {revocation.revocation_id}"
                )
        if len(claims) > 1:
            issues.append(f"concurrent authorization claims: {authorization.authorization_id}")
        if not revocations and not claims:
            expiry = datetime.fromisoformat(authorization.expires_at)
            if expiry.tzinfo is None:
                issues.append(f"naive authorization expiration: {authorization.authorization_id}")
            elif now.astimezone(UTC) >= expiry.astimezone(UTC):
                issues.append(f"expired authorization: {authorization.authorization_id}")
            else:
                active_by_stage[authorization.stage] += 1
        for claim, _, _ in claims:
            if claim.authorization_sha256 != digest or claim.stage != authorization.stage:
                issues.append(f"invalid authorization claim binding: {claim.consumption_id}")
            receipts = storage.receipts(
                run_id, authorization.authorization_id, claim.consumption_id
            )
            if not receipts:
                issues.append(f"authorization claim lacks completion: {claim.consumption_id}")
            elif len(receipts) > 1:
                issues.append(
                    f"authorization claim has duplicate completions: {claim.consumption_id}"
                )
            elif (
                receipts[0][0].authorization_id != authorization.authorization_id
                or receipts[0][0].stage != authorization.stage
                or receipts[0][0].consumption_id != claim.consumption_id
            ):
                issues.append(
                    f"invalid authorization completion binding: {receipts[0][0].receipt_id}"
                )
    for stage, count in active_by_stage.items():
        if count > 1:
            issues.append(f"duplicate active {stage} authorization conflict")
