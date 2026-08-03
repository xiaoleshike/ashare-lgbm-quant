"""Read-only retraining execution readiness orchestration and publication."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.promotion.gate_rules import (
    PromotionGatePolicy,
    load_promotion_gate_policy,
)
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.retraining.configuration import load_retraining_policy
from ashare_quant.retraining.readiness.closed_loop import validate_closed_loop
from ashare_quant.retraining.readiness.governance import validate_governance
from ashare_quant.retraining.readiness.policy import RetrainingReadinessPolicy
from ashare_quant.retraining.readiness.policy_validation import (
    PromotionPolicyDriftError,
    validate_promotion_policy,
)
from ashare_quant.retraining.readiness.reporting import render_readiness
from ashare_quant.retraining.readiness.scheduler import validate_scheduler
from ashare_quant.retraining.readiness.schemas import (
    ReadinessCheck,
    ReadinessResult,
    RetrainingReadinessManifest,
    RetrainingReadinessReport,
)
from ashare_quant.retraining.readiness.validators import SourceTracker
from ashare_quant.retraining.storage import RetrainingRequestStorage
from ashare_quant.retraining.validators import evidence_hash, validate_recorded_evidence
from ashare_quant.utils.manifest import atomic_write_json, config_hash, current_git_info

_CHECKS = (
    "scheduler",
    "closed_loop_manifest",
    "governance_snapshot",
    "promotion_policy",
    "training_request",
)


class RetrainingExecutionReadinessValidator:
    """Prove operational governance is coherent before future training execution."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        project_root: Path,
        retraining_policy_path: Path,
        promotion_policy_path: Path,
        policy: RetrainingReadinessPolicy | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.project_root = project_root
        self.runs_root = project_root / "runs"
        self.retraining_policy_path = retraining_policy_path
        self.promotion_policy_path = promotion_policy_path
        self.retraining_policy = load_retraining_policy(retraining_policy_path)
        self.promotion_policy: PromotionGatePolicy = load_promotion_gate_policy(
            promotion_policy_path
        )
        self.policy = policy or RetrainingReadinessPolicy()
        self.now = now or (lambda: datetime.now(UTC))
        self.storage = RetrainingRequestStorage(
            reports_root=settings.paths.reports,
            config_path=config_path,
            policy=self.retraining_policy,
            promotion_policy=self.promotion_policy,
        )

    def validate(self, as_of: str, *, request_id: str | None = None) -> ReadinessResult:
        tracker = SourceTracker()
        tracker.track(self.retraining_policy_path)
        tracker.track(self.promotion_policy_path)
        details: list[ReadinessCheck] = []
        production_run_id: str | None = None
        governance_hash: str | None = None
        request_hash: str | None = None
        feature_hash: str | None = None
        universe_hash: str | None = None
        label_hash: str | None = None
        resolved_request_id = request_id
        failed = False
        scheduler_context = None
        closed_context = None
        governance_context = None
        request = None
        for name in _CHECKS:
            if failed:
                details.append(
                    ReadinessCheck(name=name, status="NOT_RUN", message="prior check failed")
                )
                continue
            try:
                if name == "scheduler":
                    scheduler_context = validate_scheduler(
                        settings=self.settings,
                        project_root=self.project_root,
                        runs_root=self.runs_root,
                        as_of=as_of,
                        policy=self.policy,
                        tracker=tracker,
                        now=self.now(),
                    )
                    production_run_id = scheduler_context.production_run_id
                elif name == "closed_loop_manifest":
                    assert scheduler_context is not None
                    closed_context = validate_closed_loop(
                        reports_root=self.settings.paths.reports,
                        runs_root=self.runs_root,
                        as_of=as_of,
                        scheduler=scheduler_context,
                        tracker=tracker,
                    )
                elif name == "governance_snapshot":
                    assert closed_context is not None
                    governance_context = validate_governance(
                        settings=self.settings,
                        reports_root=self.settings.paths.reports,
                        as_of=as_of,
                        closed_loop=closed_context,
                        tracker=tracker,
                    )
                    governance_hash = governance_context.snapshot_hash
                elif name == "promotion_policy":
                    assert governance_context is not None
                    validate_promotion_policy(self.promotion_policy, governance_context)
                else:
                    (
                        resolved_request_id,
                        request,
                        request_hash,
                        feature_hash,
                        universe_hash,
                        label_hash,
                    ) = self._validate_request(as_of, resolved_request_id, tracker)
                    assert governance_context is not None
                    validate_promotion_policy(self.promotion_policy, governance_context, request)
                details.append(ReadinessCheck(name=name, status="PASS", message="validated"))
            except PromotionPolicyDriftError as error:
                failed = True
                details.append(
                    ReadinessCheck(name=name, status="FAILED_POLICY_DRIFT", message=str(error))
                )
            except (DataValidationError, OSError, ValueError, AssertionError) as error:
                failed = True
                details.append(ReadinessCheck(name=name, status="FAIL", message=str(error)))
        checks = {item.name: item.status for item in details}
        identity = {
            "as_of": as_of,
            "request_id": resolved_request_id,
            "production_run_id": production_run_id,
            "source_hashes": dict(sorted(tracker.hashes.items())),
            "promotion_policy_hash": self.promotion_policy.policy_hash,
            "request_hash": request_hash,
            "feature_hash": feature_hash,
            "universe_hash": universe_hash,
            "label_hash": label_hash,
            "checks": checks,
        }
        run_id = f"readiness_{as_of}_{canonical_payload_hash(identity)[:16]}"
        report = RetrainingReadinessReport(
            run_id=run_id,
            as_of=as_of,
            request_id=resolved_request_id,
            status="FAILED" if failed else "READY",
            checks=checks,
            check_details=tuple(details),
            production_run_id=production_run_id,
            governance_snapshot_hash=governance_hash,
            promotion_policy_hash=self.promotion_policy.policy_hash,
            request_hash=request_hash,
            feature_hash=feature_hash,
            universe_hash=universe_hash,
            label_hash=label_hash,
        )
        return self._publish(report, tracker.hashes)

    def _validate_request(
        self, as_of: str, request_id: str | None, tracker: SourceTracker
    ) -> tuple[str, Any, str, str, str, str]:
        if request_id is None:
            candidates: list[str] = []
            for directory in sorted(self.storage.requests_root.glob("training_*")):
                stored = self.storage.read(directory.name)
                if stored is not None and stored[0].as_of == as_of:
                    candidates.append(directory.name)
            if len(candidates) != 1:
                raise DataValidationError(
                    f"readiness requires exactly one retraining request for {as_of}; "
                    f"found={len(candidates)}; pass --request-id when multiple exist"
                )
            request_id = candidates[0]
        stored = self.storage.read(request_id)
        if stored is None:
            raise DataValidationError(f"retraining request does not exist: {request_id}")
        request, manifest = stored
        if request.as_of != as_of or request.status not in {"CREATED", "VALIDATED"}:
            raise DataValidationError("retraining request date or lifecycle status is invalid")
        validate_recorded_evidence(self.settings.paths.reports, request.evidence)
        if evidence_hash(request.evidence) != request.evidence_hash:
            raise DataValidationError("retraining request evidence hash mismatch")
        if request.policy_hash != self.retraining_policy.policy_hash:
            raise DataValidationError("retraining trigger policy changed after request creation")
        request_path = self.storage.requests_root / request_id / "training_request.json"
        manifest_path = self.storage.requests_root / request_id / "manifest.json"
        tracker.track(request_path)
        tracker.track(manifest_path)
        if manifest.request_file_sha256 != file_sha256(request_path):
            raise DataValidationError("retraining request manifest hash mismatch")
        processed = self.settings.paths.processed_data
        features_manifest = processed / "features_daily" / "_manifest.json"
        universe_manifest = processed / "universe_daily" / "_manifest.json"
        label_manifest = processed / "labels_forward" / "_manifest.json"
        for path in (features_manifest, universe_manifest, label_manifest):
            tracker.track(path)
        return (
            request_id,
            request,
            file_sha256(request_path),
            file_sha256(features_manifest),
            file_sha256(universe_manifest),
            file_sha256(label_manifest),
        )

    def _publish(
        self, report: RetrainingReadinessReport, hashes: dict[str, str]
    ) -> ReadinessResult:
        output = self.settings.paths.reports / "retraining" / "readiness" / report.as_of
        if output.exists():
            manifest_path = output / "manifest.json"
            if not manifest_path.is_file():
                raise DataValidationError("incomplete readiness output directory exists")
            try:
                existing = RetrainingReadinessReport.model_validate_json(
                    (output / "readiness.json").read_text(encoding="utf-8")
                )
                manifest = RetrainingReadinessManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                raise DataValidationError(
                    f"invalid existing readiness artifact: {error}"
                ) from error
            if existing != report:
                raise DataValidationError("immutable readiness identity differs")
            if (
                manifest.run_id != report.run_id
                or manifest.report_sha256 != file_sha256(output / "readiness.json")
                or manifest.markdown_sha256 != file_sha256(output / "report.md")
            ):
                raise DataValidationError("existing readiness artifact hash mismatch")
            return ReadinessResult(report, output, True)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging_root = output.parent / ".tmp"
        staging_root.mkdir(exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=staging_root, prefix="readiness_"))
        try:
            atomic_write_json(staging / "readiness.json", report.model_dump(mode="json"))
            (staging / "report.md").write_text(render_readiness(report), encoding="utf-8")
            RetrainingReadinessReport.model_validate_json(
                (staging / "readiness.json").read_text(encoding="utf-8")
            )
            if not (staging / "report.md").read_text(encoding="utf-8").strip():
                raise DataValidationError("staged readiness report is empty")
            git = current_git_info()
            manifest = RetrainingReadinessManifest(
                run_id=report.run_id,
                as_of=report.as_of,
                request_id=report.request_id,
                status=report.status,
                checks=report.checks,
                source_artifacts=tuple(sorted(hashes)),
                source_hashes=dict(sorted(hashes.items())),
                promotion_policy_hash=report.promotion_policy_hash,
                request_hash=report.request_hash,
                feature_hash=report.feature_hash,
                universe_hash=report.universe_hash,
                label_hash=report.label_hash,
                git_commit=git["commit"],
                git_dirty=bool(git["dirty"]),
                config_hash=config_hash(self.config_path),
                report_sha256=file_sha256(staging / "readiness.json"),
                markdown_sha256=file_sha256(staging / "report.md"),
                generated_at=self.now().astimezone(UTC).isoformat(),
            )
            atomic_write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
            os.replace(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return ReadinessResult(report, output)
