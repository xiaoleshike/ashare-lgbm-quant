"""Read-only promotion evidence validation and gate evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.promotion.contracts import validate_deployment_contract
from ashare_quant.models.promotion.evidence import verify_evidence_snapshot
from ashare_quant.models.promotion.gate_report import publish_gate_result, read_gate_result
from ashare_quant.models.promotion.gate_rules import (
    PromotionGatePolicy,
    performance_checks,
    review_checks,
)
from ashare_quant.models.promotion.gate_schemas import GateCheck, GateResult, GateStatus
from ashare_quant.models.promotion.schemas import EvidenceReference
from ashare_quant.models.promotion.storage import PromotionBundle, PromotionStorage
from ashare_quant.models.promotion.validation import validate_bundle
from ashare_quant.models.registry import ModelRegistry
from ashare_quant.models.shadow.storage import canonical_payload_hash, file_sha256
from ashare_quant.utils.manifest import utc_now_iso


@dataclass(frozen=True, slots=True)
class PromotionGateEvaluation:
    """Published gate result metadata."""

    request_id: str
    candidate_model_id: str
    status: str
    output_dir: Path
    checks: int
    idempotent: bool


class PromotionGateEngine:
    """Evaluate immutable evidence without changing any model lifecycle state."""

    def __init__(
        self,
        *,
        models_root: Path,
        reports_root: Path,
        policy: PromotionGatePolicy | None = None,
    ) -> None:
        self.models_root = models_root
        self.reports_root = reports_root
        self.registry = ModelRegistry(models_root)
        self.storage = PromotionStorage(models_root)
        self.policy = policy or PromotionGatePolicy()

    def evaluate(self, request_id: str) -> PromotionGateEvaluation:
        """Evaluate and immutably publish PASS, FAIL, or REVIEW_REQUIRED."""

        bundle = self.storage.read(request_id)
        if bundle is None:
            raise DataValidationError(f"complete promotion request does not exist: {request_id}")
        registry_before = file_sha256(self.registry.registry_path)
        checks: list[GateCheck] = []
        checks.extend(self._request_and_registry_checks(bundle, registry_before))
        payloads = self._evidence_payloads(bundle, checks)
        checks.extend(self._candidate_checks(bundle, payloads))
        checks.extend(self._deployment_checks(bundle, payloads))
        checks.extend(self._performance_checks(bundle, payloads))
        checks.extend(self._monitoring_checks(bundle, payloads))
        checks.extend(self._review_checks(bundle, payloads))
        ordered = tuple(sorted(checks, key=lambda item: item.name))
        status = (
            "FAIL"
            if any(item.status == "FAIL" for item in ordered)
            else (
                "REVIEW_REQUIRED" if any(item.status == "WARNING" for item in ordered) else "PASS"
            )
        )
        source_request_hash = file_sha256(bundle.output_dir / "manifest.json")
        source_state = self._source_state(bundle)
        gate_identity = canonical_payload_hash(
            {
                "request_manifest_hash": source_request_hash,
                "policy_hash": self.policy.policy_hash,
                "registry_hash": registry_before,
                "source_state": source_state,
            }
        )
        result = GateResult(
            request_id=request_id,
            candidate_model_id=bundle.request.candidate.model_id,
            status=cast(GateStatus, status),
            checks=ordered,
            policy_hash=self.policy.policy_hash,
            created_at=utc_now_iso(),
        )
        output_dir, idempotent = publish_gate_result(
            reports_root=self.reports_root,
            result=result,
            gate_identity=gate_identity,
            source_request_manifest_hash=source_request_hash,
        )
        if file_sha256(self.registry.registry_path) != registry_before:
            raise DataValidationError("registry changed during read-only promotion gate evaluation")
        return PromotionGateEvaluation(
            request_id=request_id,
            candidate_model_id=bundle.request.candidate.model_id,
            status=status,
            output_dir=output_dir,
            checks=len(ordered),
            idempotent=idempotent,
        )

    def status(self, request_id: str) -> PromotionGateEvaluation | None:
        """Read one previously published gate result."""

        output_dir = self.reports_root / "promotion_gate" / request_id
        stored = read_gate_result(output_dir)
        if stored is None:
            return None
        result, _ = stored
        return PromotionGateEvaluation(
            request_id=request_id,
            candidate_model_id=result.candidate_model_id,
            status=result.status,
            output_dir=output_dir,
            checks=len(result.checks),
            idempotent=True,
        )

    def _request_and_registry_checks(
        self, bundle: PromotionBundle, registry_hash: str
    ) -> tuple[GateCheck, ...]:
        evidence_hash = bundle.manifest.identity_hash
        checks: list[GateCheck] = []
        try:
            validate_bundle(bundle, self.reports_root, self.registry.registry_path)
        except (DataValidationError, OSError) as error:
            checks.append(_check("request_integrity", "FAIL", str(error), evidence_hash))
        else:
            checks.append(
                _check(
                    "request_integrity",
                    "PASS",
                    "request, evidence snapshot, and deployment contract hashes are valid",
                    evidence_hash,
                )
            )
        registry_matches = registry_hash == bundle.request.registry_hash
        checks.append(
            _check(
                "registry_precondition",
                "PASS" if registry_matches else "FAIL",
                "registry hash matches promotion request"
                if registry_matches
                else "registry_changed_after_request",
                registry_hash,
            )
        )
        champion = self.registry.get_champion(bundle.request.current_champion.model_type)
        assignment_matches = (
            champion is not None
            and champion.model_id == bundle.request.current_champion.model_id
            and champion.status == "champion"
            and file_sha256(Path(champion.artifact_path) / "manifest.json")
            == bundle.request.current_champion.artifact_manifest_sha256
        )
        checks.append(
            _check(
                "champion_assignment_precondition",
                "PASS" if assignment_matches else "FAIL",
                "current champion assignment is unchanged"
                if assignment_matches
                else "current champion assignment changed after request",
                registry_hash,
            )
        )
        return tuple(checks)

    def _evidence_payloads(
        self, bundle: PromotionBundle, checks: list[GateCheck]
    ) -> dict[str, dict[str, Any]]:
        try:
            verify_evidence_snapshot(bundle.evidence, self.reports_root)
        except (DataValidationError, OSError) as error:
            checks.append(
                _check(
                    "evidence_snapshot",
                    "FAIL",
                    str(error),
                    bundle.evidence.evidence_snapshot_hash,
                )
            )
        else:
            checks.append(
                _check(
                    "evidence_snapshot",
                    "PASS",
                    "all six evidence sources match frozen hashes and cutoff",
                    bundle.evidence.evidence_snapshot_hash,
                )
            )
        payloads: dict[str, dict[str, Any]] = {}
        for source in bundle.evidence.sources:
            path = (self.reports_root / source.source_path).resolve()
            try:
                payloads[source.evidence_type] = _load_json(path)
            except DataValidationError:
                continue
        return payloads

    def _candidate_checks(
        self, bundle: PromotionBundle, payloads: dict[str, dict[str, Any]]
    ) -> tuple[GateCheck, ...]:
        request = bundle.request
        evidence_hash = request.candidate.artifact_manifest_sha256
        model = next(
            (
                item
                for item in self.registry.list_models()
                if item.model_id == request.candidate.model_id
            ),
            None,
        )
        exists = model is not None
        status_ok = model is not None and model.status == "candidate"
        checks = [
            _check(
                "candidate_registered",
                "PASS" if exists else "FAIL",
                "candidate exists in registry" if exists else "candidate is absent from registry",
                evidence_hash,
            ),
            _check(
                "candidate_status",
                "PASS" if status_ok else "FAIL",
                "candidate registry status is candidate"
                if status_ok
                else "candidate registry status is not candidate",
                evidence_hash,
            ),
        ]
        if model is None:
            return tuple(checks)
        manifest_path = Path(model.artifact_path) / "manifest.json"
        manifest_hash_ok = (
            manifest_path.is_file()
            and file_sha256(manifest_path) == request.candidate.artifact_manifest_sha256
        )
        checks.append(
            _check(
                "candidate_artifact_manifest",
                "PASS" if manifest_hash_ok else "FAIL",
                "candidate artifact manifest hash matches request"
                if manifest_hash_ok
                else "candidate artifact manifest is missing or changed",
                evidence_hash,
            )
        )
        model_file_exists = (Path(model.artifact_path) / "model.txt").is_file()
        checks.append(
            _check(
                "candidate_model_artifact",
                "PASS" if model_file_exists else "FAIL",
                "candidate model artifact exists"
                if model_file_exists
                else "candidate model artifact is missing",
                evidence_hash,
            )
        )
        try:
            feature_payload = _load_json(Path(model.artifact_path) / "feature_list.json")
            features = feature_payload.get("features")
            computed = (
                feature_list_hash(tuple(str(item) for item in features))
                if isinstance(features, list) and features
                else ""
            )
        except DataValidationError:
            computed = ""
        feature_ok = computed == model.feature_hash == request.candidate.feature_hash
        checks.append(
            _check(
                "candidate_feature_hash",
                "PASS" if feature_ok else "FAIL",
                "candidate feature hash matches registry and request"
                if feature_ok
                else "candidate feature hash mismatch",
                evidence_hash,
            )
        )
        try:
            manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
        except DataValidationError:
            manifest = {}
        expected_universe = manifest.get("universe_hash")
        challenger_universe = payloads.get("challenger_evaluation", {}).get("universe_hash")
        shadow_universe = payloads.get("shadow_prediction", {}).get("universe_hash")
        universe_ok = bool(expected_universe) and all(
            value == expected_universe for value in (challenger_universe, shadow_universe)
        )
        checks.append(
            _check(
                "candidate_universe_hash",
                "PASS" if universe_ok else "FAIL",
                "candidate, evaluation, and shadow universe hashes match"
                if universe_ok
                else "candidate universe hash is missing or inconsistent",
                evidence_hash,
            )
        )
        return tuple(checks)

    def _deployment_checks(
        self, bundle: PromotionBundle, payloads: dict[str, dict[str, Any]]
    ) -> tuple[GateCheck, ...]:
        contract = bundle.contract
        evidence_hash = contract.deployment_contract_hash
        checks: list[GateCheck] = []
        try:
            validate_deployment_contract(contract)
        except DataValidationError as error:
            checks.append(_check("deployment_contract", "FAIL", str(error), evidence_hash))
            return tuple(checks)
        checks.append(
            _check(
                "deployment_contract",
                "PASS",
                "deployment contract hash and inference compatibility are valid",
                evidence_hash,
            )
        )
        challenger = payloads.get("challenger_evaluation", {})
        executable = payloads.get("executable_validation", {})
        horizon_ok = all(
            value == contract.horizon
            for value in (challenger.get("horizon"), executable.get("horizon"))
        )
        holding_ok = all(
            value == contract.holding_period
            for value in (challenger.get("holding_period"), executable.get("holding_period"))
        )
        feature_ok = challenger.get("feature_hash") == contract.feature_hash
        execution_ok = challenger.get("execution_rule") == contract.execution_rule and (
            executable.get("execution_rule")
            == "signal_close_t_next_open_entry_and_horizon_open_exit"
        )
        for name, passed, message in (
            ("deployment_horizon", horizon_ok, "horizon"),
            ("deployment_holding_period", holding_ok, "holding_period"),
            ("deployment_feature_hash", feature_ok, "feature_hash"),
            ("deployment_execution_rule", execution_ok, "execution_rule"),
        ):
            checks.append(
                _check(
                    name,
                    "PASS" if passed else "FAIL",
                    f"evidence {message} matches deployment contract"
                    if passed
                    else f"evidence {message} differs from deployment contract",
                    evidence_hash,
                )
            )
        return tuple(checks)

    def _performance_checks(
        self, bundle: PromotionBundle, payloads: dict[str, dict[str, Any]]
    ) -> tuple[GateCheck, ...]:
        manifest = payloads.get("performance_observation")
        source = _source(bundle, "performance_observation")
        if manifest is None or source is None:
            return (
                _check(
                    "performance_observation",
                    "FAIL",
                    "performance observation evidence is missing",
                    bundle.evidence.evidence_snapshot_hash,
                ),
            )
        metrics_path = (self.reports_root / source.source_path).parent / "metrics.json"
        expected_hash = manifest.get("metrics_file_sha256")
        if not metrics_path.is_file() or file_sha256(metrics_path) != expected_hash:
            return (
                _check(
                    "performance_metrics_hash",
                    "FAIL",
                    "performance metrics file is missing or differs from manifest",
                    source.sha256,
                ),
            )
        metrics = _load_json(metrics_path)
        return performance_checks(
            manifest=manifest,
            metrics=metrics,
            candidate_model_id=bundle.request.candidate.model_id,
            evidence_hash=source.sha256,
            policy=self.policy,
        )

    def _monitoring_checks(
        self, bundle: PromotionBundle, payloads: dict[str, dict[str, Any]]
    ) -> tuple[GateCheck, ...]:
        alert_manifest = payloads.get("alerts")
        source = _source(bundle, "alerts")
        if alert_manifest is None or source is None:
            return (
                _check(
                    "monitoring_alerts",
                    "FAIL",
                    "alerts manifest is missing",
                    bundle.evidence.evidence_snapshot_hash,
                ),
            )
        alerts_path = (self.reports_root / source.source_path).parent / "alerts.json"
        if not alerts_path.is_file() or file_sha256(alerts_path) != alert_manifest.get(
            "alerts_file_sha256"
        ):
            return (
                _check(
                    "monitoring_alerts_hash",
                    "FAIL",
                    "alerts payload is missing or differs from manifest",
                    source.sha256,
                ),
            )
        payload = _load_json(alerts_path)
        raw_alerts = payload.get("alerts")
        alerts = raw_alerts if isinstance(raw_alerts, list) else []
        active = [
            item
            for item in alerts
            if isinstance(item, dict)
            and item.get("model_id") == bundle.request.candidate.model_id
            and item.get("status") in {"NEW", "ACTIVE"}
        ]
        critical = [item for item in active if item.get("severity") == "CRITICAL"]
        warning = [item for item in active if item.get("severity") == "WARNING"]
        return (
            _check(
                "candidate_critical_alerts",
                "FAIL" if critical else "PASS",
                f"unresolved candidate CRITICAL alerts={len(critical)}",
                source.sha256,
            ),
            _check(
                "candidate_warning_alerts",
                "WARNING" if warning else "PASS",
                f"unresolved candidate WARNING alerts={len(warning)}",
                source.sha256,
            ),
        )

    def _review_checks(
        self, bundle: PromotionBundle, payloads: dict[str, dict[str, Any]]
    ) -> tuple[GateCheck, ...]:
        challenger = payloads.get("challenger_evaluation", {})
        executable = payloads.get("executable_validation", {})
        monitoring = payloads.get("monitoring_summary", {})
        return review_checks(
            challenger_manifest=challenger,
            executable_manifest=executable,
            monitoring_summary=monitoring,
            evidence_hash=bundle.evidence.evidence_snapshot_hash,
            policy=self.policy,
        )

    def _source_state(self, bundle: PromotionBundle) -> list[dict[str, str]]:
        state: list[dict[str, str]] = []
        for source in bundle.evidence.sources:
            path = self.reports_root / source.source_path
            state.append(
                {
                    "evidence_type": source.evidence_type,
                    "current_hash": file_sha256(path) if path.is_file() else "missing",
                }
            )
        return state


def _source(bundle: PromotionBundle, evidence_type: str) -> EvidenceReference | None:
    return next(
        (item for item in bundle.evidence.sources if item.evidence_type == evidence_type), None
    )


def _check(name: str, status: str, message: str, evidence_hash: str) -> GateCheck:
    from typing import cast

    from ashare_quant.models.promotion.gate_schemas import GateCheckStatus

    return GateCheck(
        name=name,
        status=cast(GateCheckStatus, status),
        message=message,
        evidence_hash=evidence_hash,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataValidationError(f"required gate evidence does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataValidationError(f"invalid gate evidence JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise DataValidationError(f"gate evidence must contain an object: {path}")
    return payload
