"""Immutable continuous STRICT_OOS replay using frozen predictions and the existing engine."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.backtest.continuous_evidence import (
    accounting_evidence,
    comparison_metrics,
    evidence_summary,
    period_metrics,
)
from ashare_quant.backtest.continuous_source import (
    ContinuousSource,
    file_hash,
    load_continuous_source,
    payload_hash,
)
from ashare_quant.backtest.corporate_actions import default_corporate_action_execution_policy
from ashare_quant.backtest.costs import ExecutionCostPolicy
from ashare_quant.backtest.data import load_benchmark, load_calendar, load_execution_prices
from ashare_quant.backtest.engine import (
    ACCOUNTING_SCHEMA_VERSION,
    BacktestInputs,
    BacktestResult,
    simulate_portfolio,
)
from ashare_quant.backtest.executable_validation import REQUIRED_TOP_N, _signals
from ashare_quant.config.settings import AppSettings, BacktestSettings, load_settings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.research_source_snapshot import validate_research_source_snapshot
from ashare_quant.data.security_identity import SecurityIdentityResolver
from ashare_quant.data.security_identity_transition import SecurityIdentityTransitionResolver
from ashare_quant.data.security_lifecycle import SecurityLifecycleResolver
from ashare_quant.data.security_lifecycle_audit import (
    LifecycleAuditPolicy,
    validate_pass_lifecycle_scan,
)
from ashare_quant.data.security_listing_metadata import SecurityListingMetadataResolver
from ashare_quant.models.research_policy import load_research_policy
from ashare_quant.models.walk_forward_evaluation import (
    _execution_signal_codes,
    _governed_execution_cutoff,
    _validated_execution_tail_contract,
)
from ashare_quant.utils.manifest import atomic_write_json

CONTRACT = "continuous_strict_oos_replay_v1"
CAPITAL_POLICY = "existing_engine_all_available_cash_v1"
TABLES = (
    "daily_returns",
    "trades",
    "holdings",
    "execution_intents",
    "corporate_actions",
    "terminal_events",
)


@dataclass(frozen=True)
class ReplayPreflight:
    """Validated immutable sources; construction never calls the simulator."""

    source: ContinuousSource
    logical: dict[str, Any]
    calendar: tuple[str, ...]
    settings: BacktestSettings
    identity: SecurityIdentityResolver
    lifecycle: SecurityLifecycleResolver
    transitions: SecurityIdentityTransitionResolver

    @property
    def run_id(self) -> str:
        return f"continuous_strict_oos_{payload_hash(self.logical)[:24]}"

    def summary(self) -> dict[str, Any]:
        """Small operator-visible source check, not a performance result."""
        return {
            "status": "PREFLIGHT_PASS",
            "continuous_run_id": self.run_id,
            **{k: v for k, v in self.source.identity.items() if k not in {"selected_folds"}},
            "strict_oos_folds": len(self.source.identity["selected_folds"]),
            "duplicate_prediction_keys": 0,
            "non_finite_scores": 0,
            "execution_inputs": self.logical["execution_inputs"],
            "real_continuous_replay_status": "NOT_RUN",
        }


class ContinuousStrictOOSReplayService:
    """One account per frozen Top-N; no labels, training or source mutation."""

    def __init__(
        self,
        *,
        source_run: Path,
        raw_root: Path,
        execution_processed_root: Path,
        lifecycle_scan_manifest: Path,
        identity_transition_artifact: Path,
        config_path: Path,
        output_root: Path,
    ) -> None:
        self.source_run = source_run
        self.raw_root = raw_root
        self.execution_root = execution_processed_root
        self.scan = lifecycle_scan_manifest
        self.transitions_path = identity_transition_artifact
        self.config_path = config_path
        self.output_root = output_root

    def preflight(self) -> ReplayPreflight:
        """Validate sources and all planned exits without reading labels or fitting models."""
        source = load_continuous_source(self.source_run)
        settings = load_settings(self.config_path)
        if file_hash(self.config_path) != source.manifest["experiment"]["config_hash"]:
            raise DataValidationError("CONTINUOUS_SOURCE_CONFIG_MISMATCH")
        return self._execution_context(source, settings)

    def _execution_context(
        self, source: ContinuousSource, settings: AppSettings
    ) -> ReplayPreflight:
        contract = source.manifest["execution_contract"]
        tail = _validated_execution_tail_contract(contract)
        policy = load_research_policy(Path(source.manifest["research_policy_path"]))
        if (
            policy.policy_hash != source.manifest["research_policy_hash"]
            or tail["prospective_lockbox_start_exclusive"] != policy.prospective_lockbox.start_date
        ):
            raise DataValidationError("CONTINUOUS_RESEARCH_POLICY_MISMATCH")
        cutoff = _governed_execution_cutoff(tail)
        snapshot_root = self.raw_root.resolve().parent
        if self.raw_root.resolve().name != "datasets":
            raise DataValidationError("CONTINUOUS_FROZEN_SNAPSHOT_REQUIRED")
        snapshot = validate_research_source_snapshot(snapshot_root)
        identity = SecurityIdentityResolver.from_path(settings.security_identity.mapping_path)
        lifecycle = SecurityLifecycleResolver.from_path(settings.security_identity.lifecycle_path)
        transitions = SecurityIdentityTransitionResolver.from_path(self.transitions_path)
        transitions.validate_alias_coexistence(identity)
        metadata = SecurityListingMetadataResolver.from_path(
            settings.security_identity.listing_metadata_path
        )
        execution = settings.backtest.model_copy(
            update={"holding_period_days": 5, "execution": "next_open", "top_n": REQUIRED_TOP_N}
        )
        corporate = default_corporate_action_execution_policy()
        universe_hash = file_hash(self.execution_root / "universe_daily/_manifest.json")
        checks = {
            "execution_universe_manifest_hash": universe_hash,
            "cost_policy_hash": ExecutionCostPolicy.from_backtest_settings(execution).policy_hash,
            "corporate_action_execution_policy_hash": corporate.policy_hash,
            "accounting_schema_version": ACCOUNTING_SCHEMA_VERSION,
            "execution_mode": execution.execution,
            "holding_period_days": 5,
            "sell_delay_max_days": execution.sell_delay_max_days,
            "top_n": list(REQUIRED_TOP_N),
        }
        if (
            any(contract.get(k) != v for k, v in checks.items())
            or transitions.artifact_hash
            != contract["security_identity_transitions"]["security_identity_transition_hash"]
        ):
            raise DataValidationError("CONTINUOUS_EXECUTION_CONTRACT_MISMATCH")
        scan = validate_pass_lifecycle_scan(
            self.scan,
            required_start=source.identity["signal_start"],
            required_end=cutoff,
            raw_root=self.raw_root,
            processed_root=self.execution_root,
            identity_resolver=identity,
            lifecycle_evidence=lifecycle,
            identity_transitions=transitions,
            listing_metadata=metadata,
            lifecycle_policy=LifecycleAuditPolicy.from_path(
                settings.security_identity.lifecycle_policy_path
            ),
        )
        if (
            file_hash(self.scan)
            != contract["security_lifecycle_audit"]["lifecycle_scan_manifest_hash"]
        ):
            raise DataValidationError("CONTINUOUS_LIFECYCLE_IDENTITY_MISMATCH")
        calendar = tuple(
            load_calendar(
                self.raw_root, source.identity["signal_start"], cutoff, None, maximum_date=cutoff
            )
        )
        validate_signal_calendar(source, calendar)
        logical = {
            "contract_version": CONTRACT,
            "source": source.identity,
            "execution_inputs": {
                **checks,
                "raw_snapshot_id": snapshot["snapshot_id"],
                "raw_snapshot_manifest_hash": file_hash(snapshot_root / "manifest.json"),
                "lifecycle_scan_id": scan["scan_id"],
                "lifecycle_scan_manifest_hash": file_hash(self.scan),
                "source_inventory_hash": scan["source_inventory_hash"],
                "identity_transition_version": transitions.artifact_version,
                "identity_transition_hash": transitions.artifact_hash,
                "identity_mapping_hash": identity.mapping_hash,
                "typed_lifecycle_hash": lifecycle.policy_hash,
                "listing_metadata_hash": metadata.overlay_hash,
                "corporate_action_policy_version": corporate.version,
                "research_policy_hash": policy.policy_hash,
                "governed_execution_cutoff": cutoff,
                "lockbox_start": policy.prospective_lockbox.start_date,
                "calendar_hash": payload_hash(calendar),
                "price_tolerance": settings.universe.price_tolerance,
            },
            "capital_deployment_policy": CAPITAL_POLICY,
            "delayed_exit_policy": "carry_to_calendar_end",
            "purpose": "oos_evidence",
            "backtest_settings": execution.model_dump(mode="json"),
            "implementation_hash": implementation_hash(),
        }
        return ReplayPreflight(
            source, logical, calendar, execution, identity, lifecycle, transitions
        )

    def run(self) -> Path:
        """Explicit operator operation: simulate once per Top-N and publish atomically."""
        context = self.preflight()
        target = self.output_root / context.run_id
        if target.exists():
            validate_continuous_strict_oos_artifact(target)
            return target
        signals = _signals(
            context.source.predictions[["trade_date", "ts_code", "prediction_score"]]
        )
        settings = load_settings(self.config_path)
        prices = load_execution_prices(
            self.raw_root,
            self.execution_root,
            context.calendar[0],
            context.calendar[-1],
            settings.universe.price_tolerance,
            identity_resolver=context.identity,
            lifecycle_resolver=context.lifecycle,
            identity_transitions=context.transitions,
            ts_codes=_execution_signal_codes(signals, max(REQUIRED_TOP_N)),
        )
        benchmark = load_benchmark(
            self.raw_root,
            context.settings.benchmark_index_code,
            context.calendar[0],
            context.calendar[-1],
        )
        inputs = BacktestInputs(
            signals,
            prices,
            context.calendar,
            benchmark,
            identity_transitions=context.transitions.transition_records(),
            identity_transition_version=context.transitions.artifact_version,
            identity_transition_hash=context.transitions.artifact_hash,
        )
        results = []
        for top_n in REQUIRED_TOP_N:
            try:
                results.append(
                    simulate_portfolio(
                        inputs,
                        top_n=top_n,
                        settings=context.settings,
                        purpose="oos_evidence",
                        delayed_exit_policy="carry_to_calendar_end",
                        record_execution_audit=True,
                    )
                )
            except DataValidationError as error:
                if "CORPORATE_ACTION_EXECUTION_UNSUPPORTED" in str(error):
                    raise DataValidationError(
                        f"UNSUPPORTED_REAL_TRANSITION_CROSSING: {error}"
                    ) from error
                raise
        return self._publish(context, results)

    def _locators(self) -> dict[str, str]:
        return {
            "source_run": str(self.source_run.resolve()),
            "raw_root": str(self.raw_root.resolve()),
            "execution_processed_root": str(self.execution_root.resolve()),
            "lifecycle_scan_manifest": str(self.scan.resolve()),
            "identity_transition_artifact": str(self.transitions_path.resolve()),
            "config_path": str(self.config_path.resolve()),
        }

    def _publish(self, context: ReplayPreflight, results: list[BacktestResult]) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        target = self.output_root / context.run_id
        with tempfile.TemporaryDirectory(
            prefix=".continuous-staging-", dir=self.output_root
        ) as temporary:
            stage = Path(temporary) / context.run_id
            stage.mkdir()
            context.source.predictions.to_parquet(
                stage / "stitched_predictions.parquet", index=False
            )
            context.source.lineage.to_parquet(stage / "signal_lineage.parquet", index=False)
            atomic_write_json(
                stage / "source_inventory.json",
                {"logical_identity": context.logical, "locators": self._locators()},
            )
            for result in results:
                folder = stage / f"top_{result.top_n}"
                folder.mkdir()
                for name in TABLES:
                    getattr(result, name).to_parquet(folder / f"{name}.parquet", index=False)
                reconciliation = accounting_evidence(result, context.settings)
                reconciliation.to_parquet(
                    folder / "daily_accounting_reconciliation.parquet", index=False
                )
                atomic_write_json(folder / "metrics.json", result.metrics)
                atomic_write_json(folder / "accounting_summary.json", result.accounting_summary)
                atomic_write_json(
                    folder / "evidence_summary.json",
                    evidence_summary(result, reconciliation, context.settings),
                )
            for frequency in ("month", "year"):
                pd.concat(
                    [period_metrics(r, frequency) for r in results], ignore_index=True
                ).to_csv(stage / f"{frequency}ly_metrics.csv", index=False)
            atomic_write_json(
                stage / "comparison.json",
                comparison_metrics(
                    results, context.settings, context.source.identity["signal_end"]
                ),
            )
            manifest = {
                "schema_version": 1,
                "artifact_name": CONTRACT,
                "status": "COMPLETE",
                "run_id": context.run_id,
                "identity": payload_hash(context.logical),
                "logical_identity": context.logical,
                "terminal_writeoff_assumption": "current_zero_cash_recovery",
                "artifact_hashes": {
                    p.relative_to(stage).as_posix(): file_hash(p)
                    for p in sorted(stage.rglob("*"))
                    if p.is_file()
                },
            }
            atomic_write_json(stage / "manifest.json", manifest)
            validate_continuous_strict_oos_artifact(stage)
            if target.exists():
                validate_continuous_strict_oos_artifact(target)
            else:
                stage.rename(target)
        return target


def implementation_hash() -> str:
    """Bind task-owned content and engine semantics, including uncommitted reviewed code."""
    root = Path(__file__).parent
    names = (
        "continuous.py",
        "continuous_source.py",
        "continuous_evidence.py",
        "continuous_validation.py",
        "engine.py",
        "data.py",
        "costs.py",
        "corporate_actions.py",
    )
    dependencies = (
        "models/walk_forward_evaluation.py",
        "config/settings.py",
        "data/security_lifecycle_audit.py",
        "data/security_identity_transition.py",
        "data/security_identity.py",
        "data/security_lifecycle.py",
        "data/security_listing_metadata.py",
        "data/research_source_snapshot.py",
    )
    return payload_hash(
        {
            **{f"backtest/{name}": file_hash(root / name) for name in names},
            **{name: file_hash(root.parent / name) for name in dependencies},
        }
    )


def validate_signal_calendar(source: ContinuousSource, calendar: tuple[str, ...]) -> None:
    """No missing signal sessions or clamped H+1 exits, using the frozen exchange calendar."""
    if not calendar or list(calendar) != sorted(set(calendar)):
        raise DataValidationError("CONTINUOUS_INVALID_CALENDAR")
    for fold in source.identity["selected_folds"]:
        expected = [d for d in calendar if fold["evaluation_start"] <= d <= fold["evaluation_end"]]
        actual = source.lineage.loc[
            source.lineage.source_fold_id == fold["source_fold_id"], "trade_date"
        ].tolist()
        if not expected or expected != actual:
            raise DataValidationError("CONTINUOUS_MISSING_SIGNAL_SESSION")
        if calendar.index(expected[-1]) + 6 >= len(calendar):
            raise DataValidationError("WALK_FORWARD_EXECUTION_TAIL_INSUFFICIENT")


def validate_continuous_strict_oos_artifact(path: Path) -> dict[str, Any]:
    """Validate root-to-leaf plus source provenance and accounting-derived evidence."""
    from ashare_quant.backtest.continuous_validation import validate_continuous_artifact

    return validate_continuous_artifact(path)
