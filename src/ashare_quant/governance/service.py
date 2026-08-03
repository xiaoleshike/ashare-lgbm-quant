"""Application service for read-only governance operations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from ashare_quant.config.settings import AppSettings
from ashare_quant.governance.recovery import (
    RegistryRecoveryPreview,
    registry_recovery_preview,
    validate_recovery_state,
)
from ashare_quant.governance.reporting import GovernanceReportPublisher
from ashare_quant.governance.schemas import GovernanceCheck, GovernanceReport, overall_status
from ashare_quant.governance.status import SourceCatalog, collect_governance_status
from ashare_quant.governance.validation import validate_production_state


@dataclass(frozen=True, slots=True)
class GovernanceResult:
    """One published governance operation result."""

    report: GovernanceReport
    report_path: Path
    manifest_path: Path


class GovernanceService:
    """Read immutable operational artifacts and publish governance reports."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        config_path: Path,
        project_root: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.config_path = config_path
        self.project_root = project_root or _project_root(config_path)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.publisher = GovernanceReportPublisher(settings.paths.reports)

    def status(self) -> GovernanceResult:
        """Publish one read-only whole-system status projection."""

        sources = SourceCatalog()
        summary, checks = collect_governance_status(
            settings=self.settings,
            project_root=self.project_root,
            sources=sources,
        )
        return self._publish("status", summary, checks, sources)

    def validate_production(self) -> GovernanceResult:
        """Validate current production outputs and governance preconditions."""

        sources = SourceCatalog()
        summary, checks = validate_production_state(
            settings=self.settings,
            config_path=self.config_path,
            project_root=self.project_root,
            sources=sources,
            now=self.clock(),
        )
        return self._publish("validation", summary, checks, sources)

    def validate_recovery(self) -> GovernanceResult:
        """Validate that immutable versions can support manual recovery."""

        sources = SourceCatalog()
        summary, checks = validate_recovery_state(settings=self.settings, sources=sources)
        return self._publish("recovery", summary, checks, sources)

    def recover_registry_dry_run(self) -> RegistryRecoveryPreview:
        """Return the latest recoverable Registry identity without restoring bytes."""

        return registry_recovery_preview(self.settings.paths.models)

    def _publish(
        self,
        report_type: Literal["status", "validation", "recovery"],
        summary: dict[str, object],
        checks: list[GovernanceCheck],
        sources: SourceCatalog,
    ) -> GovernanceResult:
        report = GovernanceReport(
            artifact_name=f"governance_{report_type}_report",
            report_type=report_type,
            status=overall_status(checks),
            generated_at=self.clock().astimezone(UTC).isoformat(),
            summary=summary,
            checks=tuple(checks),
            source_hashes=dict(sorted(sources.hashes.items())),
        )
        report_path, manifest_path = self.publisher.publish(report)
        return GovernanceResult(report, report_path, manifest_path)


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    return resolved.parent.parent if resolved.parent.name == "config" else Path.cwd().resolve()
