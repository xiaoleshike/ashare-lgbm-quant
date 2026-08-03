"""Read-only collection of current production governance status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.governance.schemas import GovernanceCheck
from ashare_quant.models.promotion.approval_schema import ApprovalEvent
from ashare_quant.models.shadow.storage import file_sha256

_DATE_LENGTH = 8


class SourceCatalog:
    """Track every file consumed by a governance report."""

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def json(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"JSON must contain an object: {path}")
        self.hashes[str(path)] = file_sha256(path)
        return payload

    def track(self, path: Path) -> None:
        self.hashes[str(path)] = file_sha256(path)


def collect_governance_status(
    *,
    settings: AppSettings,
    project_root: Path,
    sources: SourceCatalog,
) -> tuple[dict[str, Any], list[GovernanceCheck]]:
    """Collect a stable cross-system status projection without side effects."""

    checks: list[GovernanceCheck] = []
    production = _production_status(settings.paths.reports, project_root, sources, checks)
    champion = _champion_status(settings.paths.models, sources, checks)
    monitoring = _monitoring_status(settings.paths.reports, sources, checks)
    paper = _paper_status(settings.paths.paper_trading, settings, sources, checks)
    promotion = _promotion_status(settings.paths.models, sources, checks)
    rollback = _rollback_status(settings.paths.models, champion.get("model_id"), sources)
    research = _research_status(settings.paths.reports, sources, checks)
    return {
        "production": production,
        "champion": champion,
        "monitoring": monitoring,
        "paper_trading": paper,
        "promotion": promotion,
        "rollback": rollback,
        "research_agent": research,
    }, checks


def _production_status(
    reports_root: Path,
    project_root: Path,
    sources: SourceCatalog,
    checks: list[GovernanceCheck],
) -> dict[str, Any]:
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    if reports_root.is_dir():
        for directory in reports_root.iterdir():
            if not _is_date_dir(directory):
                continue
            path = directory / "production_summary.json"
            if not path.is_file():
                continue
            try:
                payload = sources.json(path)
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            candidates.append((directory.name, path, payload))
    if not candidates:
        checks.append(
            GovernanceCheck(
                name="production.latest",
                status="WARNING",
                message="no successful production summary found",
            )
        )
        return {
            "latest_run_id": None,
            "latest_success_time": None,
            "pipeline_status": "missing",
            "summary_path": None,
        }
    _, path, payload = max(candidates, key=lambda item: (item[0], str(item[1])))
    run_id = str(payload.get("run_id") or "") or None
    run_manifest = _find_run_manifest(project_root / "runs", run_id)
    pipeline_status = "success"
    if run_manifest is not None:
        run_payload = sources.json(run_manifest)
        pipeline_status = str(run_payload.get("status") or "unknown")
    checks.append(
        GovernanceCheck(
            name="production.latest",
            status="PASS" if pipeline_status == "success" else "WARNING",
            message=f"latest production publication is {pipeline_status}",
            source_path=str(path),
        )
    )
    return {
        "latest_run_id": run_id,
        "latest_success_time": payload.get("completed_time"),
        "pipeline_status": pipeline_status,
        "summary_path": str(path),
        "run_manifest_path": None if run_manifest is None else str(run_manifest),
    }


def _champion_status(
    models_root: Path, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> dict[str, Any]:
    registry = models_root / "registry.json"
    if not registry.is_file():
        checks.append(
            GovernanceCheck(
                name="registry.current",
                status="FAIL",
                message="model registry is missing",
                source_path=str(registry),
            )
        )
        return {
            "model_id": None,
            "assignment_id": None,
            "registry_version": None,
            "last_change": None,
        }
    try:
        payload = sources.json(registry)
        models = payload.get("models")
        champions = (
            [item for item in models if isinstance(item, dict) and item.get("status") == "champion"]
            if isinstance(models, list)
            else []
        )
    except (OSError, json.JSONDecodeError, ValueError):
        champions = []
        payload = {}
    if len(champions) != 1:
        checks.append(
            GovernanceCheck(
                name="registry.champion",
                status="FAIL",
                message=f"expected one Champion, found {len(champions)}",
                source_path=str(registry),
            )
        )
        model_id = None
    else:
        model_id = str(champions[0].get("model_id"))
        checks.append(
            GovernanceCheck(
                name="registry.champion",
                status="PASS",
                message=f"Champion={model_id}",
                source_path=str(registry),
            )
        )
    assignments = _champion_assignments(models_root, sources)
    current = next(
        (item for item in reversed(assignments) if item.get("model_id") == model_id), None
    )
    if assignments and current is None:
        checks.append(
            GovernanceCheck(
                name="champion.assignment",
                status="FAIL",
                message="Champion does not match immutable assignment history",
            )
        )
    elif not assignments:
        checks.append(
            GovernanceCheck(
                name="champion.assignment",
                status="WARNING",
                message="Champion assignment history is unavailable for this legacy registry",
            )
        )
    return {
        "model_id": model_id,
        "assignment_id": None if current is None else current.get("champion_assignment_id"),
        "registry_version": payload.get("registry_version_id"),
        "last_change": payload.get("updated_at")
        if current is None
        else current.get("activated_at"),
    }


def _monitoring_status(
    reports_root: Path, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> dict[str, Any]:
    root = reports_root / "model_monitor"
    paths = (
        sorted(
            path for path in root.glob("????????/monitor_summary.json") if _is_date_dir(path.parent)
        )
        if root.exists()
        else []
    )
    if not paths:
        checks.append(
            GovernanceCheck(
                name="monitor.latest", status="WARNING", message="no monitoring snapshot found"
            )
        )
        return {"latest_monitor_date": None, "critical_alerts": 0, "warning_alerts": 0}
    path = paths[-1]
    try:
        payload = sources.json(path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        checks.append(
            GovernanceCheck(
                name="monitor.latest",
                status="FAIL",
                message=f"latest monitor summary is invalid: {error}",
                source_path=str(path),
            )
        )
        return {
            "latest_monitor_date": path.parent.name,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "summary_path": str(path),
        }
    alerts = payload.get("alerts")
    rows = alerts.get("alerts", []) if isinstance(alerts, dict) else []
    critical = sum(
        1
        for item in rows
        if isinstance(item, dict)
        and item.get("severity") == "CRITICAL"
        and item.get("status") != "RECOVERED"
    )
    warning = sum(
        1
        for item in rows
        if isinstance(item, dict)
        and item.get("severity") == "WARNING"
        and item.get("status") != "RECOVERED"
    )
    checks.append(
        GovernanceCheck(
            name="monitor.latest",
            status="PASS",
            message=f"monitor snapshot={path.parent.name}",
            source_path=str(path),
        )
    )
    return {
        "latest_monitor_date": path.parent.name,
        "critical_alerts": critical,
        "warning_alerts": warning,
        "summary_path": str(path),
    }


def _paper_status(
    paper_root: Path, settings: AppSettings, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    for configured in settings.paper_trading.portfolios:
        path = paper_root / configured.portfolio_id / "equity_curve.parquet"
        if not path.is_file():
            metrics.append(
                {"portfolio_id": configured.portfolio_id, "latest_nav": None, "max_drawdown": None}
            )
            continue
        try:
            sources.track(path)
            frame = pd.read_parquet(path, columns=["as_of", "nav", "drawdown"])
        except Exception as error:
            checks.append(
                GovernanceCheck(
                    name=f"paper_trading.{configured.portfolio_id}",
                    status="FAIL",
                    message=f"equity ledger is invalid: {error}",
                    source_path=str(path),
                )
            )
            metrics.append(
                {
                    "portfolio_id": configured.portfolio_id,
                    "latest_nav": None,
                    "max_drawdown": None,
                }
            )
            continue
        if frame.empty:
            metrics.append(
                {"portfolio_id": configured.portfolio_id, "latest_nav": None, "max_drawdown": None}
            )
            continue
        ordered = frame.sort_values("as_of", kind="mergesort")
        metrics.append(
            {
                "portfolio_id": configured.portfolio_id,
                "latest_nav": float(ordered.iloc[-1]["nav"]),
                "max_drawdown": float(pd.to_numeric(ordered["drawdown"], errors="coerce").min()),
            }
        )
    checks.append(
        GovernanceCheck(
            name="paper_trading.status",
            status="PASS" if metrics else "WARNING",
            message=f"paper portfolios={len(metrics)}",
        )
    )
    return {"portfolios": len(metrics), "portfolio_metrics": metrics}


def _promotion_status(
    models_root: Path, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> dict[str, Any]:
    root = models_root / "promotion_requests"
    pending = 0
    approved = 0
    invalid: list[str] = []
    recent: list[str] = []
    if root.exists():
        for request_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            complete = (request_dir / "manifest.json").is_file()
            if not complete:
                continue
            sources.track(request_dir / "manifest.json")
            applied = (
                list((request_dir / "apply").glob("*/manifest.json"))
                if (request_dir / "apply").exists()
                else []
            )
            events = (
                sorted((request_dir / "approval_events").glob("*.json"))
                if (request_dir / "approval_events").exists()
                else []
            )
            event_paths = [path for path in events if not path.name.endswith(".manifest.json")]
            if applied:
                recent.append(request_dir.name)
            elif event_paths:
                try:
                    event = ApprovalEvent.model_validate(sources.json(event_paths[0]))
                    approved += int(event.event_type == "APPROVED")
                except Exception:
                    invalid.append(request_dir.name)
            else:
                pending += 1
    checks.append(
        GovernanceCheck(
            name="promotion.requests",
            status="FAIL" if invalid else "PASS",
            message=(
                f"invalid requests={invalid}"
                if invalid
                else f"pending={pending} approved_pending_apply={approved}"
            ),
        )
    )
    return {
        "pending_requests": pending,
        "approved_pending_apply": approved,
        "recent_promotions": recent[-10:],
        "invalid_requests": invalid,
    }


def _rollback_status(
    models_root: Path, current_model: object, sources: SourceCatalog
) -> dict[str, Any]:
    previous = {
        str(item.get("previous_champion_model_id"))
        for item in _champion_assignments(models_root, sources)
        if item.get("previous_champion_model_id")
    }
    previous.discard(str(current_model))
    return {"available_historical_champions": sorted(previous)}


def _research_status(
    reports_root: Path, sources: SourceCatalog, checks: list[GovernanceCheck]
) -> dict[str, Any]:
    root = reports_root / "research_agent"
    paths = (
        sorted(
            path
            for path in root.glob("????????/research_summary.json")
            if _is_date_dir(path.parent)
        )
        if root.exists()
        else []
    )
    if not paths:
        checks.append(
            GovernanceCheck(
                name="research_agent.latest",
                status="WARNING",
                message="no research-agent report found",
            )
        )
        return {"latest_report": None, "generation_mode": None}
    try:
        payload = sources.json(paths[-1])
    except (OSError, json.JSONDecodeError, ValueError) as error:
        checks.append(
            GovernanceCheck(
                name="research_agent.latest",
                status="FAIL",
                message=f"latest research-agent report is invalid: {error}",
                source_path=str(paths[-1]),
            )
        )
        return {"latest_report": str(paths[-1]), "generation_mode": None}
    checks.append(
        GovernanceCheck(
            name="research_agent.latest",
            status="PASS",
            message=f"latest report={paths[-1].parent.name}",
            source_path=str(paths[-1]),
        )
    )
    return {"latest_report": str(paths[-1]), "generation_mode": payload.get("generation_mode")}


def _champion_assignments(models_root: Path, sources: SourceCatalog) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = models_root / "champion_history"
    if root.exists():
        for path in sorted(root.glob("*.json")):
            try:
                rows.append(sources.json(path))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
    return sorted(
        rows,
        key=lambda item: (
            str(item.get("activated_at") or ""),
            str(item.get("champion_assignment_id") or ""),
        ),
    )


def _find_run_manifest(runs_root: Path, run_id: str | None) -> Path | None:
    if not run_id or not runs_root.exists():
        return None
    matches = list(runs_root.glob(f"????????/{run_id}/manifest.json"))
    return matches[0] if len(matches) == 1 else None


def resolve_artifact_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _is_date_dir(path: Path) -> bool:
    return path.is_dir() and len(path.name) == _DATE_LENGTH and path.name.isdigit()
