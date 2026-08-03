"""Read-only validation of the current production governance state."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_quant.config.settings import AppSettings
from ashare_quant.governance.schemas import GovernanceCheck
from ashare_quant.governance.status import (
    SourceCatalog,
    collect_governance_status,
    resolve_artifact_path,
)
from ashare_quant.models.feature_lists import feature_list_hash
from ashare_quant.models.promotion.approval_schema import ApprovalEvent
from ashare_quant.models.promotion.registry_versions import load_registry_records
from ashare_quant.models.shadow.storage import file_sha256

_PAPER_KEYS = {
    "orders.parquet": "order_id",
    "trades.parquet": "trade_id",
    "positions.parquet": "event_id",
    "equity_curve.parquet": "equity_id",
}


def validate_production_state(
    *,
    settings: AppSettings,
    config_path: Path,
    project_root: Path,
    sources: SourceCatalog,
    now: datetime | None = None,
) -> tuple[dict[str, Any], list[GovernanceCheck]]:
    """Validate current production, model, monitor, ledger, and approval integrity."""

    summary, status_checks = collect_governance_status(
        settings=settings, project_root=project_root, sources=sources
    )
    checks: list[GovernanceCheck] = []
    checks.extend(_validate_pipeline(summary, project_root, sources))
    checks.extend(_validate_registry(settings.paths.models, sources))
    checks.extend(_validate_monitor(settings.paths.reports, summary, sources))
    checks.extend(_validate_paper(settings.paths.paper_trading, settings, sources))
    checks.extend(
        _validate_pending_approvals(
            settings.paths.models,
            sources,
            now=(now or datetime.now(UTC)).astimezone(UTC),
        )
    )
    # Status-only warnings provide useful context but must not duplicate hard checks.
    checks.extend(item for item in status_checks if item.name.startswith("research_agent."))
    return summary, checks


def _validate_pipeline(
    summary: dict[str, Any], project_root: Path, sources: SourceCatalog
) -> list[GovernanceCheck]:
    checks: list[GovernanceCheck] = []
    production = summary["production"]
    summary_path = production.get("summary_path")
    if not isinstance(summary_path, str):
        return [
            GovernanceCheck(
                name="pipeline.manifest",
                status="FAIL",
                message="latest production summary is missing",
            )
        ]
    payload = sources.json(Path(summary_path))
    if payload.get("artifact_name") != "production_daily_summary":
        checks.append(
            GovernanceCheck(
                name="pipeline.summary_schema",
                status="FAIL",
                message="unexpected production summary identity",
                source_path=summary_path,
            )
        )
    run_path = production.get("run_manifest_path")
    if not isinstance(run_path, str):
        checks.append(
            GovernanceCheck(
                name="pipeline.run_manifest",
                status="FAIL",
                message="production run manifest is missing",
            )
        )
    else:
        run = sources.json(Path(run_path))
        valid = run.get("status") == "success" and run.get("run_id") == payload.get("run_id")
        checks.append(
            GovernanceCheck(
                name="pipeline.run_manifest",
                status="PASS" if valid else "FAIL",
                message="production run manifest is successful and linked"
                if valid
                else "production run manifest is not successful or linked",
                source_path=run_path,
            )
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        checks.append(
            GovernanceCheck(
                name="pipeline.artifacts",
                status="FAIL",
                message="production summary has no artifact list",
            )
        )
    else:
        missing: list[str] = []
        for value in artifacts:
            if not isinstance(value, str):
                missing.append(str(value))
                continue
            path = resolve_artifact_path(value, project_root)
            if path.is_file():
                sources.track(path)
            else:
                missing.append(str(path))
        checks.append(
            GovernanceCheck(
                name="pipeline.artifacts",
                status="FAIL" if missing else "PASS",
                message=f"missing production artifacts={missing}"
                if missing
                else f"validated {len(artifacts)} production artifacts",
                details={"missing": missing},
            )
        )
    checks.append(_validate_production_lineage(Path(summary_path), payload, sources))
    return checks


def _validate_production_lineage(
    summary_path: Path,
    summary: dict[str, Any],
    sources: SourceCatalog,
) -> GovernanceCheck:
    root = summary_path.parent
    prediction_manifest_path = root / "manifest.json"
    candidate_manifest_path = root / "candidates_manifest.json"
    predictions_path = root / "predictions.parquet"
    candidates_path = root / "candidates.csv"
    required = (
        prediction_manifest_path,
        candidate_manifest_path,
        predictions_path,
        candidates_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return GovernanceCheck(
            name="pipeline.artifact_lineage",
            status="FAIL",
            message=f"required production lineage artifacts are missing={missing}",
        )
    try:
        prediction_manifest = sources.json(prediction_manifest_path)
        candidate_manifest = sources.json(candidate_manifest_path)
        sources.track(predictions_path)
        sources.track(candidates_path)
        predictions = pd.read_parquet(
            predictions_path,
            columns=["trade_date", "ts_code", "model_id"],
        )
        candidates = pd.read_csv(
            candidates_path,
            usecols=["trade_date", "ts_code", "model_id"],
            dtype={"trade_date": str, "ts_code": str, "model_id": str},
        )
        as_of = str(summary.get("as_of") or "")
        model_id = str(summary.get("model_id") or "")
        embedded = candidate_manifest.get("prediction_manifest")
        valid = (
            prediction_manifest.get("artifact_name") == "production_predictions"
            and candidate_manifest.get("artifact_name") == "production_candidates"
            and str(prediction_manifest.get("as_of")) == as_of
            and str(candidate_manifest.get("as_of")) == as_of
            and prediction_manifest.get("model_id") == model_id
            and candidate_manifest.get("model_id") == model_id
            and candidate_manifest.get("feature_hash") == prediction_manifest.get("feature_hash")
            and isinstance(embedded, dict)
            and _payload_hash(embedded) == _payload_hash(prediction_manifest)
            and int(prediction_manifest.get("prediction_count", -1)) == len(predictions)
            and int(candidate_manifest.get("candidate_count", -1)) == len(candidates)
            and int(summary.get("candidate_count", -1)) == len(candidates)
            and set(predictions["trade_date"].astype(str)) == {as_of}
            and set(candidates["trade_date"].astype(str)) in ({as_of}, set())
            and set(predictions["model_id"].astype(str)) == {model_id}
            and set(candidates["model_id"].astype(str)) in ({model_id}, set())
            and not predictions.duplicated(["trade_date", "ts_code"]).any()
            and not candidates.duplicated(["trade_date", "ts_code"]).any()
        )
    except Exception as error:
        return GovernanceCheck(
            name="pipeline.artifact_lineage",
            status="FAIL",
            message=f"production lineage validation failed: {error}",
        )
    return GovernanceCheck(
        name="pipeline.artifact_lineage",
        status="PASS" if valid else "FAIL",
        message=(
            "prediction/candidate lineage is valid"
            if valid
            else "prediction/candidate identity, hash binding, or row counts differ"
        ),
    )


def _validate_registry(models_root: Path, sources: SourceCatalog) -> list[GovernanceCheck]:
    registry = models_root / "registry.json"
    if not registry.is_file():
        return [
            GovernanceCheck(
                name="model.registry",
                status="FAIL",
                message="registry.json is missing",
                source_path=str(registry),
            )
        ]
    try:
        records = load_registry_records(registry)
        sources.track(registry)
    except Exception as error:
        return [
            GovernanceCheck(
                name="model.registry",
                status="FAIL",
                message=f"registry is invalid: {error}",
                source_path=str(registry),
            )
        ]
    champions = [item for item in records if item.status == "champion"]
    checks = [
        GovernanceCheck(
            name="model.champion",
            status="PASS" if len(champions) == 1 else "FAIL",
            message=f"Champion count={len(champions)}",
            source_path=str(registry),
        )
    ]
    if len(champions) != 1:
        return checks
    champion = champions[0]
    artifact = Path(champion.artifact_path)
    required = ("model.txt", "feature_list.json", "manifest.json", "metrics.json")
    missing = [name for name in required if not (artifact / name).is_file()]
    if missing:
        checks.append(
            GovernanceCheck(
                name="model.artifact",
                status="FAIL",
                message=f"Champion artifact files missing={missing}",
                source_path=str(artifact),
            )
        )
        return checks
    for name in required:
        sources.track(artifact / name)
    try:
        features = _load_json(artifact / "feature_list.json")
        names = features.get("features")
        computed = (
            feature_list_hash(tuple(str(item) for item in names)) if isinstance(names, list) else ""
        )
        valid_hash = bool(computed) and computed == champion.feature_hash
    except Exception:
        valid_hash = False
    checks.append(
        GovernanceCheck(
            name="model.artifact",
            status="PASS" if valid_hash else "FAIL",
            message="Champion artifact and feature hash are valid"
            if valid_hash
            else "Champion feature hash does not match artifact",
            source_path=str(artifact),
        )
    )
    history = (
        sorted((models_root / "champion_history").glob("*.json"))
        if (models_root / "champion_history").exists()
        else []
    )
    if not history:
        checks.append(
            GovernanceCheck(
                name="model.assignment",
                status="WARNING",
                message="legacy Champion has no immutable assignment history",
            )
        )
    else:
        assignments = [sources.json(path) for path in history]
        latest = max(assignments, key=lambda item: str(item.get("activated_at") or ""))
        checks.append(
            GovernanceCheck(
                name="model.assignment",
                status="PASS" if latest.get("model_id") == champion.model_id else "FAIL",
                message="latest assignment matches Champion"
                if latest.get("model_id") == champion.model_id
                else "latest assignment differs from Champion",
            )
        )
    return checks


def _validate_monitor(
    reports_root: Path, summary: dict[str, Any], sources: SourceCatalog
) -> list[GovernanceCheck]:
    monitor_date = summary["monitoring"].get("latest_monitor_date")
    if not monitor_date:
        return [
            GovernanceCheck(
                name="monitor.manifest", status="WARNING", message="monitor snapshot is unavailable"
            )
        ]
    root = reports_root / "model_monitor" / str(monitor_date)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return [
            GovernanceCheck(
                name="monitor.manifest",
                status="FAIL",
                message="monitor manifest is missing",
                source_path=str(manifest_path),
            )
        ]
    try:
        manifest = sources.json(manifest_path)
        hashes = manifest.get("monitor_metric_file_hashes")
        mapping = {
            "health": root / "health.json",
            "performance_metrics": root / "performance" / "performance_metrics.parquet",
            "performance_manifest": root / "performance" / "manifest.json",
            "portfolio_metrics": root / "portfolio_metrics.parquet",
        }
        invalid = [
            name
            for name, path in mapping.items()
            if not isinstance(hashes, dict) or hashes.get(name) != file_sha256(path)
        ]
        for path in mapping.values():
            sources.track(path)
    except Exception as error:
        return [
            GovernanceCheck(
                name="monitor.manifest",
                status="FAIL",
                message=f"monitor manifest validation failed: {error}",
                source_path=str(manifest_path),
            )
        ]
    return [
        GovernanceCheck(
            name="monitor.manifest",
            status="FAIL" if invalid else "PASS",
            message=f"monitor hash mismatches={invalid}"
            if invalid
            else "monitor manifest and metric hashes are valid",
            source_path=str(manifest_path),
        )
    ]


def _validate_paper(
    paper_root: Path, settings: AppSettings, sources: SourceCatalog
) -> list[GovernanceCheck]:
    checks: list[GovernanceCheck] = []
    for portfolio in settings.paper_trading.portfolios:
        root = paper_root / portfolio.portfolio_id
        for filename, key in _PAPER_KEYS.items():
            path = root / filename
            if not path.is_file():
                checks.append(
                    GovernanceCheck(
                        name=f"paper.{portfolio.portfolio_id}.{filename}",
                        status="WARNING",
                        message="ledger is not initialized",
                        source_path=str(path),
                    )
                )
                continue
            try:
                frame = pd.read_parquet(path, columns=[key])
                sources.track(path)
                duplicate = frame[key].duplicated().any()
            except Exception as error:
                checks.append(
                    GovernanceCheck(
                        name=f"paper.{portfolio.portfolio_id}.{filename}",
                        status="FAIL",
                        message=f"ledger cannot be validated: {error}",
                        source_path=str(path),
                    )
                )
                continue
            checks.append(
                GovernanceCheck(
                    name=f"paper.{portfolio.portfolio_id}.{filename}",
                    status="FAIL" if duplicate else "PASS",
                    message=f"duplicate {key} values" if duplicate else f"ledger rows={len(frame)}",
                    source_path=str(path),
                )
            )
    return checks


def _validate_pending_approvals(
    models_root: Path, sources: SourceCatalog, now: datetime
) -> list[GovernanceCheck]:
    root = models_root / "promotion_requests"
    invalid: list[str] = []
    expired: list[str] = []
    if root.exists():
        registry = models_root / "registry.json"
        registry_hash = file_sha256(registry) if registry.is_file() else ""
        for request_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if (
                list((request_dir / "apply").glob("*/manifest.json"))
                if (request_dir / "apply").exists()
                else []
            ):
                continue
            events = (
                [
                    path
                    for path in sorted((request_dir / "approval_events").glob("*.json"))
                    if not path.name.endswith(".manifest.json")
                ]
                if (request_dir / "approval_events").exists()
                else []
            )
            if not events:
                continue
            try:
                event = ApprovalEvent.model_validate(sources.json(events[0]))
                manifest_path = events[0].with_name(f"{events[0].stem}.manifest.json")
                manifest = sources.json(manifest_path)
                if (
                    manifest.get("event_file_sha256") != file_sha256(events[0])
                    or event.registry_hash_at_review != registry_hash
                ):
                    invalid.append(request_dir.name)
                elif event.event_type == "APPROVED" and _parse_time(event.expires_at) < now:
                    expired.append(request_dir.name)
            except Exception:
                invalid.append(request_dir.name)
    return [
        GovernanceCheck(
            name="promotion.pending_approval_integrity",
            status="FAIL" if invalid else "PASS",
            message=f"invalid pending approvals={invalid}"
            if invalid
            else "pending approvals are hash-bound",
            details={"request_ids": invalid},
        ),
        GovernanceCheck(
            name="promotion.pending_approval_expiry",
            status="FAIL" if expired else "PASS",
            message=f"expired approvals waiting apply={expired}"
            if expired
            else "no expired approval is waiting apply",
            details={"request_ids": expired},
        ),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
