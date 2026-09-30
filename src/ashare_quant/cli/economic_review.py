"""Thin operator entry points for frozen event review and optional metadata discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.economic_event_supplements import compile_supplements, validate_supplements
from ashare_quant.data.economic_review_artifacts import (
    compile_review,
    load_review_inputs,
    validate_reviewed_economic_evidence_artifact,
)
from ashare_quant.data.terminal_announcement_sources import probe_announcements
from ashare_quant.data.tushare_client import TushareClient, TushareClientConfig

COMMANDS = {
    "economic-review-triage",
    "economic-review-compile",
    "economic-review-validate",
    "terminal-announcement-probe",
    "economic-review-supplement",
    "economic-review-supplement-validate",
}


def add_review_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name in sorted(COMMANDS):
        p = commands.add_parser(
            name, help="Evidence review only; never approve automatically or execute a portfolio."
        )
        if name in {"economic-review-validate", "economic-review-supplement-validate"}:
            p.add_argument("--artifact", type=Path, required=True)
            continue
        if name == "economic-review-supplement":
            p.add_argument("--parent", type=Path, required=True)
            p.add_argument("--parent-manifest-sha256", required=True)
            p.add_argument("--supplements", type=Path, required=True)
            p.add_argument("--output-root", type=Path, required=True)
            continue
        p.add_argument("--initial-evidence", type=Path, required=True)
        p.add_argument("--initial-manifest-sha256", required=True)
        p.add_argument("--provider-source", type=Path, required=True)
        p.add_argument("--provider-manifest-sha256", required=True)
        p.add_argument("--output-root", type=Path, required=True)
        if name == "economic-review-compile":
            p.add_argument("--decisions", type=Path, required=True)
            p.add_argument("--terminal-package", type=Path, action="append", default=[])
            p.add_argument("--announcement-source", type=Path, default=None)


def run_review_command(args: argparse.Namespace, settings: AppSettings) -> int:
    if args.data_command == "economic-review-supplement-validate":
        manifest = validate_supplements(args.artifact)
        print(json.dumps({"artifact_id": manifest["artifact_id"], "validation": "PASS"}))
        return 0
    if args.data_command == "economic-review-supplement":
        path = compile_supplements(
            args.parent, args.parent_manifest_sha256, args.supplements, args.output_root
        )
        print(json.dumps({"artifact": str(path.resolve())}))
        return 0
    if args.data_command == "economic-review-validate":
        manifest = validate_reviewed_economic_evidence_artifact(args.artifact)
        print(json.dumps({"artifact_id": manifest["artifact_id"], "validation": "PASS"}))
        return 0
    inputs = load_review_inputs(
        args.initial_evidence,
        args.provider_source,
        args.initial_manifest_sha256,
        args.provider_manifest_sha256,
    )
    if args.data_command == "terminal-announcement-probe":
        client = TushareClient(
            token=settings.tushare_token,
            config=TushareClientConfig(
                retry_attempts=settings.data.retry_attempts,
                rate_limit_per_minute=settings.data.rate_limit_per_minute,
                request_interval_seconds=settings.data.request_interval_seconds,
                backoff_base_seconds=settings.data.backoff_base_seconds,
                backoff_max_seconds=settings.data.backoff_max_seconds,
                endpoint_rate_limits_per_minute=dict(settings.data.endpoint_rate_limits_per_minute),
            ),
        )
        path = probe_announcements(inputs.source, client, args.output_root)
    else:
        path = compile_review(
            inputs,
            args.output_root,
            decisions=getattr(args, "decisions", None),
            terminal_packages=tuple(getattr(args, "terminal_package", [])),
            announcement_source=getattr(args, "announcement_source", None),
        )
    print(json.dumps({"artifact": str(path.resolve())}))
    return 0
