"""Thin explicit operator commands; acquisition is never part of validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashare_quant.config.settings import AppSettings
from ashare_quant.data.economic_event_evidence import (
    compile_economic_evidence,
    publish_terminal_evidence,
    validate_economic_event_evidence_artifact,
)
from ashare_quant.data.economic_event_sources import (
    load_exposure_source,
    probe_dividends,
    query_targets,
)
from ashare_quant.data.tushare_client import TushareClient, TushareClientConfig


def add_economic_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name in (
        "economic-evidence-preflight",
        "economic-dividend-probe",
        "economic-evidence-compile",
    ):
        command = commands.add_parser(
            name, help="Isolated economic evidence; never execute a portfolio."
        )
        command.add_argument("--continuous-run", type=Path, required=True)
        command.add_argument("--continuous-manifest-sha256", required=True)
        command.add_argument("--audit-root", type=Path, required=True)
        command.add_argument("--audit-manifest-sha256", required=True)
        if name != "economic-evidence-preflight":
            command.add_argument("--output-root", type=Path, required=True)
        if name == "economic-evidence-compile":
            command.add_argument("--provider-source", type=Path, required=True)
            command.add_argument(
                "--dividend-review",
                type=Path,
                default=None,
                help="Explicit reviewed raw-row hashes; omission never approves events.",
            )
            command.add_argument("--terminal-package", type=Path, action="append", default=[])
    validate = commands.add_parser(
        "economic-evidence-validate", help="Read-only recursive evidence validation."
    )
    validate.add_argument("--artifact", type=Path, required=True)
    terminal = commands.add_parser(
        "terminal-economic-evidence-publish", help="Freeze reviewed terms and local document bytes."
    )
    terminal.add_argument("--review", type=Path, required=True)
    terminal.add_argument("--documents-root", type=Path, required=True)
    terminal.add_argument("--output-root", type=Path, required=True)


def run_economic_command(args: argparse.Namespace, settings: AppSettings) -> int:
    if args.data_command == "economic-evidence-validate":
        manifest = validate_economic_event_evidence_artifact(args.artifact)
        print(json.dumps({"artifact_id": manifest["artifact_id"], "validation": "PASS"}))
        return 0
    if args.data_command == "terminal-economic-evidence-publish":
        path = publish_terminal_evidence(args.review, args.documents_root, args.output_root)
    else:
        source = load_exposure_source(
            args.continuous_run,
            args.audit_root,
            continuous_hash=args.continuous_manifest_sha256,
            audit_hash=args.audit_manifest_sha256,
        )
        if args.data_command == "economic-evidence-preflight":
            print(
                json.dumps(
                    {
                        "source": source.identity,
                        "exposures": len(source.exposures),
                        "unique_boundaries": source.exposures.economic_boundary_id.nunique(),
                        "unique_securities": source.exposures.ts_code.nunique(),
                        "provider_query_codes": len(query_targets(source)),
                        "unique_adjustment_boundaries": source.exposures[
                            source.exposures.exposure_type.eq("ADJ_FACTOR_CHANGE")
                        ].economic_boundary_id.nunique(),
                        "unique_terminal_boundaries": source.exposures[
                            source.exposures.exposure_type.eq("TERMINAL_EVENT")
                        ].economic_boundary_id.nunique(),
                        "by_top_n": {
                            str(n): {
                                "positions": g.position_id.nunique(),
                                "adjustment_positions": g[
                                    g.exposure_type.eq("ADJ_FACTOR_CHANGE")
                                ].position_id.nunique(),
                                "adjustment_change_rows": int(
                                    g.exposure_type.eq("ADJ_FACTOR_CHANGE").sum()
                                ),
                                "terminal_rows": int(g.exposure_type.eq("TERMINAL_EVENT").sum()),
                            }
                            for n, g in source.exposures.groupby("top_n")
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.data_command == "economic-dividend-probe":
            client = TushareClient(
                token=settings.tushare_token,
                config=TushareClientConfig(
                    retry_attempts=settings.data.retry_attempts,
                    rate_limit_per_minute=settings.data.rate_limit_per_minute,
                    request_interval_seconds=settings.data.request_interval_seconds,
                    backoff_base_seconds=settings.data.backoff_base_seconds,
                    backoff_max_seconds=settings.data.backoff_max_seconds,
                    endpoint_rate_limits_per_minute=dict(
                        settings.data.endpoint_rate_limits_per_minute
                    ),
                ),
            )
            path = probe_dividends(source, client, args.output_root)
        else:
            path = compile_economic_evidence(
                source,
                args.provider_source,
                args.output_root,
                dividend_review=args.dividend_review,
                terminal_packages=tuple(args.terminal_package),
            )
    print(json.dumps({"artifact": str(path.resolve())}))
    return 0
