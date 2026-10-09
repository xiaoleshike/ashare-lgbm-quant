"""Explicit single-security acquisition and offline validation/extraction commands."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

from ashare_quant.backtest.continuous_source import file_hash, read_json
from ashare_quant.data.announcement_http import CaptureCache, Limits, SafeHTTPClient, check
from ashare_quant.data.announcement_sources import acquire, probe, validate_announcement_source
from ashare_quant.data.announcement_text_extraction import publish_text, validate_text_artifact
from ashare_quant.data.cninfo_announcement_provider import AnnouncementQuery

COMMANDS = {
    "announcement-cninfo-probe",
    "announcement-cninfo-acquire",
    "announcement-source-validate",
    "announcement-text-extract",
    "announcement-text-validate",
}


def add_announcement_parsers(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    for name in sorted(COMMANDS):
        parser = commands.add_parser(name, help="Discovery/extraction only; offline by default.")
        if name.endswith("-validate"):
            parser.add_argument("--artifact", type=Path, required=True)
            continue
        parser.add_argument("--output-root", type=Path, required=True)
        if name == "announcement-cninfo-probe":
            parser.add_argument("--ts-code", required=True)
            parser.add_argument("--start-date", required=True)
            parser.add_argument("--end-date", required=True)
            parser.add_argument("--keyword", default="")
            parser.add_argument("--category", default="")
            parser.add_argument("--max-pages", type=int, default=100)
        else:
            parser.add_argument("--source", type=Path, required=True)
            parser.add_argument("--source-manifest-sha256", required=True)
        if name.startswith("announcement-cninfo-"):
            parser.add_argument("--cache-root", type=Path, required=True)
            parser.add_argument(
                "--allow-network",
                action="store_true",
                help="Explicit operator authorization for this bounded acquisition only.",
            )
            parser.add_argument(
                "--retry-failed",
                action="store_true",
                help="Retry cached failures, retaining earlier captures; requires --allow-network.",
            )
        if name == "announcement-cninfo-acquire":
            parser.add_argument("--announcement-id", action="append", required=True)


def run_announcement_command(args: argparse.Namespace) -> int:
    if args.data_command.endswith("-validate"):
        validator = (
            validate_text_artifact
            if args.data_command == "announcement-text-validate"
            else validate_announcement_source
        )
        manifest = validator(args.artifact)
        print(json.dumps({"artifact_id": manifest["artifact_id"], "validation": "PASS"}))
        return 0
    if hasattr(args, "source"):
        check(
            file_hash(args.source / "manifest.json") == args.source_manifest_sha256,
            "ANNOUNCEMENT_PARENT_HASH",
        )
    if args.data_command == "announcement-text-extract":
        path = publish_text(args.source, args.output_root)
    else:
        check(not args.retry_failed or args.allow_network, "RETRY_REQUIRES_NETWORK_AUTHORIZATION")
        policy = (
            Limits.model_validate(read_json(args.source / "responses.json")["limits"])
            if hasattr(args, "source")
            else Limits()
        )
        args.cache_root.mkdir(parents=True, exist_ok=True)
        # Lock only the explicit acquisition cache; never a production lock or root.
        with (args.cache_root / ".acquisition.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            client = SafeHTTPClient(
                authorized=args.allow_network,
                policy=policy,
                cache=CaptureCache(args.cache_root, policy),
                retry_failed=args.retry_failed,
            )
            if args.data_command == "announcement-cninfo-probe":
                query = AnnouncementQuery(
                    ts_code=args.ts_code,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    keyword=args.keyword,
                    category=args.category,
                    max_pages=args.max_pages,
                )
                path = probe(query, client, args.output_root)
            else:
                path = acquire(args.source, args.announcement_id, client, args.output_root)
    result = {"artifact": str(path.resolve())}
    if args.data_command.startswith("announcement-cninfo-"):
        summary = read_json(path / "acquisition_summary.json")
        result.update(
            discovery_status=summary["discovery_status"],
            acquisition_status=summary["acquisition_status"],
        )
        print(json.dumps(result))
        complete = summary["discovery_status"] == "COMPLETE" and (
            args.data_command == "announcement-cninfo-probe"
            or summary["acquisition_status"] == "COMPLETE"
        )
        return 0 if complete else 2
    print(json.dumps(result))
    return 0
