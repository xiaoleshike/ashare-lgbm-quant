"""Command-line entry points for pipeline operations."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from ashare_quant.config import load_settings
from ashare_quant.utils import configure_logging

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(prog="ashare-quant")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Validate local configuration and runtime prerequisites.")
    subparsers.add_parser("config-check", help="Load and validate configuration.")
    return parser


def run_doctor(config_path: str | None) -> int:
    """Run lightweight environment checks without contacting external services."""

    settings = load_settings(config_path)
    configure_logging(settings.logging.level, settings.logging.json_logs)
    LOGGER.info("configuration loaded", extra={"project": settings.project_name})
    if not settings.has_tushare_token:
        LOGGER.warning("TUSHARE_TOKEN is not set; data ingestion commands will fail later")
    return 0


def run_config_check(config_path: str | None) -> int:
    """Validate configuration and print a non-secret summary."""

    settings = load_settings(config_path)
    print(f"project={settings.project_name}")
    print(f"environment={settings.environment}")
    print(f"tushare_token_set={settings.has_tushare_token}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return run_doctor(args.config)
    if args.command == "config-check":
        return run_config_check(args.config)
    raise ValueError(f"Unsupported command: {args.command}")
