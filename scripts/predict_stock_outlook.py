#!/usr/bin/env python3
"""Print a registered 10-day Ranker's relative outlook for one A-share."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ashare_quant.config import load_settings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.models.stock_outlook import StockOutlookPredictor


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone stock-outlook CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model-id", required=True, help="Registered horizon-10 model ID.")
    parser.add_argument(
        "--ts-code", required=True, help="Tushare stock code, for example 000001.SZ."
    )
    parser.add_argument("--as-of", required=True, help="Feature date in YYYYMMDD.")
    parser.add_argument(
        "--horizon", type=int, default=10, help="Required model horizon; default 10."
    )
    parser.add_argument("--storage-root", default=None, help="Override raw Parquet root.")
    parser.add_argument("--processed-root", default=None, help="Override processed Parquet root.")
    parser.add_argument("--models-root", default=None, help="Override model registry root.")
    parser.add_argument("--output", default=None, help="Optional JSON output file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one read-only relative outlook prediction."""

    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    predictor = StockOutlookPredictor(
        raw_root=settings.paths.parquet_store
        if args.storage_root is None
        else Path(args.storage_root),
        processed_root=(
            settings.paths.processed_data
            if args.processed_root is None
            else Path(args.processed_root)
        ),
        models_root=settings.paths.models if args.models_root is None else Path(args.models_root),
    )
    try:
        result = predictor.predict(
            model_id=args.model_id,
            ts_code=args.ts_code,
            as_of=args.as_of,
            horizon=args.horizon,
        )
    except (DataValidationError, OSError, ValueError) as error:
        print(f"stock outlook failed: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"stock_outlook: output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
