#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(realpath "${1:-.}")"
cd "$PROJECT_DIR"

echo "== Timers =="
systemctl is-enabled ashare-quant-production.timer 2>/dev/null || true
systemctl is-active ashare-quant-production.timer 2>/dev/null || true
systemctl list-timers --all \
  ashare-quant-production.timer ashare-quant-full-update.timer --no-pager || true

echo "== Latest Service =="
systemctl --no-pager status ashare-quant-production.service 2>/dev/null || true

echo "== Production Runs =="
"$PROJECT_DIR/.venv/bin/python" - "$PROJECT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
runs = []
for path in sorted((root / "runs").glob("[0-9]*/**/manifest.json")):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if value.get("pipeline_type") == "production_daily":
        runs.append((path, value))

successful = [item for item in runs if item[1].get("status") == "success"]
failed = [item for item in runs if item[1].get("status") == "failed"]
latest_success = successful[-1] if successful else None
latest_failure = failed[-1] if failed else None
if latest_success:
    print(f"latest_success_date={latest_success[1].get('as_of')}")
    print(f"latest_success_run_id={latest_success[1].get('run_id')}")
    summary = root / "reports" / str(latest_success[1].get("as_of")) / "production_summary.json"
    print(f"production_summary={summary}")
else:
    print("latest_success_date=None")
    print("latest_success_run_id=None")
    print("production_summary=None")
print(
    "latest_failed_run_id="
    + (str(latest_failure[1].get("run_id")) if latest_failure else "None")
)
PY

echo "== Recent Logs =="
journalctl -u ashare-quant-production.service -n 50 --no-pager 2>/dev/null || true
