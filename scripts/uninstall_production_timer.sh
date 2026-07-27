#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
elif (($#)); then
  echo "Usage: scripts/uninstall_production_timer.sh [--dry-run]" >&2
  exit 2
fi

UNITS=(
  ashare-quant-production.timer
  ashare-quant-production.service
  ashare-quant-full-update.timer
  ashare-quant-full-update.service
)

if [[ "$DRY_RUN" == true ]]; then
  printf 'Would disable and remove: %s\n' "${UNITS[*]}"
  exit 0
fi
[[ "$EUID" -eq 0 ]] || { echo "Uninstall requires root." >&2; exit 2; }

systemctl disable --now ashare-quant-production.timer ashare-quant-full-update.timer || true
for unit in "${UNITS[@]}"; do
  rm -f "/etc/systemd/system/$unit"
done
systemctl daemon-reload
systemctl reset-failed
echo "Production scheduler units removed. Environment files were not modified."
