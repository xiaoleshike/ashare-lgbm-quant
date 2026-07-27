#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/install_production_timer.sh [options]

Options:
  --project-dir PATH  Repository root (default: current directory)
  --user USER         Non-root service user (default: SUDO_USER or current user)
  --venv PATH         Python virtualenv path (default: PROJECT_DIR/.venv)
  --env-file PATH     Existing systemd EnvironmentFile
  --dry-run           Render and print units without changing the system
  --help              Show this help
EOF
}

PROJECT_DIR="$(pwd -P)"
RUN_USER="${SUDO_USER:-$(id -un)}"
VENV_PATH=""
ENV_FILE=""
DRY_RUN=false

while (($#)); do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --user) RUN_USER="$2"; shift 2 ;;
    --venv) VENV_PATH="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PROJECT_DIR="$(realpath "$PROJECT_DIR")"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/deploy/systemd/ashare-quant.env}"
VENV_PATH="$(realpath -m "$VENV_PATH")"
ENV_FILE="$(realpath -m "$ENV_FILE")"

[[ -d "$PROJECT_DIR" ]] || { echo "Project directory not found: $PROJECT_DIR" >&2; exit 2; }
[[ -f "$PROJECT_DIR/config/default.yaml" ]] || {
  echo "Missing config/default.yaml under $PROJECT_DIR" >&2
  exit 2
}
[[ -x "$VENV_PATH/bin/ashare-quant" ]] || {
  echo "ashare-quant is not executable: $VENV_PATH/bin/ashare-quant" >&2
  exit 2
}
[[ -f "$ENV_FILE" ]] || {
  echo "Environment file not found: $ENV_FILE" >&2
  echo "Create it from deploy/systemd/ashare-quant.env.example and set TUSHARE_TOKEN." >&2
  exit 2
}
[[ "$RUN_USER" != "root" ]] || {
  echo "Refusing to install the quant service with User=root" >&2
  exit 2
}
id "$RUN_USER" >/dev/null 2>&1 || { echo "Unknown Linux user: $RUN_USER" >&2; exit 2; }

for directory in reports runs data; do
  path="$PROJECT_DIR/$directory"
  [[ -d "$path" ]] || { echo "Required writable directory does not exist: $path" >&2; exit 2; }
  if [[ "$(id -un)" == "$RUN_USER" ]]; then
    [[ -w "$path" ]] || { echo "Directory is not writable by $RUN_USER: $path" >&2; exit 2; }
  elif command -v runuser >/dev/null 2>&1; then
    runuser -u "$RUN_USER" -- test -w "$path" || {
      echo "Directory is not writable by $RUN_USER: $path" >&2
      exit 2
    }
  else
    echo "Cannot verify write access for $RUN_USER because runuser is unavailable" >&2
    exit 2
  fi
done

render_unit() {
  local source="$1"
  local target="$2"
  sed \
    -e "s|@PROJECT_DIR@|$PROJECT_DIR|g" \
    -e "s|@RUN_USER@|$RUN_USER|g" \
    -e "s|@VENV_BIN@|$VENV_PATH/bin|g" \
    -e "s|@ENV_FILE@|$ENV_FILE|g" \
    "$source" >"$target"
}

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
for unit in \
  ashare-quant-production.service \
  ashare-quant-production.timer \
  ashare-quant-full-update.service \
  ashare-quant-full-update.timer; do
  render_unit "$PROJECT_DIR/deploy/systemd/$unit" "$TEMP_DIR/$unit"
done

if [[ "$DRY_RUN" == true ]]; then
  echo "Dry run: no files or systemd state will be modified."
  for unit in "$TEMP_DIR"/*; do
    echo "===== $(basename "$unit") ====="
    cat "$unit"
  done
  exit 0
fi

[[ "$EUID" -eq 0 ]] || {
  echo "Installation requires root; rerun with sudo or use --dry-run." >&2
  exit 2
}

install -m 0644 "$TEMP_DIR"/* /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ashare-quant-production.timer ashare-quant-full-update.timer
systemctl --no-pager status ashare-quant-production.timer ashare-quant-full-update.timer || true
systemctl list-timers --all \
  ashare-quant-production.timer ashare-quant-full-update.timer --no-pager

cat <<EOF
Installed production timers.
Logs:
  journalctl -u ashare-quant-production.service -n 100 --no-pager
  journalctl -u ashare-quant-full-update.service -n 100 --no-pager
Manual triggers:
  systemctl start ashare-quant-production.service
  systemctl start ashare-quant-full-update.service
EOF
