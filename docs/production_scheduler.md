# Production Scheduler

## Prerequisites

The scheduler uses systemd, the repository virtual environment, the existing
production lock, and the local `trade_cal`. The service user must not be root and
must be able to write `data/`, `reports/`, and `runs/`.

Create the private environment file without committing it:

```bash
cp deploy/systemd/ashare-quant.env.example deploy/systemd/ashare-quant.env
chmod 600 deploy/systemd/ashare-quant.env
vim deploy/systemd/ashare-quant.env
```

Set `TUSHARE_TOKEN` in that file. Never place the token in a unit or command line.

## Install

Review rendered units without modifying the host:

```bash
scripts/install_production_timer.sh \
  --project-dir "$PWD" \
  --user "$USER" \
  --venv "$PWD/.venv" \
  --env-file "$PWD/deploy/systemd/ashare-quant.env" \
  --dry-run
```

Install both timers:

```bash
sudo scripts/install_production_timer.sh \
  --project-dir "$PWD" \
  --user "$USER" \
  --venv "$PWD/.venv" \
  --env-file "$PWD/deploy/systemd/ashare-quant.env"
```

The production timer runs Monday through Friday at 19:30 Asia/Shanghai. The CLI
uses `trade_cal`, requires the configured 18:30 data-ready threshold, skips
non-trading days, and does not repeat an already valid publication. The full-data
timer runs Wednesday and Sunday at 12:00 and resolves its end date from
`trade_cal`; it performs all-dataset update, snapshot refresh, and gap repair.

## Operations

Manual production execution:

```bash
ashare-quant --config config/default.yaml pipeline production
ashare-quant --config config/default.yaml pipeline production --as-of 20260724
```

Manual full-data maintenance:

```bash
ashare-quant --config config/default.yaml pipeline full-update
```

Inspect timer and report state:

```bash
scripts/production_status.sh "$PWD"
systemctl list-timers --all ashare-quant-production.timer ashare-quant-full-update.timer
journalctl -u ashare-quant-production.service -n 100 --no-pager
```

A successful report requires `reports/YYYYMMDD/production_summary.json`, a
matching successful run manifest, and every referenced artifact. A zero process
exit without these checks is treated as failure. A non-trading day, pre-readiness
trigger, or already completed date is a legal skip with exit code zero.

Tushare requests already use bounded exponential retry. Scheduler-level retries
therefore default to one pipeline attempt to avoid nested retry storms. If request
retry is deliberately reduced, `production.scheduler.max_pipeline_attempts` can
be raised to at most three; only timeout, connection, rate-limit, and temporary
service failures are retried. Readiness, schema, provenance, and model failures
are never retried.

For recovery, inspect the failed stage and invocation under `runs/`, correct the
underlying data/configuration problem, then trigger the service manually:

```bash
sudo systemctl start ashare-quant-production.service
```

Change schedules by editing the timer templates, reinstalling, and checking
`systemctl list-timers`. Disable or re-enable without deleting files:

```bash
sudo systemctl disable --now ashare-quant-production.timer ashare-quant-full-update.timer
sudo systemctl enable --now ashare-quant-production.timer ashare-quant-full-update.timer
```

Uninstall units while preserving the private environment file and all artifacts:

```bash
sudo scripts/uninstall_production_timer.sh
```
