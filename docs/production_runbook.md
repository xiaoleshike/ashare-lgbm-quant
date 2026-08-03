# Production Governance Runbook

## System Overview

```text
Data
  |
Features
  |
Production Pipeline
  |
Prediction
  |
Paper Trading
  |
Monitoring
  |
Research Agent
  |
Promotion Governance
```

All commands must run from the repository root with the production virtual environment and the
same configuration used to build current artifacts. Governance commands are read-only with respect
to models, registries, trading state, and production reports. They publish audit snapshots under
`reports/governance/`.

## Daily Operation

After the configured market-data-ready time:

```bash
ashare-quant --config config/default.yaml pipeline production
```

An explicit replay uses:

```bash
ashare-quant --config config/default.yaml pipeline production --as-of YYYYMMDD
```

Expected output is a successful run manifest under `runs/` and
`reports/YYYYMMDD/production_summary.json`. A legal non-trading-day skip is not a production
publication.

Run monitoring after the production artifacts exist:

```bash
ashare-quant --config config/default.yaml monitor run --as-of YYYYMMDD
```

Generate the read-only research report:

```bash
ashare-quant --config config/default.yaml research-agent generate --as-of YYYYMMDD
```

Check the whole system:

```bash
ashare-quant --config config/default.yaml governance status
ashare-quant --config config/default.yaml governance validate-production
```

`validate-production` returns nonzero on `FAIL`. `WARNING` means the current publication remains
usable but requires operator review, such as missing optional monitoring history or a legacy
Champion without assignment history.

## Backup and Retention

Back up critical paths daily after the production run and after every governance state transition:

```text
models/registry.json
models/registry_versions/
models/champion_history/
models/promotion_requests/
models/rollback_requests/
models/challengers/ and all registered model artifact directories
runs/
reports/*/production_summary.json
reports/model_monitor/
reports/performance_observation/
reports/governance/
paper_trading/
```

Keep at least one off-host copy and retain every registry version, Champion assignment, approval,
apply, and rollback event indefinitely. Raw caches and temporary staging directories are optional;
do not back up incomplete `.*` staging directories as valid artifacts.

Restore in this order:

1. Model artifacts.
2. `registry_versions/`, `champion_history/`, promotion and rollback requests.
3. A validated `registry.json` version.
4. Production run manifests and reports.
5. Performance observations, monitoring, and Paper Trading ledgers.
6. Run `governance validate-recovery`, then `governance validate-production`.

## Safety Rules

Never manually edit `registry.json`, overwrite a model artifact, modify a historical observation,
modify an approval/promotion event, delete Champion history, or use frozen evaluation artifacts as
prospective production evidence. Do not copy a partially published directory that lacks its final
manifest.

See [model_lifecycle.md](model_lifecycle.md), [incident_response.md](incident_response.md), and
[recovery_manual.md](recovery_manual.md) for governed transitions and recovery.
