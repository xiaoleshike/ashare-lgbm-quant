# Model Monitoring Alert Engine

The alert engine is a read-only layer over published monitoring metrics. It
does not read labels, features, prices, inference artifacts, backtests, or
model-training artifacts directly.

## Inputs

For an `as_of` date, the standalone alert command reads only:

- `reports/model_monitor/YYYYMMDD/health.json`
- `reports/model_monitor/YYYYMMDD/performance/`
- `reports/model_monitor/YYYYMMDD/portfolio_metrics.parquet`
- `reports/model_monitor/YYYYMMDD/manifest.json`
- prior `reports/model_monitor/history/alert_history.parquet`

Input hashes are checked against the monitoring manifest before evaluation.
The normal `monitor run` path evaluates the same rules from its in-memory
health, performance, and portfolio results before publishing one atomic
monitoring snapshot.

## Rules And Lifecycle

Thresholds live under `monitoring.alerts` in the application configuration.
Rules cover model alpha decay, Rank IC decline, score collapse, existing drift
metrics, universe coverage, portfolio drawdown, concentration, and execution
quality.

An alert has a deterministic identity based on alert type, model, portfolio,
and metric. Its lifecycle is append-only:

- `NEW`: the condition first appears.
- `ACTIVE`: the condition remains present on a later date.
- `RECOVERED`: an evaluated condition returns to its healthy range.

Unavailable optional metrics generate warnings and do not create false
recovery events. Industry concentration and some drift metrics remain optional
until their upstream monitoring artifacts provide them.

## Commands

```bash
ashare-quant --config config/default.yaml monitor alerts --as-of YYYYMMDD
ashare-quant --config config/default.yaml monitor alerts-validate --as-of YYYYMMDD
ashare-quant --config config/default.yaml monitor alerts-status --as-of YYYYMMDD
```

`monitor run` also evaluates alerts as part of the complete monitoring
snapshot. Alert output is written under:

```text
reports/model_monitor/YYYYMMDD/alerts/
  alerts.json
  alert_report.md
  manifest.json
```

Lifecycle history is stored at:

```text
reports/model_monitor/history/alert_history.parquet
```

The manifest is written last. Publication failure preserves the previous
successful monitoring snapshot and alert history.
