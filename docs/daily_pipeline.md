# Daily Production Pipeline

Phase 1.3 provides the first locked production entry point:

```bash
ashare-quant --config config/default.yaml pipeline daily
```

The default as-of date is the latest open date in local `trade_cal` whose session has completed.
The cutoff uses Asia/Shanghai time and treats the current open date as completed at 15:00. To run
an explicit completed trading date:

```bash
ashare-quant --config config/default.yaml pipeline daily --as-of 20260717
```

The command acquires `runs/.production.lock`, creates a separate run manifest, and executes these
hard gates in order. In default mode, data update runs first and refreshes `trade_cal`; the as-of
date is then resolved from the updated calendar. With explicit `--as-of`, data update is bounded to
that date.

1. `data update --repair-gaps` (plus `--end-date <as-of>` when explicitly requested)
2. `data validate`
3. `raw_freshness_gate`
4. `universe build --start-date <as-of> --end-date <as-of>`
5. `universe validate --start-date <as-of> --end-date <as-of>`
6. `universe_readiness_gate`
7. `features build --start-date <as-of> --end-date <as-of>`
8. `features validate --start-date <as-of> --end-date <as-of>`
9. `features_readiness_gate`

Each stage reuses the existing CLI handler in the same process. Any non-zero stage exit stops the
pipeline immediately and marks the run failed. Success is recorded only after all nine stages pass.
Stage records include command arguments, exit code, elapsed time, and generated artifact manifests
when available. Pre-run universe, label, and feature manifests are retained as source provenance.
For incremental universe and feature builds, `build_scope` records the requested dates, changed
rows, and changed partitions. `canonical_artifact` separately records the complete resulting
Parquet artifact row count, partition count, and date range, so a one-day run does not replace
full-history provenance with one-day statistics.

`index_daily` gap detection supports per-code inception boundaries through
`data.index_first_available_dates`. Dates before a configured boundary are reported as
`excluded_before_inception` and are never requested for repair. A code without a configured
boundary is handled conservatively: every open date in the requested range remains expected.

Run records are stored under `runs/YYYYMMDD/<run_id>/manifest.json`. Repeating the same as-of date
is safe because raw and processed stores retain their existing idempotent merge behavior, while
each attempt receives a distinct run ID. This phase does not generate labels, predictions,
recommendations, trades, or models.

## Readiness Inspection

Run all gates without writing raw or processed data:

```bash
ashare-quant --config config/default.yaml pipeline readiness --as-of 20260717
```

The command returns zero only for `READY`. Hard raw datasets must contain the requested completed
session, benchmark entities must be present, and an empty conditional dataset such as `suspend_d`
requires an explicit empty-result marker. Low-frequency financial and event datasets use lag or
existence warnings rather than same-day requirements.

Universe and feature gates compare the requested cross-section with up to 20 prior completed
sessions. Severe count deviations fail; moderate deviations warn. Feature hard requirements are
empty by default until `production.freshness.required_feature_list_path` or
`hard_required_features` is explicitly configured. Structurally sparse financial features are
reported but do not fail readiness merely because they are mostly null.

Every gate stores structured failures, warnings, row counts, baselines, missingness, and thresholds
in the run manifest. A successful artifact build also moves its new manifest into
`source_provenance.resulting_manifests` and the current `upstream_manifests`; pre-run identities
remain under `input_manifests`.
