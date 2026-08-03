# Recovery Manual

Governance recovery commands only validate; they never repair state:

```bash
ashare-quant --config config/default.yaml governance validate-recovery
```

## Registry Restore

1. Stop production jobs and verify no process owns `runs/.production.lock`.
2. Copy the current `models/registry.json` to incident evidence storage without modifying it.
3. Read `reports/governance/recovery.json` and select the latest valid entry from
   `models/registry_versions/` that has intact model artifacts and matching Champion history.
4. Validate the selected JSON schema, model artifact files, feature hashes, and one Champion per
   model type.
5. Restore `registry.json` using an atomic same-filesystem replacement under the documented
   operator change procedure. Never edit its JSON manually.
6. Rerun `governance validate-recovery` and `governance validate-production`.
7. Record the incident, source registry version, operator, hashes, and time.

The governance command deliberately does not provide an automatic restore switch. Restoration is an
exceptional operator action and must preserve both old and restored bytes.

## Interrupted Apply or Rollback

An `apply_pending.json` or `rollback_apply_pending.json` without its completion manifest indicates an
interrupted transaction. Do not rerun arbitrary registry commands. Compare the current registry hash
with the journal's parent and target versions, retain all files, and use the existing idempotent apply
or rollback command only after the request, approval, and artifact hashes remain valid.

## Production Recovery

1. Inspect the failed `runs/YYYYMMDD/RUN_ID/manifest.json` and failed stage.
2. Confirm no success `production_summary.json` was published for that failed run.
3. Validate source data and processed artifact manifests.
4. Resolve the original failure; do not relabel the failed manifest.
5. Start a new production run for the session. It receives a new run ID.
6. Confirm summary/run ID linkage and run `governance validate-production`.

## Artifact Restore Verification

Verify model files, source manifest hashes, registry feature hashes, observation identities, and
Paper Trading primary keys. A directory without its final manifest is incomplete and cannot be
promoted to valid state merely by adding a manifest. Historical observations, approval events,
Champion assignments, and registry versions are append-only.
