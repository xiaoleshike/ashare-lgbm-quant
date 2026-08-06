# Controlled Operational Qualification

Phase 2.8.2G qualifies the governed retraining lifecycle against existing immutable project
artifacts. It is an operator-controlled rehearsal, not a production rollout, Promotion decision,
or trading workflow.

## Isolation Contract

Generated model, registration, validation, and Shadow artifacts record `qualification_run_id`,
`qualification_only=true`, phase `2.8.2G`, and explicit Promotion/trading prohibitions.
Qualification models are not written to `registry.json`; Promotion evidence preparation, the Gate,
and Apply reject them. Candidate selection, Paper Trading, and order execution are not called.

`QUALIFIED` means only that preflight, dry-run, readiness, training, validation, isolated Shadow,
observation integration, and protected-state checks completed. It does not mean Promotion Gate
PASS, human approval, Champion replacement, or real-trading readiness.

## Safe Defaults

```yaml
retraining:
  qualification:
    enabled: true
    allow_real_training: false
    allow_real_shadow: false
    require_manual_stage_advance: true
```

CLI intent cannot override a disabled stage. Resource thresholds are optional; when unset,
preflight warns instead of inventing a hard threshold. Qualification policy has a separate hash and
does not alter Training Request, trigger-policy, lifecycle-policy, or Promotion Policy identities.

## Artifacts

```text
reports/retraining/qualification/<qualification_run_id>/
  qualification_summary.json
  qualification_events.parquet
  checkpoint_results.json
  source_inventory.json
  invariant_results.json
  report.md
  manifest.json

reports/shadow_predictions/YYYYMMDD/qualification/<qualification_run_id>/<model_id>/
  predictions.parquet
  manifest.json
```

Publication is atomic and manifest-last. Snapshot updates require the previous events as an exact
prefix and cannot remove successful checkpoint evidence. The production Shadow bundle and normal
`retrained/` sidecars remain unchanged.

## Operator Runbook

### 1. Start with Real Stages Disabled

```bash
ashare-quant --config config/default.yaml retraining qualification-start \
  --request-id REQUEST_ID \
  --as-of YYYYMMDD
```

Expected state: `TRAINING_PENDING_APPROVAL`. Inspect the report, source inventory, protected
invariant baseline, dry-run, readiness manifest, and every warning.

### 2. Enable and Run Training

After explicit review, set `allow_real_training: true` and run:

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to training
```

Expected state: `VALIDATION_PENDING_APPROVAL`. Verify dataset and fold identities, feature/label
hashes, candidate registration, final-test exclusion, and qualification fields. Failed attempts and
retries consume the Asia/Shanghai daily training-attempt budget.

### 3. Run Validation

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to validation
```

Expected state: `SHADOW_PENDING_APPROVAL`. Inspect offline metrics, executable OOS evidence,
leakage contracts, unresolved-position rejection, and Shadow eligibility.

### 4. Enable and Run Isolated Shadow

After review, set `allow_real_shadow: true` and run:

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to shadow
```

Expected state: `SHADOW_ENROLLED`. Confirm the qualification namespace, production lineage,
prediction hash, isolation fields, and unchanged production Shadow bundle.

### 5. Verify Observation Integration

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to observation
```

The checkpoint does not run Observation, fabricate rows, or backfill historical OOS predictions.
Immediate state is normally `OBSERVATION_PENDING`; exact mature prospective rows may produce
`OBSERVATION_ACCUMULATING`. Either state permits `QUALIFIED` after protected invariants pass.

### 6. Status, Recovery, and Cancellation

```bash
ashare-quant --config config/default.yaml retraining qualification-status \
  --run-id QUALIFICATION_RUN_ID
ashare-quant --config config/default.yaml retraining qualification-recovery \
  --run-id QUALIFICATION_RUN_ID
ashare-quant --config config/default.yaml retraining qualification-cancel \
  --run-id QUALIFICATION_RUN_ID --reason "OPERATOR_REASON"
```

Recovery is read-only. Cancellation appends a terminal event and deletes nothing.

## Preflight and Safety

Preflight validates exact request/evidence hashes, parent model, policy/config availability, locks,
conflicting lifecycle state, stale staging/backup paths, resource capacity, and the daily
qualification limit. Existing lifecycle dry-run and readiness services enforce scheduler,
closed-loop, governance, policy, processed-artifact, and Training Request contracts.

Protected baselines cover Registry, Champion history and artifact, Promotion/Approval state,
Rollback state, Paper Trading ledgers, production candidates, and production Shadow. Any change
causes `FAILED`; no destructive automatic rollback is attempted.

Lock order is:

```text
runs/.retraining-qualification.lock
  -> runs/.retraining-lifecycle.lock
  -> runs/.production.lock
```

Before training, policy/config drift blocks execution. After training, validation and Shadow may
continue under the frozen qualification contract, but immutable data and completed stage evidence
must still match.

The current GitHub Actions CI issue is deferred technical debt. This phase does not investigate or
modify CI. Local fixture-only verification remains required.
