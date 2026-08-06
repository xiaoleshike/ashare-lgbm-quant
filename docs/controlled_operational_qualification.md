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

Phase 2.8.2G.1 separates three controls:

1. The static policy is the frozen safety contract and participates in qualification identity.
2. `allow_real_training` and `allow_real_shadow` are runtime kill switches and are excluded from
   the static identity hash.
3. A stage authorization is one asserted operator decision for one execution attempt.

Changing a runtime switch does not change `qualification_run_id`. A true switch is not an
authorization, an authorization cannot override a false switch, and `qualification-advance` never
creates an authorization. `approved_by` is an auditable asserted identity, not a cryptographic
signature.

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
  authorizations/{training,shadow}/<authorization_id>/
  authorization_revocations/<authorization_id>/<revocation_id>/
  authorization_consumptions/<authorization_id>/<consumption_id>/

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

### 2. Authorize Training

Authorization may be created while the runtime capability is disabled:

```bash
ashare-quant --config config/default.yaml retraining qualification-authorize \
  --run-id QUALIFICATION_RUN_ID --stage training \
  --approved-by OPERATOR_ID --reason "Controlled real training qualification"

ashare-quant --config config/default.yaml retraining qualification-authorization-status \
  --run-id QUALIFICATION_RUN_ID --stage training
```

The default validity is 60 minutes and the configured maximum is 240 minutes. Expired, revoked,
stale, or consumed authorizations cannot execute.

Authorization creation uses exact active idempotency. Repeating the same request while its exact
snapshot-bound authorization is still `ACTIVE` returns the same `authorization_id` without writing
another event or extending `issued_at` or `expires_at`. An expired, revoked, consumed, or stale
authorization is retained as history but does not block a new review using the same `approved_by`
and reason; the current snapshot and new validity window produce a new authorization identity.

Only one `ACTIVE` authorization may exist for a stage and reviewed snapshot. A different approver,
reason, or explicit expiry returns `ACTIVE_AUTHORIZATION_CONFLICT`; it does not supersede the active
record. Inspect and explicitly revoke that authorization before issuing its replacement. Corrupt or
ambiguous authorization, revocation, or consumption storage fails closed and requires recovery
inspection.

### 3. Enable and Run Training

After explicit review, set `allow_real_training: true` and run:

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to training
```

Expected state: `VALIDATION_PENDING_APPROVAL`. Verify dataset and fold identities, feature/label
hashes, candidate registration, final-test exclusion, and qualification fields. Failed attempts and
retries consume the Asia/Shanghai daily training-attempt budget. Entering `TRAINING` also consumes
the authorization even when training fails; retry requires a new authorization.
After the lifecycle explicitly returns to `TRAINING_PENDING_APPROVAL`, issue that new authorization;
the consumed authorization and failed-attempt receipts remain immutable.

### 4. Run Validation

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to validation
```

Expected state: `SHADOW_PENDING_APPROVAL`. Inspect offline metrics, executable OOS evidence,
leakage contracts, unresolved-position rejection, and Shadow eligibility.

### 5. Authorize, Enable, and Run Isolated Shadow

After reviewing validation, authorize the exact current snapshot:

```bash
ashare-quant --config config/default.yaml retraining qualification-authorize \
  --run-id QUALIFICATION_RUN_ID --stage shadow \
  --approved-by OPERATOR_ID --reason "Controlled qualification Shadow enrollment"
```

Then set `allow_real_shadow: true` and run:

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to shadow
```

Expected state: `SHADOW_ENROLLED`. Confirm the qualification namespace, production lineage,
prediction hash, isolation fields, and unchanged production Shadow bundle.

### 6. Verify Observation Integration

```bash
ashare-quant --config config/default.yaml retraining qualification-advance \
  --run-id QUALIFICATION_RUN_ID --to observation
```

The checkpoint does not run Observation, fabricate rows, or backfill historical OOS predictions.
Immediate state is normally `OBSERVATION_PENDING`; exact mature prospective rows may produce
`OBSERVATION_ACCUMULATING`. Either state permits `QUALIFIED` after protected invariants pass.

### 7. Revocation, Status, Recovery, and Cancellation

An unconsumed authorization can be revoked without deleting it:

```bash
ashare-quant --config config/default.yaml retraining qualification-revoke-authorization \
  --run-id QUALIFICATION_RUN_ID --authorization-id AUTHORIZATION_ID \
  --revoked-by OPERATOR_ID --reason "Authorization withdrawn"
```

To replace an active authorization, first inspect status, revoke the active ID, confirm it is
`REVOKED`, and then run `qualification-authorize` again. The replacement binds to the post-revocation
snapshot. There is no force, replace, supersede, or authorization-reuse option.

```bash
ashare-quant --config config/default.yaml retraining qualification-status \
  --run-id QUALIFICATION_RUN_ID
ashare-quant --config config/default.yaml retraining qualification-recovery \
  --run-id QUALIFICATION_RUN_ID
ashare-quant --config config/default.yaml retraining qualification-cancel \
  --run-id QUALIFICATION_RUN_ID --reason "OPERATOR_REASON"
```

Recovery is read-only and reports corrupt hashes, stale bindings, multiple active records, implicit
supersession, duplicate claims, claims without terminal receipts, invalid revocation/consumption
lineage, static-policy drift, and legacy identity ambiguity. Historical expiry or revocation alone
is not corruption.
Cancellation appends a terminal event and deletes nothing.

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

Before training and Shadow, static policy/config drift blocks execution. Runtime capability changes
do not alter identity or completed checkpoints. After training, validation and Shadow may
continue under the frozen qualification contract, but immutable data and completed stage evidence
must still match.

The current GitHub Actions CI issue is deferred technical debt. This phase does not investigate or
modify CI. Local fixture-only verification remains required.
