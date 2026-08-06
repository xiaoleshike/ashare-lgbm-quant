# Retrained Challenger Lifecycle

## Scope

The lifecycle orchestrator coordinates existing governed services. It does not contain model
training, validation, Shadow scoring, observation, promotion, or trading business logic.

```text
Training Request
  -> Execution Readiness
  -> Challenger Training
  -> Offline/Executable Validation
  -> Prospective Shadow Enrollment
  -> Mature Observation Tracking
  -> Evidence Preparation Readiness
```

`EVIDENCE_READY` means only that immutable inputs can be collected by the separate promotion
workflow. It does not mean `PROMOTION_ELIGIBLE`, gate approval, human approval, or promotion.

## State Model

The normal path is:

```text
REQUEST_ACCEPTED
READINESS_CHECKING
READINESS_READY
TRAINING
TRAINING_COMPLETED
VALIDATING
VALIDATION_COMPLETED
SHADOW_ENROLLING
SHADOW_ENROLLED
OBSERVATION_PENDING
OBSERVATION_ACCUMULATING
OBSERVATION_SUFFICIENT
EVIDENCE_READY
```

Stage-specific failure states prevent downstream execution. Ambiguous `TRAINING` or incomplete
snapshot states require operator inspection; the orchestrator does not guess whether training
completed.

All transitions are append-only events. The files under
`reports/retraining/lifecycle/<lifecycle_run_id>/` are an atomically published materialized
snapshot of those events.

## Commands

Run the complete currently available lifecycle:

```bash
ashare-quant --config config/default.yaml retraining lifecycle-run \
  --request-id REQUEST_ID
```

Stop at an operational checkpoint:

```bash
ashare-quant --config config/default.yaml retraining lifecycle-run \
  --request-id REQUEST_ID \
  --stop-after validation
```

Supported checkpoints are `readiness`, `training`, `validation`, and `shadow`.

Inspect, resume, and perform read-only recovery inspection:

```bash
ashare-quant --config config/default.yaml retraining lifecycle-status \
  --run-id LIFECYCLE_RUN_ID

ashare-quant --config config/default.yaml retraining lifecycle-resume \
  --run-id LIFECYCLE_RUN_ID

ashare-quant --config config/default.yaml retraining lifecycle-recovery \
  --run-id LIFECYCLE_RUN_ID
```

The recovery command never deletes staging paths or changes model state.

## Prospective Observation

Observation tracking reads only completed `performance_observation` artifacts. It does not read
labels directly. A session counts only when all of these hold:

- `model_origin` is `retrained_challenger`;
- model, horizon, training run, and validation run identities match;
- `label_status` is `available`;
- the mature excess return is present.

Champion, research Challenger, historical OOS, immature, and mature-unavailable rows do not
count. Required sessions are configured by horizon in `config/retraining_policy.yaml`.

Run `lifecycle-resume` after a new production session to publish the model's latest prospective
Shadow sidecar and refresh observation progress. The lifecycle remains separate from the hard
daily Production Pipeline.

## Evidence Hardening

Initial `shadow_enrollment` and later `shadow_refresh` attempts are separate. After enrollment
succeeds, a failed refresh is an append-only operational warning and cannot erase the enrollment
manifest, its hash, or earlier successful `shadow_run_id` values. `SHADOW_FAILED` applies only
before the first successful enrollment.

Observation progress appends an event whenever mature-session coverage, cutoff, accepted Shadow
identities, source artifacts, or the aggregate evidence hash changes. Removed or mutated historical
observation artifacts fail closed.

Promotion evidence must match the exact request, parent, model, horizon, training run, validation
run, accepted Shadow runs, and observation/monitoring cutoff. An unrelated Paper Trading portfolio
never satisfies a required Paper Trading gate. Without a dedicated retrained portfolio, evidence
remains `NOT_READY`.

## Frozen Identity and Policy Review

The lifecycle ID freezes the training-request hash, creation-time retraining/lifecycle/Promotion
Policy hashes, and creation-time config hash. Readiness run IDs, current config, timestamps, and
daily Shadow/Observation identities do not change it. `lifecycle-resume` loads that identity
directly instead of recursively invoking `lifecycle-run`.

If the current Promotion Policy differs from the frozen or last evaluated policy, status becomes
`POLICY_REVIEW_REQUIRED`. Revalidate exact evidence without retraining:

```bash
ashare-quant --config config/default.yaml retraining lifecycle-revalidate-evidence \
  --run-id LIFECYCLE_RUN_ID
```

`EVIDENCE_READY` means only that exact immutable evidence can be prepared under the recorded
evaluated Promotion Policy. It does not mean Promotion Gate PASS, human approval, Champion
replacement, or automatic deployment.

## Operational Controls

Training-attempt budgets use `production.timezone`, defaulting to `Asia/Shanghai`. Every transition
into `TRAINING`, including failed and retried attempts, consumes the budget. Unreadable or
incomplete lifecycle history blocks training. `lifecycle.cooldown_days` blocks the first training
attempt of another lifecycle for the same parent model and horizon. It does not block same-run
resume, Shadow refresh, observation tracking, evidence revalidation, or recovery. Request cooldown
prevents repeated trigger requests; lifecycle cooldown prevents repeated model fits.

Lock order is lifecycle orchestration lock first, then lower-level production/execution locks. A
lower-level service must never acquire the lifecycle lock. `max_parallel_runs` remains one.

## Controlled Dry Run

```bash
ashare-quant --config config/default.yaml retraining lifecycle-dry-run \
  --request-id REQUEST_ID
```

The dry run checks request integrity, readiness, policy, lock availability, cooldown, and daily
budget. It never trains, validates a model, scores Shadow predictions, reads observations, creates
Promotion evidence, or changes production state.

For a future operator-authorized historical rehearsal:

1. Select a historical trading date with complete immutable production artifacts.
2. Confirm no production or training process is active.
3. Run `lifecycle-dry-run` and inspect every warning.
4. Stop if any warning is unexplained.
5. After separate authorization, run `lifecycle-run --stop-after readiness` only.
6. Do not proceed to real training without explicit authorization.

## Failure Recovery

1. Run `lifecycle-status` and `lifecycle-recovery`.
2. Verify the latest completed lower-level artifact and its manifest.
3. Correct an external cause and use `lifecycle-resume` only for an unambiguous stage state.
4. For interrupted training, run the existing retraining execution recovery inspection first.
5. Never edit lifecycle events, model artifacts, `registry.json`, or Champion history manually.

## Governance Boundaries

The orchestrator never approves, applies, or rolls back a model. It never modifies Champion,
`registry.json`, Paper Trading, candidate selection, features, hyperparameters, or orders.

## Controlled Operational Qualification

Phase 2.8.2G adds an independent, explicitly advanced qualification state machine. It reuses
lifecycle dry-run/readiness and the governed execution, validation, and retrained Shadow services.
Qualification artifacts carry `qualification_only=true` and cannot enter Promotion or trading
discovery. See `docs/controlled_operational_qualification.md` for the exact operator runbook.
