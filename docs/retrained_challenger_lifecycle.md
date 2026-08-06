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

## Failure Recovery

1. Run `lifecycle-status` and `lifecycle-recovery`.
2. Verify the latest completed lower-level artifact and its manifest.
3. Correct an external cause and use `lifecycle-resume` only for an unambiguous stage state.
4. For interrupted training, run the existing retraining execution recovery inspection first.
5. Never edit lifecycle events, model artifacts, `registry.json`, or Champion history manually.

## Governance Boundaries

The orchestrator never approves, applies, or rolls back a model. It never modifies Champion,
`registry.json`, Paper Trading, candidate selection, features, hyperparameters, or orders.
