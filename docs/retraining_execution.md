# Governed Retraining Execution

Phase 2.8.2B consumes one immutable retraining request only after the matching
execution-readiness artifact reports `READY`. It produces a new Challenger and
does not alter the Champion or `models/registry.json`.

For controlled qualification, execution accepts a typed context. The deterministic model identity,
artifact, execution record, and candidate registration bind `qualification_run_id` and
`qualification_only=true`. The registration remains outside `registry.json`, and CLI flags cannot
construct this context outside the qualification service.

## Safety Contract

Before loading training rows, execution revalidates the request, evidence,
retraining policy, promotion policy, configuration, and the frozen feature,
universe, and label manifests. The repository production lock is held while
the dataset is loaded and the model is trained, preventing concurrent processed
artifact publication.

The first implementation supports only `challenger_refresh`. It reuses the
source model's frozen feature list and a compatible multi-horizon experiment and
walk-forward selection fold. It loads only that fold's training and validation
periods. Final-test rows, evaluation labels, and production observations are not
loaded.

## Commands

Validate readiness first:

```bash
ashare-quant --config config/default.yaml retraining readiness \
  --as-of YYYYMMDD \
  --request-id REQUEST_ID
```

Execute the frozen request:

```bash
ashare-quant --config config/default.yaml retraining execute \
  --request-id REQUEST_ID
```

Before execution, use `retraining lifecycle-dry-run --request-id REQUEST_ID`. It checks current
readiness, the Asia/Shanghai daily training-attempt budget, and lifecycle cooldown without invoking
LightGBM or creating a candidate. Request cooldown controls trigger creation; lifecycle cooldown
separately controls repeated training starts for the same parent model and horizon.

Inspect lifecycle status:

```bash
ashare-quant --config config/default.yaml retraining execution-status \
  --run-id TRAINING_RUN_ID
```

Inspect and clean unpublished staging after an interruption:

```bash
ashare-quant --config config/default.yaml retraining recovery \
  --run-id TRAINING_RUN_ID
```

Recovery never resumes an unknown training state automatically. After recovery,
an operator may rerun the same request; the deterministic identity makes a
completed rerun idempotent.

## Artifacts

The model is published under `models/challengers/<model_id>/`. Candidate status
is recorded independently under `models/candidate_registrations/<model_id>/` so
this phase does not mutate `models/registry.json`. Execution completion is stored
under `reports/retraining/executions/<training_run_id>/`, while lifecycle events
are append-only under `reports/retraining/execution_journals/`.

Model artifact publication, candidate registration, and execution completion
are one filesystem transaction. Each directory writes its manifest last before
the staged directories are atomically moved into place. A failed transaction
does not expose a complete Challenger.

## Limitations

- Only fixed-parameter LightGBM Ranker Challenger refreshes are supported.
- A compatible horizon experiment and walk-forward schema-v2 fold plan must
  already exist.
- Candidate registration is deliberately separate from the production model
  registry. Governance can import or evaluate it in a later phase.
- Recovery removes only unpublished staging for the specified run and never
  repairs or overwrites a published artifact.
