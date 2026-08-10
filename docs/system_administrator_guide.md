# A-share Quant System Administrator Guide

## Daily Operation

The production pipeline remains the only command that may update market data, processed features,
Champion predictions, candidates, and virtual Paper Trading state:

```bash
ashare-quant --config config/default.yaml pipeline production
```

The integrated closed loop subsequently attempts prospective Shadow Prediction, Monitoring,
Research Agent generation, and a dated Governance snapshot. These are soft operational components:
their failures are recorded as warnings and do not alter Champion candidates or Paper Trading. The
run publishes `reports/YYYYMMDD/closed_loop_manifest.json` with component run IDs, durations,
artifact hashes, and warnings.

Standalone operational checks remain available:

```bash
ashare-quant --config config/default.yaml monitor run --as-of YYYYMMDD
ashare-quant --config config/default.yaml research-agent generate --as-of YYYYMMDD
ashare-quant --config config/default.yaml governance status
ashare-quant --config config/default.yaml governance validate-production
```

Expected artifacts include:

```text
reports/YYYYMMDD/production_summary.json
reports/shadow_predictions/YYYYMMDD/
reports/model_monitor/YYYYMMDD/
reports/research_agent/YYYYMMDD/
reports/governance/YYYYMMDD/
```

## Scheduler Guidance

Recommended Asia/Shanghai schedule:

```text
18:30  pipeline production
19:00  monitor run --as-of resolved completed session
19:15  research-agent generate --as-of resolved completed session
Weekly governance validate-production
Weekly governance validate-recovery
```

The standalone 19:00 and 19:15 jobs are idempotent verification/retry opportunities. Do not add
automatic promotion or rollback timers. Use the existing production lock for the integrated daily
pipeline and do not add a second lock implementation.

## Model Lifecycle

The only permitted lifecycle is:

```text
Candidate -> Evidence -> Gate -> Review -> Approve -> Apply -> Monitor

### Retrained Challenger lifecycle

Governed retraining requests run separately from the daily Production Pipeline:

Before an intended training run, inspect the explicit LightGBM backend:

```bash
ashare-quant --config config/default.yaml models training-backend-status
```

CPU is the safe default. CUDA must be configured explicitly and pass the LightGBM smoke probe.
Keep CPU fallback disabled for governed execution unless an audited fallback is intentionally
acceptable. Use the isolated CPU/CUDA benchmark and consistency comparison before enabling CUDA;
see `docs/training_compute_backend.md`. Diagnostics, inference, Shadow, Paper Trading, and
monitoring remain CPU-only.

```bash
ashare-quant --config config/default.yaml retraining lifecycle-run \
  --request-id REQUEST_ID

ashare-quant --config config/default.yaml retraining lifecycle-status \
  --run-id LIFECYCLE_RUN_ID

ashare-quant --config config/default.yaml retraining lifecycle-resume \
  --run-id LIFECYCLE_RUN_ID
```

Use `--stop-after readiness|training|validation|shadow` for controlled checks. Inspect ambiguous
runs with `retraining lifecycle-recovery --run-id LIFECYCLE_RUN_ID`. `EVIDENCE_READY` allows only
separate evidence preparation and never changes Champion automatically. See
`docs/retrained_challenger_lifecycle.md` for the full state and recovery contract.
```

Before execution, run the no-training rehearsal:

```bash
ashare-quant --config config/default.yaml retraining lifecycle-dry-run \
  --request-id REQUEST_ID
```

Training budgets and lifecycle cooldown dates use the configured production timezone, normally
Asia/Shanghai. Failed model fits still consume a daily attempt. Corrupt lifecycle history blocks
training. Shadow refresh failures after successful enrollment preserve all successful evidence.

When status reports `POLICY_REVIEW_REQUIRED`, revalidate exact evidence before creating a
Promotion Request:

```bash
ashare-quant --config config/default.yaml retraining lifecycle-revalidate-evidence \
  --run-id LIFECYCLE_RUN_ID
```

Only exact model, horizon, training, validation, Shadow, observation, monitoring, alert, and any
policy-required Paper Trading lineage is accepted.

Discover and freeze current evidence:

```bash
ashare-quant --config config/default.yaml models promotion prepare --model-id MODEL_ID
```

Validate with the versioned policy in `config/promotion_policy.yaml`:

```bash
ashare-quant --config config/default.yaml models promotion validate \
  --request-id REQUEST_ID
```

Review and approve using the dedicated reviewer configuration, then preview the transition:

```bash
ashare-quant --config config/promotion_review.yaml models promotion review \
  --request-id REQUEST_ID
ashare-quant --config config/promotion_review.yaml models promotion approve \
  --request-id REQUEST_ID --comments-file review.txt
ashare-quant --config config/promotion_review.yaml models promotion apply \
  --request-id REQUEST_ID --dry-run
```

Only after the dry-run matches the reviewed request may an operator omit `--dry-run`. No metric,
alert, scheduler, or research-agent output can apply a promotion automatically.

## Rollback

Rollback only targets a historical Champion in the same deployment slot:

```bash
ashare-quant --config config/promotion_review.yaml models promotion rollback-create \
  --model-id HISTORICAL_CHAMPION_ID --reason-file rollback_reason.txt
ashare-quant --config config/promotion_review.yaml models promotion rollback-validate \
  --request-id REQUEST_ID
ashare-quant --config config/promotion_review.yaml models promotion review \
  --request-id REQUEST_ID
ashare-quant --config config/promotion_review.yaml models promotion approve \
  --request-id REQUEST_ID --comments-file rollback_review.txt
ashare-quant --config config/promotion_review.yaml models promotion rollback-apply \
  --request-id REQUEST_ID
```

There is no automatic rollback.

## Disaster Recovery

Preview the latest lineage-valid Registry without changing files:

```bash
ashare-quant --config config/default.yaml governance recover-registry --dry-run
ashare-quant --config config/default.yaml governance validate-recovery
```

Recovery order is model artifacts, immutable Registry versions, Champion history, current Registry,
production reports, observations, Monitoring, then Paper Trading ledgers. Preserve interrupted apply
journals before analysis. A governed Registry version is recoverable only when its Champion assignment
and promotion or rollback apply manifest hashes also match. The recovery assistant never restores bytes
automatically.

## Forbidden Operations

Never manually edit `models/registry.json`, overwrite model artifacts, modify historical prediction
or performance observations, delete Champion history, bypass human approval, use frozen OOS reports
as prospective production evidence, or treat the LLM Research Agent as a trading or promotion
authority.

## Controlled Operational Qualification

Use `retraining qualification-start` for preflight, dry-run, and readiness. Real training and
Shadow are disabled by default. Each privileged stage requires an unchanged static policy, an
enabled runtime capability, a short-lived single-use `qualification-authorize` artifact, and a
separate `qualification-advance` command. Capability switches do not change qualification identity
and never replace authorization. An exact repeated `ACTIVE` authorization request is idempotent and
does not extend its expiry. Expired, revoked, consumed, and stale records may be reviewed again with
the same operator and reason, producing a new immutable authorization. A different request cannot
replace an existing active authorization: inspect it, revoke it explicitly, confirm `REVOKED`, then
issue the replacement. Failed training or Shadow attempts consume their authorization, so a retry
requires both the explicit pending retry state and a new authorization. Never skip checkpoints or
treat `QUALIFIED` as Promotion approval. See `docs/controlled_operational_qualification.md` for the
complete procedure and recovery commands.

## Backtest Evidence Integrity

Before treating a backtest as governance evidence, verify that its manifest uses schema version 2,
has `purpose=OOS_EVIDENCE`, records a model boundary and exact model-manifest hash, and records
`accounting_schema_version=2` plus a `cost_policy_hash`. A live model is not automatically valid for
historical performance measurement: evaluation must begin strictly after its immutable training and
selection boundary.

Suspended holdings use their last valid close for valuation but remain untradeable. Unexplained quote
gaps and unresolved exits fail evidence-grade runs. The sell-delay threshold never authorizes a zero
write-off; only a verified terminal-security event does.

Legacy or suspect backtests must not be edited or deleted. Publish a review record instead:

```bash
ashare-quant --config config/default.yaml backtest invalidate \
  --backtest-id BACKTEST_ID \
  --reason IN_SAMPLE_MODEL_EVALUATION \
  --reviewed-by OPERATOR \
  --note "Reason and replacement plan"
```

The default cost schedule resolves sell-side stamp duty by effective trade date, including the
2023-08-28 transition. Promotion evidence preparation rejects executable validation generated under
the legacy accounting schema or without exact cost-policy identity. See `docs/backtest.md` for the
metric formulas and artifact contract.
# Research Validation and Prospective Lockbox

Research policy is stored in `config/research_policy.yaml`. The prospective lockbox begins on
`20260810`. Do not move this boundary or change its consumer lists without treating the change as a
new reviewed research policy identity. A forbidden research command that reaches the boundary must
stop with `RESEARCH_LOCKBOX_VIOLATION`; do not bypass it by editing generated manifests.

Walk-forward gaps default to `auto`: H5/H10/H20/H60 shared plans resolve to 61 purge and embargo
sessions. An explicit smaller value is a configuration error. Inspect planned resolved gaps and the
research policy hash before starting an expensive run.

Multi-fold runs write only to `reports/research/walk_forward/<run_id>`. Use `models
walk-forward-status` after completion and `models walk-forward-recovery` after interruption.
Recovery is read-only. A missing/corrupt child manifest or source hash requires operator review;
never average only the remaining folds. The legacy robust feature set is marked
`LEGACY_PROVENANCE_INCOMPLETE` and cannot support new governed multi-fold evidence until a new
feature-set artifact is created from exact immutable diagnostics.

Full commands and evidence semantics are documented in
[`research_validation.md`](research_validation.md).
