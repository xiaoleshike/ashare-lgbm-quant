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
```

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
