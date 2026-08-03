# Production Incident Response

## First Response

1. Stop manual production invocations and allow the production lock owner to finish or fail.
2. Record the failing run ID and preserve its manifest and logs.
3. Run `ashare-quant --config config/default.yaml governance status`.
4. Run both governance validation commands. Do not repair before preserving evidence.

## Data Failure

Stop downstream publication. Validate raw and processed manifests, gap status, and readiness gates.
Restore or rebuild data only through documented CLI commands. Never edit Parquet partitions by hand.
After repair, rerun production explicitly for the affected session and validate production again.

## Model Failure

Keep the current Champion assignment unchanged while investigating. Do not promote a challenger to
mask an inference or artifact problem. If the current Champion is operationally unsafe, use the
governed rollback procedure and require human approval.

## Monitoring Failure

Monitoring is read-only and must not mutate prediction or Paper Trading state. Preserve the last
successful monitoring snapshot, repair missing/corrupt monitor inputs, and rerun `monitor run`.
Failure of monitoring does not authorize bypassing production readiness or promotion gates.

## Registry Failure

Do not manually reconstruct `registry.json`. Run `governance validate-recovery`, identify the latest
valid immutable registry version, verify its referenced models and Champion history, then follow the
manual restoration procedure in `recovery_manual.md`.

## Escalation Evidence

Preserve run manifests, governance reports, systemd journal excerpts, current registry bytes,
registry versions, Champion history, and relevant artifact manifests. Never delete an interrupted
apply journal before determining whether the current registry matches its parent or target version.
