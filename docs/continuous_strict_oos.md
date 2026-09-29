# Continuous STRICT_OOS Execution Evidence

## Scope

`backtest continuous-oos` replays frozen, validated contract-9 H5 predictions. It does not
train, rescore, read labels, select features, repair data, or authorize production trading.
The source run must be COMPLETE; `validate_completed_walk_forward_artifact` validates its
root and leaves. Selection uses each immutable fold's `STRICT_OOS` classification, never
ranking performance or label maturity. Replay folds are excluded. Integrity verification
hashes existing ranking-metric files as opaque bytes; their values are never consumed.

Every selected prediction is preserved, including the last partly label-mature fold.
`stitched_predictions.parquet` preserves original scores and source fold/prediction/model
hashes. `signal_lineage.parquet` binds each exchange session to exactly one fold and model.
Missing calendar sessions, overlapping ownership, duplicate keys, and nonfinite scores fail
closed. Arrow content hashes normalize chunks so serialization layout is not identity.

## Execution Contract

Contract: `continuous_strict_oos_replay_v1`. Accounting remains schema 3.
`execution_semantics_changed = false`.

One invocation of the existing `simulate_portfolio` runs for each of Top10, Top20, Top50.
Every account starts once. Cash, positions, original basis, target exits, delayed-exit and
stale-valuation state persist across fold and month boundaries. The capital policy is
`existing_engine_all_available_cash_v1`, not staggered H5 sleeves or daily rebalancing.
Repeated held-security signals do not restart positions. The three accounts share exactly
the same stitched source. Equal scores retain the existing security-code tie break.

The service reuses `load_calendar`, `load_execution_prices`, `load_benchmark`, the current
cost policy and corporate-action policy. Price loading restricts codes to the union of daily
Top50 and the existing loader's transition-code closure. No outcome filters enter this set.
Modeling labels/features are not inputs. The separately governed execution Universe, frozen
source snapshot, and matching PASS lifecycle scan are mandatory. The existing contract-9
tail/cutoff logic determines the maximum execution date before the prospective lockbox.
Normal H+1 planned exits are checked before simulation; delayed exits can still fail at cutoff.

Unsupported held transition crossings fail with `UNSUPPORTED_REAL_TRANSITION_CROSSING` and
the engine's Top-N, position ID, predecessor, successor, entry, transition and target-exit
details. No portfolio or security is silently dropped. No COMPLETE artifact is published on
failure. Known supported transformations remain non-trade events under the existing policy.

## Ledgers And Interpretation

The optional `record_execution_audit` engine flag only records observations. Tests compare
old and instrumented fills, costs, daily equity, holdings, metrics and corporate actions.
For every selected signal, `execution_intents.parquet` records signal date, entry-attempt
date, rank, code, cash before entries, position ID where present, and one final outcome:
FILLED, NOT_BUYABLE, ALREADY_HELD, NO_AVAILABLE_CASH, INSUFFICIENT_AFFORDABLE_GROSS, TERMINAL.
`selected=true` is not an order submission. Outcomes follow engine checks, after scheduled
exits, and do not fabricate exchange rejection messages. When multiple conditions apply,
held/terminal/not-buyable status is recorded before the cash condition.

Each Top-N publishes daily_returns, trades, holdings, execution_intents, corporate_actions,
terminal_events and daily_accounting_reconciliation Parquet files, plus metrics,
accounting_summary and evidence_summary JSON files. Terminal events preserve the prior mark,
last valid quote date, entry/target dates and holding sessions. **Terminal writeoff uses the
current zero-cash-recovery accounting assumption; it is not a successful cash sale.**

Reconciliation verifies cash + holdings = equity, share marks, effective-dated trade costs,
cash flows, turnover, position continuity and closure, corporate-action evidence and metrics.
Numerical reconciliation uses rtol=1e-10 and atol=1e-6; the corporate-action engine retains
its stricter existing mark tolerance. Stale marks are valuations, never executable cash.
Locked capital is marked value for holdings strictly past target exit. Daily cash/invested
ratios, maximum/mean locked fraction, delayed attempts, breach resolutions, stale position-days,
cost components and yearly costs remain inspectable. Ratios at zero equity are null, with a
count of undefined days. Approximately zero cash uses absolute tolerance 1e-6.

The deeper economic market/realized/writeoff/dividend P&L decomposition is **NOT_PROVEN** in
this phase. Raw ledgers are retained for the separate P0-C accounting audit. Neither raw-price
economics nor terminal recovery policy has been changed.

Metrics and monthly/yearly attribution derive from each single continuous daily curve, not
old fold returns. `comparison.json` provides full-lifecycle metrics and a common-window
comparison ending at the minimum actual account resolution date. Common-window open holdings
remain marked, not force-sold; trade win statistics include only closed trades by that date.
No Top-N winner is selected. This is inspected historical OOS evidence, not pristine holdout
evidence or qualification.

## Publication And Validation

Logical identity binds source run/manifest, selected fold and leaf hashes, normalized stitched
content, ownership, signal coverage, raw snapshot, execution Universe, lifecycle scan,
transition/mapping/catalog/metadata identities, policies, cutoff/lockbox, calendar, settings,
Top-N, capital policy and implementation content hashes. Absolute operational paths live only
in source inventory locators. No host, PID, timestamp or mtime enters logical identity.

Publication writes a staging directory, writes the manifest last, runs the complete public
validator, then atomically renames to the content-defined target. An existing target is
validated and returned, never overwritten. A changed executable dependency fails current-code
validation; historical artifacts are not silently upgraded.

`validate_continuous_strict_oos_artifact(Path(...))` recursively revalidates the COMPLETE
source run and execution inputs, exact selected predictions/ownership, exact Top-N/child sets,
child hashes, costs, position histories, accounting, metrics and comparisons. Rehashed but
inconsistent business tables fail too. Labels need not be present or accessible.

## Reviewed Operator Launch

Implementation review baseline: `main`, `a88b7a02e9cca686397df63ddc5e2c51c45e5c93`.
Initial tracked tree and index were clean. No operator files were edited, staged or deleted.
Owned changes (not committed or pushed):

- `src/ashare_quant/backtest/continuous.py`: preflight, existing-engine orchestration, publication.
- `src/ashare_quant/backtest/continuous_source.py`: label-free immutable prediction selection.
- `src/ashare_quant/backtest/continuous_evidence.py`: derived accounting and comparison evidence.
- `src/ashare_quant/backtest/continuous_validation.py`: recursive and business validation.
- `src/ashare_quant/backtest/engine.py`: optional intent/writeoff observations and error context.
- `src/ashare_quant/cli/__init__.py`: thin `continuous-oos` command.
- `tests/test_continuous_oos_backtest.py`: 40 fixture/regression cases.
- `docs/continuous_strict_oos.md` and `docs/research_validation.md`: contract/operator documentation.

Observed read-only real-source preflight: PASS, 42 STRICT_OOS folds, 3,712,578 predictions,
835 sessions, 20230201 through 20260710, no duplicate keys or nonfinite scores. Source manifest:
`295db1d4d4700ca659a1154df70d57dfc19a77553f78743aba9243a5828375be`.
Predicted output identity under the reviewed source content:
`continuous_strict_oos_f0eeb5ddd6e0faf28bf4feeb` (NOT PUBLISHED / NOT RUN).
Original run, snapshot, execution Universe and scan hashes matched the operator's supplied
values on final recheck. No production mutation service was invoked; there was no full
before/after inventory of independently running production state.

Checks run with `.venv/bin/python -m`: full `pytest` (992 passed), focused `pytest
tests/test_backtest.py tests/test_backtest_integrity.py tests/test_corporate_action_accounting.py
tests/test_security_identity_transition.py tests/test_continuous_oos_backtest.py -q`
(100 passed), `ruff check .`, `ruff format --check .`, `mypy src`, and `git diff --check`
(all passed). The shell block below passed `bash -n`; it was not launched. All actual
portfolio simulations in this task used synthetic fixture data, not the real prediction set.

This command is for the operator **after code review**. It was not executed by the agent.
It uses the explicitly selected D.8A inputs, not a newest-directory search. It refuses an
existing session; the content-defined publisher also returns an already COMPLETE artifact.
Before retrying after disconnection, inspect the session and exit file rather than launching
a duplicate. The CLI prints the validated artifact path on success.

```bash
set -euo pipefail
cd /home/zhangkangle/workspaces/ashare-lgbm-quant
git diff --exit-code
git diff --cached --exit-code
session=continuous_h5_oos
if tmux has-session -t "$session" 2>/dev/null; then
  echo "SESSION_EXISTS: inspect with tmux attach -t $session"; exit 2
fi
logdir="$PWD/logs/manual_continuous_oos/$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=16 HEAD)"
mkdir -p "$logdir"
cmd=(.venv/bin/ashare-quant
  --config reports/research_d6_a7f5af4bdd7fe12b_20260710/d6_research_config.yaml
  backtest
  --storage-root data/research/source_snapshots/research_source_snapshot_1f67e31303cb859ab0209a96/datasets
  --output-root reports/research_d8a_continuous_h5/research/continuous_strict_oos
  continuous-oos --walk-forward-run-id walk_forward_3e606ceb8eb60578
  --walk-forward-reports-root reports/research_d6_a7f5af4bdd7fe12b_20260710
  --execution-processed-root data/research/d7_execution_1f67e313_20260807
  --lifecycle-scan-manifest reports/research_d7_execution_1f67e313_20260807/security_lifecycle/security_lifecycle_8927c190463bd7c2f9fd1f8f/manifest.json
  --identity-transition-artifact reports/research_ia3_identity_v2_20260710/security_identity_transitions/security_identity_transitions_2ddbcfb06f1ac5189cd78431)
printf -v command_line '%q ' "${cmd[@]}"
printf '%s\n' "$command_line" > "$logdir/command.txt"
git rev-parse HEAD > "$logdir/head.txt"
printf -v workdir '%q' "$PWD"
printf -v destination '%q' "$logdir"
launch="cd $workdir || exit; set -o pipefail; logdir=$destination; date -u +%FT%TZ > \"\$logdir/start.txt\"; $command_line 2>&1 | tee \"\$logdir/replay.log\"; codes=(\"\${PIPESTATUS[@]}\"); printf 'cli=%s tee=%s\\n' \"\${codes[0]}\" \"\${codes[1]}\" > \"\$logdir/exit_code.txt\"; date -u +%FT%TZ > \"\$logdir/end.txt\""
tmux new-session -d -s "$session" bash -c "$launch"
printf 'Attach: tmux attach -t %s\nLog: tail -n 100 -f %s/replay.log\nStatus: cat %s/exit_code.txt\n' "$session" "$logdir" "$logdir"
```

Detach with Ctrl-b then d. Reconnect with `tmux attach -t continuous_h5_oos`.
Read-only preflight uses the same CLI arguments with `--preflight-only` appended; it never
simulates or publishes. No feature/label rebuild, training, full H5 rerun, or Label Support
implementation is authorized by this command.
