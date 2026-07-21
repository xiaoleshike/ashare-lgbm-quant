# Purged Walk-forward Planning

`ashare-quant models walk-forward-plan` creates chronological experiment plans only. It does not load labels, fit models, modify the champion, or publish predictions.

Open sessions come from the local `trade_cal` dataset. Evaluation periods are calendar months whose boundaries are mapped to open trading sessions. Expanding folds keep the first training session fixed; rolling folds retain a configured fixed number of trading sessions (`rolling_window_years * annual_sessions`). Validation is also measured in trading sessions.

Fold manifests are horizon-agnostic. They record actual purge and embargo session counts but do not claim that a particular label has matured. A downstream horizon plan must require at least `H+1` sessions for a next-open label with horizon `H`.

```bash
ashare-quant --config config/default.yaml models walk-forward-plan \
  --start-date 20100101 \
  --end-date 20260717 \
  --scheme expanding
```

The command uses the registered `lightgbm_ranker` champion for provenance by default. `--model-id` can name another registered immutable model. Outputs are written to `reports/walk_forward/<run_id>/folds.json` and `manifest.json`. Every fold records its date ranges, session gaps, model ID, and feature hash.
