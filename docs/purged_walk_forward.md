# Purged Walk-forward Planning

`ashare-quant models walk-forward-plan` creates chronological experiment plans only. It does not load labels, fit models, modify the champion, or publish predictions.

Open sessions come from the local `trade_cal` dataset. Evaluation periods are calendar months whose boundaries are mapped to open trading sessions. Expanding folds keep the first training session fixed; rolling folds retain a configured fixed number of trading sessions (`rolling_window_years * annual_sessions`). Validation is also measured in trading sessions.

For a horizon `H`, the existing label enters at `T+1` and exits `H` sessions after entry. Its final information date is therefore `H+1` sessions after the signal. The default five-day plan uses six purge and six embargo sessions so training labels mature before validation and validation labels mature before evaluation.

```bash
ashare-quant --config config/default.yaml models walk-forward-plan \
  --start-date 20100101 \
  --end-date 20260717 \
  --scheme expanding
```

The command uses the registered `lightgbm_ranker` champion for provenance by default. `--model-id` can name another registered immutable model. Outputs are written to `reports/walk_forward/<run_id>/folds.json` and `manifest.json`. Every fold records its date ranges, session gaps, model ID, and feature hash.
