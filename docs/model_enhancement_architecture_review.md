# LightGBM Ranker Model Enhancement Architecture Review

## Executive conclusion

The current evidence supports **alpha weakening and tail-ranking instability**, not complete alpha
disappearance. The model was trained on 2010-2019, validated on 2020-2022, and evaluated OOS from
2023 through 2026-07-09. Its 2026 Rank IC remains positive, but concentration into the highest-score
stocks and executable portfolio performance have deteriorated. The next objective is therefore to
test temporal robustness, not maximize one backtest.

Do not immediately replace the champion with a model trained through 2025. First compare expanding,
rolling, and walk-forward designs under identical purged chronological folds. The recommended
research baseline is a five-year rolling model retrained monthly, evaluated beside an expanding
model and a frozen 2010-2019 control.

## Evidence from Phase 2.6.1

| Period | Mean Rank IC | ICIR | Positive IC days | Top 10% 5d excess | Bottom 20% 5d excess |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2023 | 0.0353 | 0.512 | 73.1% | 0.470% | 0.081% |
| 2024 | 0.0852 | 0.660 | 74.4% | 0.710% | -0.735% |
| 2025 | 0.0768 | 1.017 | 88.5% | 0.746% | -0.116% |
| 2026 to July 9 | 0.0424 | 0.508 | 74.0% | -0.215% | -0.667% |

Label coverage is 99.59% over 3,762,653 predictions and 850 sessions. The broad 2026 ordering still
contains information: Top 10% outperforms Bottom 20% by about 0.45 percentage points per five-day
cohort. However, Top 1% and Top 5% underperform Top 20%, and the executable Top-10/20/50 portfolios
all underperform the benchmark in 2026. This separates three effects:

1. **Cross-sectional alpha weakened:** 2026 IC and ICIR are roughly half their 2025 values.
2. **Tail calibration failed:** larger scores do not reliably imply stronger realized returns.
3. **Ranking-to-portfolio translation failed:** a weak positive broad ordering is insufficient for
   concentrated Top-N execution after turnover and tradability effects.

The result is not yet proof of structural concept drift. The 2026 sample is partial, five-day labels
overlap, June IC was temporarily negative, and July recovered strongly. Formal attribution should
test feature PSI/quantile drift, missingness drift, score distribution drift, universe composition,
and feature-response drift. The model relies heavily on liquidity/amount variability, residual
volatility, and medium-term relative momentum; these relationships are plausible drift channels.

## Training designs

### A. Expanding training through 2025

Train on 2010-2025 and deploy from 2026. This maximizes regime coverage and sample size but gives old
regimes equal structural relevance and permits gradual obsolescence to dominate. It is a deployment
candidate, not a valid explanation by itself.

For an unbiased comparison, first run an inner experiment:

* train: 2010-2024;
* validation: 2025;
* locked test: 2026;
* purge: exclude training rows whose label exit date enters validation/test.

Only after the design is approved may the final expanding artifact train through 2025. Its 2026
result must never be fed back into feature, label, parameter, or ensemble-weight selection.

### B. Five-year rolling training with monthly retraining

For prediction month M, use only the preceding five years whose labels are fully realized before M.
Use the earliest four years for fitting and the most recent year for early stopping/model comparison,
or retain fixed parameters and use that year only as a quality gate. Apply a purge at least as long
as the largest model horizon.

Advantages are recency and adaptation to changing liquidity/universe structure. Risks are smaller
samples, unstable feature importance, model turnover, and excessive response to short regimes.
Record score correlation, feature-importance drift, and prediction turnover between consecutive
monthly models.

### C. Walk-forward evaluation

Walk-forward is the required comparison framework, not a third production model. Suggested folds:

```text
fit window -> purged validation window -> untouched evaluation window -> advance 6 or 12 months
```

Run both expanding and five-year rolling fits through the same evaluation months. All preprocessing,
model choices, horizon weights, and promotion rules are frozen from prior folds. Concatenate only
fold-level OOS predictions to evaluate stability. This is the only reliable way to distinguish a
stale model from a temporarily adverse regime.

## Label horizon review

All horizons retain the existing after-close signal and T+1 open entry semantics. Exit occurs at the
future executable open; suspended or limit-down exits follow the configured conservative rule.

| Label | Expected role | Main benefit | Main risk |
| --- | --- | --- | --- |
| 5d excess | fast alpha sleeve | responsive, many observations | noisy, high turnover, overlapping targets |
| 10d excess | short/medium sleeve | better signal-to-noise balance | slower reaction, still overlap-heavy |
| 20d excess | monthly sleeve | aligns with lower turnover | fewer independent samples, regime exposure |
| 60d excess | strategic sleeve | captures persistent trends/quality | very few independent outcomes, stale exits |

The current diagnostics support retaining 5d as a control, not as the sole target. Add 20d and 60d
labels only with the same executable-price and availability validation already applied to 5d/10d.
Compare horizons using identical signal dates and report effective independent sample size. A horizon
is accepted for stability, not because it has the highest single-period return.

## Ensemble ranker design

Train one independent Ranker per horizon. Never combine raw LightGBM scores because their scales are
not comparable. Convert each model's daily score to an eligible-universe percentile rank, then use
one of two controlled designs:

1. **Static rank blend:** equal weights as the mandatory baseline; optionally use clipped,
   validation-only ICIR weights that sum to one.
2. **Horizon sleeves:** allocate fixed research capital to separate 5d, 10d, 20d, and 60d sleeves,
   with each sleeve retaining its own holding period. This is preferred because execution horizon
   remains consistent with the target.

Do not introduce regime-dependent weights initially. Accept a blend only if it improves median and
worst-fold stability, lowers score concentration, and remains beneficial after turnover. Measure
cross-model daily score correlation; highly correlated horizons add no diversification.

```text
PIT features
  -> horizon-specific monthly models
  -> daily percentile ranks
  -> static blend and/or horizon sleeves
  -> immutable OOS predictions
  -> diagnostics and executable simulation
  -> promotion gate
```

## Validation contract

Every experiment must satisfy these gates:

* chronological train/validation/test isolation with horizon-aware purging;
* no feature selection, parameter choice, or ensemble weighting from the final test;
* daily Spearman Rank IC, IC standard deviation, ICIR, and positive-IC ratio;
* **disjoint** score deciles/quintiles and top-minus-bottom spread, because overlapping Top 1/5/10/20
  cohorts cannot establish strict monotonicity;
* yearly and rolling 20/60-session stability with confidence intervals from date-block bootstrap;
* bull, bear, and neutral results defined using contemporaneously observable benchmark trend and
  volatility, with regime labels used only for evaluation;
* breadth, missingness, turnover, score concentration, and feature/score drift;
* identical transaction and tradability assumptions for executable comparisons.

Promotion requires positive median fold Rank IC, no persistent negative recent fold, stable bucket
ordering, acceptable worst-year behavior, and improvement across several criteria. A single high
Sharpe, annual return, or 2026 recovery month is insufficient.

## Recommended implementation route

1. Extend diagnostics with disjoint deciles, horizon-aware block confidence intervals, feature PSI,
   score drift, and monthly feature-response drift.
2. Add executable 20d and 60d labels; rebuild only labels after tests pass. Keep 5d/10d unchanged as
   controls.
3. Implement a walk-forward runner that emits immutable fold predictions and manifests. Compare the
   frozen model, expanding model, and five-year rolling monthly model on the same folds.
4. Run single-horizon 5d/10d/20d/60d models with the existing 20-feature set and fixed baseline
   parameters. Do not tune features and model simultaneously.
5. Evaluate equal-weight rank blending and horizon sleeves. Consider validation-weighted blending
   only if equal weighting is robust.
6. Reserve the latest untouched period for one promotion decision, then shadow-run the challenger
   for at least several monthly retraining cycles before champion promotion.

The immediate recommendation is **walk-forward first, five-year rolling monthly as challenger, and
expanding training as control**. This directly tests whether recency restores stable ordering while
protecting against choosing a model solely because it fits 2025-2026.
