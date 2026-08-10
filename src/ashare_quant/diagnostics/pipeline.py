"""End-to-end feature diagnostics with chronological leakage barriers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from ashare_quant.config.settings import AppSettings, DiagnosticSettings
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.diagnostics.data import DiagnosticDataLoader
from ashare_quant.diagnostics.metrics import (
    daily_ic_table,
    greedy_correlation_prune,
    pairwise_correlations,
    regime_ic_statistics,
    summarize_ic,
    yearly_ic_statistics,
)
from ashare_quant.diagnostics.model import (
    choose_feature_count,
    evaluate_feature_sets,
    evaluate_model,
    model_importance,
    permutation_importance,
    train_diagnostic_model,
)
from ashare_quant.features.registry import FEATURE_REGISTRY
from ashare_quant.models.research_policy import enforce_research_window, load_research_policy
from ashare_quant.utils.manifest import config_hash, current_git_info, read_manifest

type DataFrame = pd.DataFrame


@dataclass(frozen=True, slots=True)
class ChronologicalSplit:
    """Explicit non-overlapping train, validation, and one-time test periods."""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str

    def validate(self) -> None:
        """Reject overlapping or reversed research periods."""

        values = asdict(self)
        if any(len(value) != 8 or not value.isdigit() for value in values.values()):
            raise DataValidationError("all diagnostic split dates must use YYYYMMDD")
        if not (
            self.train_start
            <= self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.test_end
        ):
            raise DataValidationError(
                "diagnostic periods must be chronological and non-overlapping: "
                "train_end < validation_start and validation_end < test_start"
            )


@dataclass(frozen=True, slots=True)
class DiagnosticRunResult:
    """Summarize one completed diagnostics report."""

    report_dir: Path
    recommended_features: tuple[str, ...]
    recommended_count: int
    train_rows: int
    validation_rows: int
    test_rows: int


class FeatureDiagnosticPipeline:
    """Select features without exposing the final test period to selection."""

    def __init__(
        self,
        processed_root: Path,
        reports_root: Path,
        settings: AppSettings,
        config_path: Path,
        research_policy_path: Path = Path("config/research_policy.yaml"),
    ) -> None:
        self.processed_root = processed_root
        self.reports_root = reports_root
        self.settings = settings
        self.config_path = config_path
        self.research_policy_path = research_policy_path
        self.feature_specs = tuple(FEATURE_REGISTRY)
        self.feature_names = tuple(spec.name for spec in self.feature_specs)
        self.family_by_feature = {spec.name: spec.family for spec in self.feature_specs}

    def run(self, split: ChronologicalSplit, horizon: int | None = None) -> DiagnosticRunResult:
        """Run train-only selection, validation comparison, then frozen OOS evaluation."""

        split.validate()
        policy = load_research_policy(self.research_policy_path)
        enforce_research_window(
            policy,
            consumer="feature_selection",
            start_date=split.train_start,
            end_date=split.validation_end,
        )
        enforce_research_window(
            policy,
            consumer="walk_forward_evaluation",
            start_date=split.test_start,
            end_date=split.test_end,
        )
        diagnostic_settings = self.settings.diagnostics
        selected_horizon = horizon or diagnostic_settings.label_horizon
        if selected_horizon not in self.settings.labels.horizons:
            raise DataValidationError(
                f"diagnostic horizon {selected_horizon} is not configured in labels.horizons"
            )
        loader = DiagnosticDataLoader(self.processed_root, self.feature_names, selected_horizon)
        coverage_counts = {feature: 0 for feature in self.feature_names}
        train_rows = 0
        daily_tables: list[DataFrame] = []
        benchmark_tables: list[DataFrame] = []
        for chunk in loader.iter_period(split.train_start, split.train_end):
            if chunk.empty:
                continue
            train_rows += len(chunk)
            for feature in self.feature_names:
                values = pd.to_numeric(chunk[feature], errors="coerce")
                coverage_counts[feature] += int((values.notna() & np.isfinite(values)).sum())
            daily_tables.append(
                daily_ic_table(
                    chunk, self.feature_names, diagnostic_settings.minimum_daily_cross_section
                )
            )
            benchmark_tables.append(
                chunk.groupby("trade_date")["benchmark_forward_ret"].mean().reset_index()
            )
        if train_rows == 0:
            raise DataValidationError("training diagnostics contain no eligible labelled rows")
        coverage = {feature: count / train_rows for feature, count in coverage_counts.items()}
        daily_ic = concat_nonempty(daily_tables)
        summary = summarize_ic(daily_ic, coverage)
        yearly = yearly_ic_statistics(daily_ic)
        benchmark_daily = concat_nonempty(benchmark_tables).drop_duplicates("trade_date")
        regimes = regime_ic_statistics(
            daily_ic, benchmark_daily, diagnostic_settings.regime_return_threshold
        )
        summary = add_stability_and_score(summary, yearly)
        eligible_summary = summary[
            (summary["coverage"] >= diagnostic_settings.minimum_coverage)
            & (summary["ic_days"] >= diagnostic_settings.minimum_ic_days)
            & summary["rank_ic_mean"].notna()
        ].sort_values(["robust_score", "feature"], ascending=[False, True])
        if eligible_summary.empty:
            raise DataValidationError("no features pass training-period coverage and IC-day gates")

        correlation_sample = loader.load(
            split.train_start,
            split.train_end,
            max_rows=diagnostic_settings.correlation_sample_rows,
        )
        eligible_features = eligible_summary["feature"].astype(str).tolist()
        correlations = pairwise_correlations(correlation_sample, eligible_features)
        accepted, pruned = greedy_correlation_prune(
            eligible_summary["feature"].tolist(),
            correlations,
            diagnostic_settings.correlation_threshold,
        )
        if not accepted:
            raise DataValidationError("correlation pruning removed every eligible feature")

        train_sample = loader.load(
            split.train_start, split.train_end, max_rows=diagnostic_settings.model_sample_rows
        )
        validation_sample = loader.load(
            split.validation_start,
            split.validation_end,
            max_rows=diagnostic_settings.model_sample_rows,
        )
        if validation_sample.empty:
            raise DataValidationError("validation period contains no eligible labelled rows")
        base_model = train_diagnostic_model(train_sample, accepted, diagnostic_settings)
        importance = model_importance(base_model)
        permutation = permutation_importance(
            base_model, validation_sample, accepted, diagnostic_settings
        )
        ranked = rank_accepted_features(accepted, eligible_summary, importance)

        candidate_sets = candidate_feature_sets(
            ranked, diagnostic_settings.candidate_feature_counts
        )
        validation_counts = evaluate_feature_sets(
            train_sample, validation_sample, candidate_sets, diagnostic_settings
        )
        recommended_name = choose_feature_count(validation_counts)
        recommended = tuple(candidate_sets[recommended_name])

        family_ablation = build_family_ablation(
            train_sample,
            validation_sample,
            recommended,
            self.family_by_feature,
            diagnostic_settings,
        )
        family_incremental = build_incremental_family_tests(
            train_sample,
            validation_sample,
            recommended,
            self.family_by_feature,
            eligible_summary,
            diagnostic_settings,
        )

        # The recommendation is frozen before any test-period rows are loaded.
        test_sample = loader.load(
            split.test_start, split.test_end, max_rows=diagnostic_settings.model_sample_rows
        )
        if test_sample.empty:
            raise DataValidationError("test period contains no eligible labelled rows")
        final_training = pd.concat([train_sample, validation_sample], ignore_index=True)
        final_model = train_diagnostic_model(final_training, recommended, diagnostic_settings)
        test_metrics = evaluate_model(final_model, test_sample, recommended, diagnostic_settings)

        result = self._write_report(
            split=split,
            horizon=selected_horizon,
            train_rows=train_rows,
            validation_rows=len(validation_sample),
            test_rows=len(test_sample),
            summary=summary,
            daily_ic=daily_ic,
            yearly=yearly,
            regimes=regimes,
            correlations=correlations,
            pruned=pruned,
            importance=importance,
            permutation=permutation,
            validation_counts=validation_counts,
            family_ablation=family_ablation,
            family_incremental=family_incremental,
            recommended_name=recommended_name,
            recommended=recommended,
            test_metrics=test_metrics,
        )
        return result

    def _write_report(self, **payload: object) -> DiagnosticRunResult:
        """Atomically publish one immutable run directory and latest pointer."""

        split = payload["split"]
        assert isinstance(split, ChronologicalSplit)
        recommended = payload["recommended"]
        assert isinstance(recommended, tuple)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        root = self.reports_root / "feature_diagnostics"
        root.mkdir(parents=True, exist_ok=True)
        final_dir = root / run_id
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            directory = Path(temporary)
            table_names = (
                "summary",
                "daily_ic",
                "yearly",
                "regimes",
                "correlations",
                "pruned",
                "importance",
                "permutation",
                "validation_counts",
                "family_ablation",
                "family_incremental",
            )
            for name in table_names:
                table = payload[name]
                assert isinstance(table, pd.DataFrame)
                table.to_csv(directory / f"{name}.csv", index=False)
            selection = {
                "recommended_set": payload["recommended_name"],
                "recommended_feature_count": len(recommended),
                "recommended_features": list(recommended),
                "selection_uses_test_period": False,
                "test_metrics": payload["test_metrics"],
            }
            (directory / "recommended_features.json").write_text(
                json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8"
            )
            git_info = current_git_info()
            manifest = {
                "schema_version": 1,
                "artifact_name": "feature_diagnostics",
                "completed_at": datetime.now(UTC).isoformat(),
                "git_commit": git_info["commit"],
                "git_dirty": git_info["dirty"],
                "config_path": str(self.config_path),
                "config_hash": config_hash(self.config_path),
                "horizon": payload["horizon"],
                "split": asdict(split),
                "train_rows": payload["train_rows"],
                "validation_sample_rows": payload["validation_rows"],
                "test_sample_rows": payload["test_rows"],
                "selection_uses_test_period": False,
                "source_manifests": {
                    artifact: read_manifest(self.processed_root / artifact)
                    for artifact in ("features_daily", "labels_forward", "universe_daily")
                },
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            (directory / "report.md").write_text(
                render_markdown_report(payload, selection), encoding="utf-8"
            )
            directory.rename(final_dir)
        latest = root / "latest.json"
        temporary_latest = root / ".latest.json.tmp"
        temporary_latest.write_text(
            json.dumps({"run_id": run_id, "report_dir": str(final_dir)}, indent=2),
            encoding="utf-8",
        )
        temporary_latest.replace(latest)
        return DiagnosticRunResult(
            report_dir=final_dir,
            recommended_features=recommended,
            recommended_count=len(recommended),
            train_rows=cast(int, payload["train_rows"]),
            validation_rows=cast(int, payload["validation_rows"]),
            test_rows=cast(int, payload["test_rows"]),
        )


def concat_nonempty(frames: list[DataFrame]) -> DataFrame:
    """Concatenate non-empty frames without emitting pandas warnings."""

    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def add_stability_and_score(summary: DataFrame, yearly: DataFrame) -> DataFrame:
    """Add directional year stability and a train-only robust ordering score."""

    working = summary.copy()
    if yearly.empty:
        working["year_direction_stability"] = 0.0
    else:
        signs = (
            yearly.assign(positive=yearly["rank_ic_mean"] > 0).groupby("feature")["positive"].mean()
        )
        stability = signs.apply(lambda value: max(float(value), 1.0 - float(value)))
        working["year_direction_stability"] = working["feature"].map(stability).fillna(0.0)
    working["robust_score"] = (
        working["rank_ic_mean"].abs()
        * working["year_direction_stability"]
        * np.sqrt(working["coverage"].clip(lower=0.0))
    )
    return working


def rank_accepted_features(
    accepted: list[str], summary: DataFrame, importance: DataFrame
) -> list[str]:
    """Rank accepted features using training-only IC and model importance evidence."""

    merged = summary[summary["feature"].isin(accepted)].merge(importance, on="feature", how="left")
    merged["ic_rank"] = merged["robust_score"].rank(pct=True)
    merged["gain_rank"] = merged["gain_importance"].rank(pct=True)
    merged["split_rank"] = merged["split_importance"].rank(pct=True)
    merged["ordering_score"] = merged[["ic_rank", "gain_rank", "split_rank"]].mean(axis=1)
    return merged.sort_values(["ordering_score", "feature"], ascending=[False, True])[
        "feature"
    ].tolist()


def candidate_feature_sets(
    ranked: list[str], requested_counts: tuple[int, ...]
) -> dict[str, list[str]]:
    """Build unique top-N candidates plus all accepted features."""

    counts = sorted({count for count in requested_counts if count < len(ranked)})
    feature_sets = {f"top_{count}": ranked[:count] for count in counts}
    feature_sets["all_accepted"] = ranked
    return feature_sets


def build_family_ablation(
    train: DataFrame,
    validation: DataFrame,
    features: tuple[str, ...],
    family_by_feature: dict[str, str],
    settings: DiagnosticSettings,
) -> DataFrame:
    """Evaluate validation degradation when each selected family is removed."""

    families = sorted({family_by_feature[feature] for feature in features})
    sets: dict[str, list[str]] = {"all_selected": list(features)}
    for family in families:
        sets[f"without_{family}"] = [
            feature for feature in features if family_by_feature[feature] != family
        ]
    return evaluate_feature_sets(train, validation, sets, settings)


def build_incremental_family_tests(
    train: DataFrame,
    validation: DataFrame,
    features: tuple[str, ...],
    family_by_feature: dict[str, str],
    summary: DataFrame,
    settings: DiagnosticSettings,
) -> DataFrame:
    """Add families in train-only strength order and measure validation contribution."""

    scores = summary.set_index("feature")["robust_score"].to_dict()
    families = sorted(
        {family_by_feature[feature] for feature in features},
        key=lambda family: (
            -float(
                np.mean(
                    [
                        scores.get(feature, 0.0)
                        for feature in features
                        if family_by_feature[feature] == family
                    ]
                )
            ),
            family,
        ),
    )
    cumulative: list[str] = []
    sets: dict[str, list[str]] = {}
    for index, family in enumerate(families, start=1):
        cumulative.extend(feature for feature in features if family_by_feature[feature] == family)
        sets[f"step_{index:02d}_{family}"] = list(cumulative)
    return evaluate_feature_sets(train, validation, sets, settings)


def render_markdown_report(payload: dict[str, object], selection: dict[str, object]) -> str:
    """Render a compact human-readable evidence summary."""

    summary = payload["summary"]
    pruned = payload["pruned"]
    regimes = payload["regimes"]
    validation = payload["validation_counts"]
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(pruned, pd.DataFrame)
    assert isinstance(regimes, pd.DataFrame)
    assert isinstance(validation, pd.DataFrame)
    robust = summary.sort_values("robust_score", ascending=False).head(20)["feature"].tolist()
    unstable = summary.sort_values("year_direction_stability").head(20)["feature"].tolist()
    return "\n".join(
        [
            "# Feature Diagnostics Report",
            "",
            "Selection uses training-period coverage, IC stability, correlation pruning, and "
            "training-fitted LightGBM importance. Validation compares candidate sizes and family "
            "contributions. Test metrics are computed only after the feature set is frozen.",
            "",
            f"- Recommended set: `{selection['recommended_set']}`",
            f"- Recommended feature count: {selection['recommended_feature_count']}",
            f"- Training rows: {payload['train_rows']}",
            f"- Validation sample rows: {payload['validation_rows']}",
            f"- Test sample rows: {payload['test_rows']}",
            f"- Correlation-pruned features: {len(pruned)}",
            f"- Regime diagnostic rows: {len(regimes)}",
            "",
            "## Robust Features",
            "",
            ", ".join(robust) or "None",
            "",
            "## Least Stable Features",
            "",
            ", ".join(unstable) or "None",
            "",
            "## Candidate Count Evidence",
            "",
            "```text",
            validation.to_string(index=False) if not validation.empty else "No results.",
            "```",
            "",
            "## Frozen Test Metrics",
            "",
            "```json",
            json.dumps(selection["test_metrics"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
