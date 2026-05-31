from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment import diagnostics_report_path, diagnostics_summary_path
from lib.analysis_utils import (
    series_is_binary,
    series_is_numeric,
    top_feature_drift,
)
from lib.utils import (
    utc_now,
    write_json,
    write_text,
)


def get_diagnostics_observations(experiment_dir: Path) -> list[tuple[int, str]]:
    """Extract priority-scored one-line observations from diagnostics/report.json."""
    report_path = diagnostics_report_path(experiment_dir)
    if not report_path.exists():
        return []
    try:
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    observations: list[tuple[int, str]] = []

    # Split drift (priority 85)
    for entry in report.get("split_comparison", []):
        for drift in entry.get("top_feature_drift", []):
            if drift.get("scaled_delta", 0) >= 1.0:
                observations.append(
                    (
                        85,
                        f"Split drift detected: feature `{drift['column']}` shifted by "
                        f"{drift['scaled_delta']:.2f} std between train and `{entry['split']}`.",
                    )
                )

    # High missingness (priority 70)
    for entry in report.get("missingness", []):
        if entry.get("missing_rate", 0) >= 0.1:
            observations.append(
                (
                    70,
                    f"High missingness: `{entry['column']}` missing in "
                    f"{entry['missing_rate']:.0%} of rows.",
                )
            )

    # Subgroup variance (priority 60)
    for entry in report.get("subgroup_slices", []):
        if abs(entry.get("gap", 0)) >= 0.05:
            observations.append(
                (
                    60,
                    f"Subgroup variance: `{entry['feature']}` shows target gap of "
                    f"{entry['gap']:.3f} between `{entry['low_group']}` and `{entry['high_group']}`.",
                )
            )

    # Interaction candidates (priority 50)
    for entry in report.get("interaction_candidates", []):
        if entry.get("lift", 0) > 0:
            observations.append(
                (
                    50,
                    f"Interaction candidate: `{entry['feature_a']}` x `{entry['feature_b']}` "
                    f"(correlation lift {entry['lift']:+.3f}).",
                )
            )

    observations.sort(key=lambda item: item[0], reverse=True)
    return observations


def generate_experiment_diagnostics(
    experiment_dir: Path,
    frames_by_split: dict[str, pd.DataFrame],
    *,
    target_column: str | None = None,
    prediction_column: str | None = None,
) -> dict[str, Any]:
    """Generate lightweight experiment-local diagnostics artifacts.

    The helper is intentionally simple: callers pass split DataFrames and optional
    target / prediction columns. Diagnostics are written to the experiment's
    ``diagnostics/`` directory without touching ``results.json``.
    """
    frames = {
        name: frame.copy()
        for name, frame in frames_by_split.items()
        if frame is not None and not frame.empty
    }
    if not frames:
        raise ValueError("Diagnostics require at least one non-empty split frame.")

    report = build_diagnostics_report(
        frames,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    write_json(diagnostics_report_path(experiment_dir), report)
    write_text(diagnostics_summary_path(experiment_dir), render_diagnostics_summary(report))
    return report


def build_diagnostics_report(
    frames_by_split: dict[str, pd.DataFrame],
    *,
    target_column: str | None = None,
    prediction_column: str | None = None,
) -> dict[str, Any]:
    frames = {
        name: frame.copy()
        for name, frame in frames_by_split.items()
        if frame is not None and not frame.empty
    }
    if not frames:
        raise ValueError("Diagnostics require at least one non-empty split frame.")

    combined = pd.concat(frames.values(), ignore_index=True, sort=False)
    analysis_split = _analysis_split_name(frames)
    analysis_frame = frames[analysis_split]

    return {
        "generated_at": utc_now(),
        "analysis_split": analysis_split,
        "target_column": target_column,
        "prediction_column": prediction_column,
        "problem_type": _infer_problem_type(analysis_frame, target_column),
        "splits": _split_summaries(frames, target_column),
        "missingness": _missingness_summary(combined),
        "split_comparison": _split_comparison_summary(frames, target_column, prediction_column),
        "subgroup_slices": _subgroup_slice_summary(
            analysis_frame, target_column, prediction_column
        ),
        "interaction_candidates": _interaction_candidates_summary(
            frames.get("train", analysis_frame), target_column, prediction_column
        ),
        "error_patterns": _error_pattern_summary(
            analysis_frame,
            target_column=target_column,
            prediction_column=prediction_column,
        ),
    }


def render_diagnostics_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Updated: `{report['generated_at']}`",
        "Advisory only. Recompute after major data, feature, or prediction changes.",
        "",
        "### Data Profile",
    ]
    lines.extend(_render_split_overview(report["splits"]))

    missingness = report["missingness"]
    if missingness:
        lines.extend(["", "### Missingness"])
        for entry in missingness[:3]:
            lines.append(f"- `{entry['column']}` missing in {entry['missing_rate']:.1%} of rows")

    split_comparison = report["split_comparison"]
    if split_comparison:
        lines.extend(["", "### Split Comparison"])
        for entry in split_comparison:
            if entry.get("target_shift") is not None:
                lines.append(
                    f"- `{entry['split']}` target shift vs train: {entry['target_shift']:+.3f}"
                )
            for drift in entry.get("top_feature_drift", [])[:2]:
                lines.append(
                    f"- `{entry['split']}` drift: `{drift['column']}` "
                    f"(delta={drift['delta']:+.3f}, scaled={drift['scaled_delta']:.2f})"
                )

    subgroup_slices = report["subgroup_slices"]
    if subgroup_slices:
        lines.extend(["", "### Subgroup Slices"])
        for entry in subgroup_slices[:3]:
            lines.append(
                f"- `{entry['feature']}`: {entry['low_group']} -> {entry['low_value']:.3f}, "
                f"{entry['high_group']} -> {entry['high_value']:.3f}"
            )

    interaction_candidates = report["interaction_candidates"]
    if interaction_candidates:
        lines.extend(["", "### Interaction Candidates"])
        for entry in interaction_candidates[:3]:
            lines.append(
                f"- `{entry['feature_a']} × {entry['feature_b']}` (corr lift {entry['lift']:+.3f})"
            )

    error_patterns = report["error_patterns"]
    if error_patterns:
        lines.extend(["", "### Error Patterns"])
        for entry in error_patterns[:3]:
            lines.append(f"- {entry['summary']}")

    return "\n".join(lines) + "\n"


def _render_split_overview(splits: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    total_rows = sum(entry["rows"] for entry in splits)
    lines.append(f"- {total_rows} rows across {len(splits)} split(s)")
    for entry in splits:
        line = (
            f"- `{entry['split']}`: {entry['rows']} rows, {entry['columns']} columns, "
            f"{entry['missing_cells']} missing cells"
        )
        target_summary = entry.get("target_summary")
        if target_summary:
            if target_summary["kind"] == "classification":
                line += f", positive rate={target_summary['positive_rate']:.3f}"
            else:
                line += (
                    f", target mean={target_summary['mean']:.3f}, std={target_summary['std']:.3f}"
                )
        lines.append(line)
    return lines


def _split_summaries(
    frames_by_split: dict[str, pd.DataFrame], target_column: str | None
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for split_name, frame in frames_by_split.items():
        entry: dict[str, Any] = {
            "split": split_name,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "missing_cells": int(frame.isna().sum().sum()),
        }
        target_summary = _target_summary(frame, target_column)
        if target_summary is not None:
            entry["target_summary"] = target_summary
        summaries.append(entry)
    return summaries


def _target_summary(frame: pd.DataFrame, target_column: str | None) -> dict[str, Any] | None:
    if not target_column or target_column not in frame:
        return None
    target = frame[target_column].dropna()
    if target.empty:
        return None
    if series_is_binary(target):
        return {
            "kind": "classification",
            "positive_rate": float(target.astype(float).mean()),
        }
    target_numeric = pd.to_numeric(target, errors="coerce").dropna()
    if target_numeric.empty:
        return None
    return {
        "kind": "regression",
        "mean": float(target_numeric.mean()),
        "std": float(target_numeric.std(ddof=0)),
    }


def _missingness_summary(combined: pd.DataFrame) -> list[dict[str, Any]]:
    missing_rates = combined.isna().mean()
    entries = [
        {
            "column": column,
            "missing_rate": float(rate),
        }
        for column, rate in missing_rates.sort_values(ascending=False).items()
        if rate > 0
    ]
    return entries[:5]


def _split_comparison_summary(
    frames_by_split: dict[str, pd.DataFrame],
    target_column: str | None,
    prediction_column: str | None,
) -> list[dict[str, Any]]:
    if "train" not in frames_by_split:
        return []
    train = frames_by_split["train"]
    comparisons: list[dict[str, Any]] = []
    for split_name, frame in frames_by_split.items():
        if split_name == "train":
            continue
        comparison: dict[str, Any] = {"split": split_name, "top_feature_drift": []}
        target_shift = _target_shift(train, frame, target_column)
        if target_shift is not None:
            comparison["target_shift"] = target_shift
        comparison["top_feature_drift"] = top_feature_drift(
            train,
            frame,
            excluded_columns={target_column, prediction_column},
        )
        if comparison.get("target_shift") is not None or comparison["top_feature_drift"]:
            comparisons.append(comparison)
    return comparisons


def _target_shift(
    train: pd.DataFrame, other: pd.DataFrame, target_column: str | None
) -> float | None:
    if not target_column or target_column not in train or target_column not in other:
        return None
    train_target = pd.to_numeric(train[target_column], errors="coerce").dropna()
    other_target = pd.to_numeric(other[target_column], errors="coerce").dropna()
    if train_target.empty or other_target.empty:
        return None
    return float(other_target.mean() - train_target.mean())


def _subgroup_slice_summary(
    frame: pd.DataFrame,
    target_column: str | None,
    prediction_column: str | None,
) -> list[dict[str, Any]]:
    if not target_column or target_column not in frame:
        return []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.dropna().empty:
        return []

    candidates: list[dict[str, Any]] = []
    categorical_columns = [
        column
        for column in frame.columns
        if column not in {target_column, prediction_column}
        and (
            pd.api.types.is_bool_dtype(frame[column])
            or pd.api.types.is_object_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        )
    ]
    for column in categorical_columns[:6]:
        grouped = (
            frame[[column, target_column]]
            .dropna()
            .groupby(column, dropna=True)[target_column]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = grouped[grouped["count"] >= 5]
        if len(grouped) < 2:
            continue
        grouped = grouped.sort_values("mean")
        low = grouped.iloc[0]
        high = grouped.iloc[-1]
        candidates.append(
            {
                "feature": column,
                "low_group": str(low[column]),
                "low_value": float(low["mean"]),
                "high_group": str(high[column]),
                "high_value": float(high["mean"]),
                "gap": float(high["mean"] - low["mean"]),
            }
        )

    numeric_ranked = _rank_numeric_features_against_target(
        frame,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    for column, _score in numeric_ranked[:2]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        valid = pd.DataFrame({"feature": numeric, "target": target}).dropna()
        if len(valid) < 12:
            continue
        try:
            valid["bucket"] = pd.qcut(valid["feature"], q=4, duplicates="drop")
        except ValueError:
            continue
        grouped = (
            valid.groupby("bucket", observed=False)["target"].agg(["mean", "count"]).reset_index()
        )
        grouped = grouped[grouped["count"] >= 5]
        if len(grouped) < 2:
            continue
        grouped = grouped.sort_values("mean")
        low = grouped.iloc[0]
        high = grouped.iloc[-1]
        candidates.append(
            {
                "feature": column,
                "low_group": str(low["bucket"]),
                "low_value": float(low["mean"]),
                "high_group": str(high["bucket"]),
                "high_value": float(high["mean"]),
                "gap": float(high["mean"] - low["mean"]),
            }
        )

    candidates.sort(key=lambda item: abs(item["gap"]), reverse=True)
    return candidates[:3]


def _interaction_candidates_summary(
    frame: pd.DataFrame,
    target_column: str | None,
    prediction_column: str | None,
) -> list[dict[str, Any]]:
    if not target_column or target_column not in frame:
        return []
    ranked = _rank_numeric_features_against_target(
        frame,
        target_column=target_column,
        prediction_column=prediction_column,
    )
    if len(ranked) < 2:
        return []
    base_scores = dict(ranked)
    candidates: list[dict[str, Any]] = []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    for feature_a, feature_b in combinations([name for name, _ in ranked[:6]], 2):
        valid = pd.DataFrame(
            {
                "a": pd.to_numeric(frame[feature_a], errors="coerce"),
                "b": pd.to_numeric(frame[feature_b], errors="coerce"),
                "target": target,
            }
        ).dropna()
        if len(valid) < 30:
            continue
        interaction = valid["a"] * valid["b"]
        if interaction.nunique() < 2:
            continue
        interaction_corr = abs(float(pd.Series(interaction).corr(valid["target"])))
        lift = interaction_corr - max(base_scores[feature_a], base_scores[feature_b])
        if not np.isfinite(lift):
            continue
        candidates.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "interaction_corr": interaction_corr,
                "lift": float(lift),
                "exploratory": len(valid) < 100,
            }
        )
    candidates.sort(key=lambda item: item["lift"], reverse=True)
    return [candidate for candidate in candidates if candidate["lift"] > 0][:3]


def _error_pattern_summary(
    frame: pd.DataFrame,
    *,
    target_column: str | None,
    prediction_column: str | None,
) -> list[dict[str, Any]]:
    if (
        not target_column
        or not prediction_column
        or target_column not in frame
        or prediction_column not in frame
    ):
        return []
    target = pd.to_numeric(frame[target_column], errors="coerce")
    prediction = pd.to_numeric(frame[prediction_column], errors="coerce")
    valid = frame.copy()
    valid[target_column] = target
    valid[prediction_column] = prediction
    valid = valid.dropna(subset=[target_column, prediction_column])
    if len(valid) < 12:
        return []

    if series_is_binary(valid[target_column]):
        return _classification_error_patterns(valid, target_column, prediction_column)
    return _regression_error_patterns(valid, target_column, prediction_column)


def _classification_error_patterns(
    frame: pd.DataFrame, target_column: str, prediction_column: str
) -> list[dict[str, Any]]:
    probabilities = frame[prediction_column].clip(0.0, 1.0)
    actual = frame[target_column].astype(int)
    predicted = (probabilities >= 0.5).astype(int)
    summaries: list[dict[str, Any]] = []

    for label, error_mask, reference_mask in [
        (
            "false negatives",
            (actual == 1) & (predicted == 0),
            (actual == 1) & (predicted == 1),
        ),
        (
            "false positives",
            (actual == 0) & (predicted == 1),
            (actual == 0) & (predicted == 0),
        ),
    ]:
        if error_mask.sum() < 3 or reference_mask.sum() < 3:
            continue
        shift = _largest_numeric_shift(
            frame,
            error_mask,
            reference_mask,
            excluded_columns={target_column, prediction_column},
        )
        if shift is None:
            summaries.append(
                {
                    "summary": f"{label.title()} present in `{prediction_column}` on the analysis split."
                }
            )
            continue
        direction = "higher" if shift["delta"] > 0 else "lower"
        summaries.append(
            {
                "summary": (
                    f"{label.title()} cluster around {direction} `{shift['column']}` "
                    f"than the comparable correctly classified cases."
                )
            }
        )
    return summaries


def _regression_error_patterns(
    frame: pd.DataFrame, target_column: str, prediction_column: str
) -> list[dict[str, Any]]:
    residual = frame[target_column] - frame[prediction_column]
    hard_mask = residual.abs() >= residual.abs().quantile(0.75)
    rest_mask = ~hard_mask
    if hard_mask.sum() < 4 or rest_mask.sum() < 4:
        return []
    shift = _largest_numeric_shift(
        frame,
        hard_mask,
        rest_mask,
        excluded_columns={target_column, prediction_column},
    )
    if shift is None:
        return [{"summary": "High-residual cases are present on the analysis split."}]
    direction = "higher" if shift["delta"] > 0 else "lower"
    return [
        {
            "summary": (
                f"High-residual cases cluster around {direction} `{shift['column']}` "
                "than the rest of the analysis split."
            )
        }
    ]


def _largest_numeric_shift(
    frame: pd.DataFrame,
    mask_a: pd.Series,
    mask_b: pd.Series,
    *,
    excluded_columns: set[str | None],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for column in frame.columns:
        if column in excluded_columns or not series_is_numeric(frame[column]):
            continue
        a = pd.to_numeric(frame.loc[mask_a, column], errors="coerce").dropna()
        b = pd.to_numeric(frame.loc[mask_b, column], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        scale = float(b.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = max(abs(float(a.mean())), abs(float(b.mean())), 1e-6)
        delta = float(a.mean() - b.mean())
        candidates.append(
            {
                "column": column,
                "delta": delta,
                "scaled_delta": float(abs(delta) / scale),
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["scaled_delta"], reverse=True)
    return candidates[0]


def _rank_numeric_features_against_target(
    frame: pd.DataFrame,
    *,
    target_column: str,
    prediction_column: str | None,
) -> list[tuple[str, float]]:
    target = pd.to_numeric(frame[target_column], errors="coerce")
    ranked: list[tuple[str, float]] = []
    for column in frame.columns:
        if column in {target_column, prediction_column}:
            continue
        if not series_is_numeric(frame[column]):
            continue
        valid = pd.DataFrame(
            {
                "feature": pd.to_numeric(frame[column], errors="coerce"),
                "target": target,
            }
        ).dropna()
        if len(valid) < 12 or valid["feature"].nunique() < 2:
            continue
        corr = abs(float(valid["feature"].corr(valid["target"])))
        if np.isfinite(corr):
            ranked.append((column, corr))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _infer_problem_type(frame: pd.DataFrame, target_column: str | None) -> str | None:
    if not target_column or target_column not in frame:
        return None
    target = frame[target_column].dropna()
    if target.empty:
        return None
    return "classification" if series_is_binary(target) else "regression"


def _analysis_split_name(frames_by_split: dict[str, pd.DataFrame]) -> str:
    for preferred in ("validation", "train", "test"):
        if preferred in frames_by_split:
            return preferred
    return next(iter(frames_by_split))
