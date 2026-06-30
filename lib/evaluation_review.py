from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment import evaluation_review_path, evaluation_review_report_path
from lib.analysis import ranked_results, series_is_binary, top_feature_drift
from lib.io import (
    utc_now,
    write_json,
    write_text,
)

logger = logging.getLogger(__name__)


_CONCERN_PRIORITY: dict[str, int] = {
    "leakage": 95,
    "split": 90,
    "instability": 80,
    "metric": 75,
}

# Columns that are present in both train and eval frames but are NOT features —
# excluding them prevents auxiliary bookkeeping columns from making every row
# appear unique to the leakage detector.
_LEAKAGE_AUXILIARY_COLUMNS: frozenset[str] = frozenset({"split", "row_id", "account_id", "id"})


def get_evaluation_observations(experiment_dir: Path) -> list[tuple[int, str]]:
    """Extract priority-scored one-line observations from evaluation_review.json."""
    report_path = evaluation_review_report_path(experiment_dir)
    if not report_path.exists():
        return []
    try:
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    observations: list[tuple[int, str]] = []
    for concern in report.get("concerns", []):
        kind = concern.get("kind", "")
        priority = _CONCERN_PRIORITY.get(kind, 60)
        title = concern.get("title", "Evaluation concern")
        text = concern.get("concern", "")
        if text:
            observations.append((priority, f"{title}: {text}"))

    observations.sort(key=lambda item: item[0], reverse=True)
    return observations


def generate_evaluation_review(
    experiment_dir: Path,
    frames_by_split: dict[str, pd.DataFrame],
    *,
    target_column: str | None = None,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate a lightweight evaluation review artifact for an experiment."""
    frames = {
        name: frame.copy()
        for name, frame in frames_by_split.items()
        if frame is not None and not frame.empty
    }
    if not frames:
        raise ValueError("Evaluation review requires at least one non-empty split frame.")

    report = evaluation_review_report(
        frames,
        target_column=target_column,
        results=results or [],
    )
    write_json(evaluation_review_report_path(experiment_dir), report)
    write_text(evaluation_review_path(experiment_dir), render_evaluation_review(report))
    return report


def evaluation_review_report(
    frames_by_split: dict[str, pd.DataFrame],
    *,
    target_column: str | None = None,
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    concerns: list[dict[str, Any]] = []

    split_concern = _split_concern(frames_by_split, target_column)
    if split_concern:
        concerns.append(split_concern)

    leakage_concern = _leakage_concern(frames_by_split, target_column)
    if leakage_concern:
        concerns.append(leakage_concern)

    instability_concern = _instability_concern(results or [])
    if instability_concern:
        concerns.append(instability_concern)

    metric_concern = _metric_mismatch_concern(results or [])
    if metric_concern:
        concerns.append(metric_concern)

    concerns.sort(key=lambda item: item["priority"], reverse=True)
    return {
        "generated_at": utc_now(),
        "target_column": target_column,
        "concerns": concerns,
        "overall_assessment": (
            "No significant evaluation concerns detected from the current evidence."
            if not concerns
            else f"{len(concerns)} evaluation concern(s) detected."
        ),
    }


def render_evaluation_review(report: dict[str, Any]) -> str:
    lines = [
        f"Updated: `{report['generated_at']}`",
        "Advisory only. `experiment.md` remains unchanged until a human approves any amendment.",
        "",
        "## Overall Assessment",
        "",
        f"- {report['overall_assessment']}",
    ]

    concerns = report.get("concerns", [])
    if not concerns:
        lines.extend(
            [
                "",
                "## Review",
                "",
                "- No recommended spec amendments right now.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.extend(["", "## Concerns"])
    for concern in concerns:
        lines.extend(
            [
                "",
                f"### {concern['title']}",
                "",
                f"- **Concern:** {concern['concern']}",
                f"- **Recommendation:** {concern['recommendation']}",
                "- **Evidence:**",
            ]
        )
        lines.extend([f"  - {item}" for item in concern["evidence"]])

    return "\n".join(lines) + "\n"


def _split_concern(
    frames_by_split: dict[str, pd.DataFrame], target_column: str | None
) -> dict[str, Any] | None:
    if "train" not in frames_by_split or not target_column:
        return None
    train = frames_by_split["train"]
    if target_column not in train:
        return None

    evidence: list[str] = []
    severity = 0.0
    train_target = pd.to_numeric(train[target_column], errors="coerce").dropna()
    if train_target.empty:
        return None
    is_binary_target = series_is_binary(train_target)

    for split_name in ("validation", "test"):
        frame = frames_by_split.get(split_name)
        if frame is None or target_column not in frame:
            continue
        split_target = pd.to_numeric(frame[target_column], errors="coerce").dropna()
        if split_target.empty:
            continue
        for item in (
            _target_shift_evidence(
                split_name,
                train_target,
                split_target,
                is_binary_target=is_binary_target,
            ),
            _feature_drift_evidence(split_name, train, frame, target_column),
        ):
            if item is None:
                continue
            item_severity, item_evidence = item
            severity = max(severity, item_severity)
            evidence.append(item_evidence)

    if not evidence:
        return None
    return {
        "title": "Split Reliability Concern",
        "concern": "The current split may be unstable or mismatched to the problem shape.",
        "recommendation": (
            "Review the split policy before trusting small leaderboard differences; "
            "consider a more stable temporal cutoff or repeated split evaluation."
        ),
        "evidence": evidence,
        "priority": 90 + severity,
        "kind": "split",
    }


def _target_shift_evidence(
    split_name: str,
    train_target: pd.Series,
    split_target: pd.Series,
    *,
    is_binary_target: bool,
) -> tuple[float, str] | None:
    shift = float(split_target.mean() - train_target.mean())
    if is_binary_target:
        scaled = abs(shift)
        if scaled < 0.05:
            return None
        return scaled, f"`{split_name}` target rate differs from train by {shift:+.3f}."

    scale = _target_shift_scale(train_target, split_target)
    scaled = abs(shift) / scale
    if scaled < 0.5:
        return None
    return (
        scaled,
        f"`{split_name}` target mean differs from train by {shift:+.3f} ({scaled:.2f} train std).",
    )


def _target_shift_scale(train_target: pd.Series, split_target: pd.Series) -> float:
    scale = float(train_target.std(ddof=0))
    if np.isfinite(scale) and scale != 0:
        return scale
    return max(
        abs(float(train_target.mean())),
        abs(float(split_target.mean())),
        1e-6,
    )


def _feature_drift_evidence(
    split_name: str,
    train: pd.DataFrame,
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[float, str] | None:
    drift = top_feature_drift(train, frame, excluded_columns={target_column})
    if not drift or drift[0]["scaled_delta"] < 1.0:
        return None
    top = drift[0]
    return (
        top["scaled_delta"],
        f"`{split_name}` shows strong feature drift on `{top['column']}` "
        f"(scaled delta {top['scaled_delta']:.2f}).",
    )


def _leakage_concern(
    frames_by_split: dict[str, pd.DataFrame], target_column: str | None
) -> dict[str, Any] | None:
    if "train" not in frames_by_split:
        return None
    train = frames_by_split["train"]
    evidence: list[str] = []
    severity = 0
    excluded = {target_column} | _LEAKAGE_AUXILIARY_COLUMNS

    for split_name in ("validation", "test"):
        frame = frames_by_split.get(split_name)
        if frame is None:
            continue
        overlap = _row_overlap_count(train, frame, excluded_columns=excluded)
        if overlap <= 0:
            continue
        severity = max(severity, overlap)
        evidence.append(
            f"`train` and `{split_name}` share {overlap} exact row fingerprints "
            "after excluding the target and auxiliary columns."
        )

    if not evidence:
        return None
    return {
        "title": "Leakage Marker Concern",
        "concern": "Exact row overlap across splits suggests a possible leakage or deduplication issue.",
        "recommendation": (
            "Audit split boundaries for duplicated entities or repeated snapshots before "
            "treating the validation result as reliable."
        ),
        "evidence": evidence,
        "priority": 95 + severity,
        "kind": "leakage",
    }


def _instability_concern(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = ranked_results(results)
    if len(ranked) < 2:
        return None

    best = ranked[0]
    runner_up = ranked[1]
    best_width = _uncertainty_width(best)
    runner_up_width = _uncertainty_width(runner_up)
    evidence: list[str] = []
    severity = 0.0

    if best_width is not None and best_width >= 0.05:
        severity = max(severity, best_width)
        evidence.append(
            f"`{best.get('candidate_id', '?')}` has a wide uncertainty interval "
            f"(width {best_width:.3f})."
        )

    if best_width is not None:
        gap = float(best["objective_score"] - runner_up["objective_score"])
        threshold = max(best_width, runner_up_width or 0.0, 0.01)
        if gap <= threshold:
            severity = max(severity, threshold)
            evidence.append(
                f"The gap between `{best.get('candidate_id', '?')}` and "
                f"`{runner_up.get('candidate_id', '?')}` is {gap:.3f}, which is "
                "not clearly larger than the reported uncertainty."
            )

    if not evidence:
        return None
    return {
        "title": "Leaderboard Instability Concern",
        "concern": "The current validation leaderboard may be too noisy to support confident winner selection.",
        "recommendation": (
            "Treat the current ranking as provisional and add bootstrap or repeated-split "
            "evidence before locking in a winner."
        ),
        "evidence": evidence,
        "priority": 80 + severity,
        "kind": "instability",
    }


def _metric_mismatch_concern(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = ranked_results(results)
    if not ranked:
        return None
    best = ranked[0]
    objective_metric = str(best.get("objective_metric", ""))
    metrics = best.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    validation = metrics.get("validation", {})
    if not isinstance(validation, dict):
        return None

    evidence: list[str] = []
    severity = 0.0
    if "auc" in objective_metric:
        precision = validation.get("precision")
        recall = validation.get("recall")
        avg_precision = validation.get("avg_precision")
        if (
            isinstance(precision, (int, float))
            and isinstance(recall, (int, float))
            and (precision == 0.0 or recall == 0.0)
        ):
            severity = max(severity, 1.0)
            evidence.append(
                f"`{best.get('candidate_id', '?')}` optimizes `{objective_metric}` "
                f"but has validation precision={precision:.3f} and recall={recall:.3f}."
            )
        if isinstance(avg_precision, (int, float)) and avg_precision < 0.1:
            severity = max(severity, 0.5)
            evidence.append(
                f"`{best.get('candidate_id', '?')}` has low validation avg_precision={avg_precision:.3f} "
                f"despite optimizing `{objective_metric}`."
            )

    if not evidence:
        return None
    return {
        "title": "Metric Fit Concern",
        "concern": "The ranking metric may not fully reflect the deployment decision quality.",
        "recommendation": (
            "Keep the ranking metric for comparability, but add an operating metric "
            "or threshold-oriented review criterion alongside it."
        ),
        "evidence": evidence,
        "priority": 70 + severity,
        "kind": "metric",
    }


def _uncertainty_width(result: dict[str, Any]) -> float | None:
    hyperparameters = result.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        return None
    objective_metric = result.get("objective_metric")
    candidate_keys: list[str] = []
    if isinstance(objective_metric, str) and objective_metric:
        candidate_keys.append(f"{objective_metric}_ci_95")
    candidate_keys.extend(
        key
        for key, value in hyperparameters.items()
        if key.endswith("_ci_95") and isinstance(value, list) and len(value) == 2
    )
    for key in candidate_keys:
        bounds = hyperparameters.get(key)
        if (
            isinstance(bounds, list)
            and len(bounds) == 2
            and all(isinstance(v, (int, float)) for v in bounds)
        ):
            width = float(bounds[1] - bounds[0])
            if width > 0:
                return width
    return None


def _row_overlap_count(
    train: pd.DataFrame, other: pd.DataFrame, *, excluded_columns: set[str | None]
) -> int:
    # Guard: if excluded_columns contains no real feature column (e.g. because
    # target_column was None), fingerprinting would include everything and is
    # unreliable.  Return 0 as a safe no-op.
    real_exclusions = {
        c for c in excluded_columns if c is not None and c not in _LEAKAGE_AUXILIARY_COLUMNS
    }
    if not real_exclusions:
        logger.warning(
            "leakage-check skipped: excluded_columns contains no real target column "
            "(got %r); this is a no-op guard — callers should provide a target column",
            excluded_columns,
        )
        return 0
    shared_columns = [
        column
        for column in train.columns.intersection(other.columns)
        if column not in excluded_columns
    ]
    if not shared_columns:
        return 0
    logger.debug(
        "leakage-check fingerprint columns: %s",
        ", ".join(sorted(map(str, shared_columns))),
    )
    train_fingerprints = _fingerprints(train[shared_columns])
    other_fingerprints = _fingerprints(other[shared_columns])
    return len(train_fingerprints & other_fingerprints)


def _fingerprints(frame: pd.DataFrame) -> set[str]:
    normalized = frame.fillna("<NA>").astype(str)
    return set(normalized.agg("||".join, axis=1).tolist())
