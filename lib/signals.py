from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from experiment import (
    count_journal_cycles,
    journal_path,
    read_diagnostics_summary,
    research_sources_path,
    results_file,
)
from lib.analysis import ranked_results
from lib.diagnostics import get_diagnostics_observations
from lib.evaluation_review import get_evaluation_observations
from lib.io import load_json
from lib.observations import merge_observations

DEFAULT_SIGNAL_LIMIT = 3
ADVISORY_SIGNAL_CAP = 8

_ERROR_ANALYSIS_TERMS = [
    "error analysis",
    "false negative",
    "false positive",
    "misclassif",
    "residual",
    "worst prediction",
    "confusion matrix",
    "error pattern",
    "failure mode",
    "error cluster",
]


@dataclass(frozen=True)
class SignalOptions:
    results: list[dict[str, Any]] | None = None
    journal_cycles: int | None = None
    external_sources: int | None = None
    has_error_analysis: bool | None = None
    has_diagnostics: bool | None = None
    limit: int = DEFAULT_SIGNAL_LIMIT


@dataclass(frozen=True)
class SignalInputs:
    results: list[dict[str, Any]]
    journal_cycles: int
    external_sources: int
    has_error_analysis: bool
    has_diagnostics: bool


def results_snapshot(experiment_dir: Path) -> list[dict[str, Any]]:
    """Load ``results.json`` as a list of result dicts; return empty on missing/invalid."""
    rf = results_file(experiment_dir)
    try:
        payload = load_json(rf)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _count_external_sources(experiment_dir: Path) -> int:
    """Count ### Source NNN: headings in research_sources.md with real content.

    The scaffold pre-populates ``### Source 001: <title>`` as a placeholder.
    We detect the untouched placeholder by checking for the literal ``<title>``
    marker in the heading and exclude only that one.
    """
    path = research_sources_path(experiment_dir)
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8")
    all_headings = re.findall(r"^### Source \d{3}:.*$", text, re.MULTILINE)
    return sum(1 for h in all_headings if "<title>" not in h)


def journal_mentions_error_analysis(experiment_dir: Path) -> bool:
    """Check if the research journal contains evidence of error analysis work."""
    path = journal_path(experiment_dir)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    return any(term in text for term in _ERROR_ANALYSIS_TERMS)


def _objective_metric_name(result: dict[str, Any]) -> str | None:
    objective_metric = result.get("objective_metric")
    if not isinstance(objective_metric, str) or not objective_metric:
        return None
    if objective_metric.startswith("val_"):
        return objective_metric[4:]
    if objective_metric.startswith("validation_"):
        return objective_metric[len("validation_") :]
    return _shared_train_validation_metric(result, objective_metric)


def _shared_train_validation_metric(result: dict[str, Any], objective_metric: str) -> str | None:
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        return None
    train_metrics = metrics.get("train", {})
    validation_metrics = metrics.get("validation", {})
    if not isinstance(train_metrics, dict) or not isinstance(validation_metrics, dict):
        return None
    if objective_metric in train_metrics and objective_metric in validation_metrics:
        return objective_metric
    common = [name for name in train_metrics if name in validation_metrics]
    return common[0] if common else None


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
        for key in hyperparameters
        if key.endswith("_ci_95")
        and isinstance(hyperparameters.get(key), list)
        and len(hyperparameters.get(key)) == 2
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


def _plateau_signal(results: list[dict[str, Any]]) -> str | None:
    ranked = ranked_results(results)[:3]
    if len(ranked) < 2:
        return None
    widths = [_uncertainty_width(result) for result in ranked[:2]]
    if any(width is None for width in widths):
        return None
    gap = float(ranked[0]["objective_score"] - ranked[1]["objective_score"])
    threshold = max(widths)
    if gap > threshold:
        return None
    return (
        "Top candidates look clustered within reported uncertainty "
        f"(`{ranked[0].get('candidate_id', '?')}` {ranked[0]['objective_score']:.3f} vs "
        f"`{ranked[1].get('candidate_id', '?')}` {ranked[1]['objective_score']:.3f}); "
        "treat this as a plateau and change angle."
    )


def _repeated_overfit_signal(results: list[dict[str, Any]]) -> str | None:
    flagged: list[dict[str, Any]] = []
    for result in ranked_results(results):
        metric_name = _objective_metric_name(result)
        metrics = result.get("metrics", {})
        if not metric_name or not isinstance(metrics, dict):
            continue
        train_metrics = metrics.get("train", {})
        validation_metrics = metrics.get("validation", {})
        if not isinstance(train_metrics, dict) or not isinstance(validation_metrics, dict):
            continue
        train_value = train_metrics.get(metric_name)
        validation_value = validation_metrics.get(metric_name)
        if not isinstance(train_value, (int, float)) or not isinstance(
            validation_value, (int, float)
        ):
            continue
        gap = float(train_value - validation_value)
        threshold = max(0.03, abs(float(validation_value)) * 0.1)
        if gap > threshold:
            flagged.append(
                {
                    "candidate_id": result.get("candidate_id", "?"),
                    "gap": gap,
                }
            )
    if len(flagged) < 2:
        return None
    worst_gap = max(item["gap"] for item in flagged)
    return (
        f"{len(flagged)} candidates show the same train-validation gap pattern "
        f"(worst gap {worst_gap:.3f}); the current line of attack may be overfitting."
    )


def _stale_research_signal(journal_cycles: int, external_sources: int) -> str | None:
    if journal_cycles < 1:
        return None
    if external_sources == 0:
        return (
            f"No external research is recorded after {journal_cycles} cycle(s); "
            "refresh the search space before more tuning."
        )
    if journal_cycles >= 4 and external_sources < 2:
        return (
            f"External research looks thin for {journal_cycles} cycle(s) "
            f"({external_sources} source card(s)); consider a fresh literature pass."
        )
    return None


def _missing_analysis_signal(
    results: list[dict[str, Any]],
    *,
    has_error_analysis: bool,
    has_diagnostics: bool,
) -> str | None:
    candidate_count = len(ranked_results(results))
    if candidate_count >= 3 and not has_error_analysis and not has_diagnostics:
        return (
            f"{candidate_count} candidates exist but neither error analysis nor "
            "diagnostics are recorded yet; inspect failures before another variant."
        )
    return None


def _baseline_signal(results: list[dict[str, Any]]) -> str | None:
    ranked = ranked_results(results)
    if len(ranked) < 2:
        return None
    baselines = [
        result
        for result in ranked
        if "baseline" in str(result.get("candidate_id", "")).lower()
        or str(result.get("model_family", "")).lower() in {"rule_based", "constant"}
    ]
    if not baselines:
        return None
    best = ranked[0]
    best_baseline = baselines[0]
    threshold = max(
        0.01,
        _uncertainty_width(best) or 0.0,
        _uncertainty_width(best_baseline) or 0.0,
    )
    delta = float(best["objective_score"] - best_baseline["objective_score"])
    if best is best_baseline:
        return (
            f"The baseline still leads at {best['objective_score']:.3f}; stronger evidence "
            "is needed before adding more complexity."
        )
    if delta > threshold:
        return None
    return (
        f"The baseline is still competitive (`{best_baseline.get('candidate_id', '?')}` "
        f"{best_baseline['objective_score']:.3f} vs best "
        f"`{best.get('candidate_id', '?')}` {best['objective_score']:.3f}); "
        "favor explanation or data work over more tuning."
    )


def _resolve_signal_inputs(experiment_dir: Path, options: SignalOptions | None) -> SignalInputs:
    options = options or SignalOptions()
    return SignalInputs(
        results=(
            options.results if options.results is not None else results_snapshot(experiment_dir)
        ),
        journal_cycles=(
            options.journal_cycles
            if options.journal_cycles is not None
            else count_journal_cycles(experiment_dir)
        ),
        external_sources=(
            options.external_sources
            if options.external_sources is not None
            else _count_external_sources(experiment_dir)
        ),
        has_error_analysis=(
            options.has_error_analysis
            if options.has_error_analysis is not None
            else journal_mentions_error_analysis(experiment_dir)
        ),
        has_diagnostics=(
            options.has_diagnostics
            if options.has_diagnostics is not None
            else read_diagnostics_summary(experiment_dir) is not None
        ),
    )


_SIGNAL_INPUT_FIELDS = {
    "results",
    "journal_cycles",
    "external_sources",
    "has_error_analysis",
    "has_diagnostics",
}


def _coerce_signal_options(
    options: SignalOptions | None,
    legacy_overrides: dict[str, Any],
    *,
    allow_limit: bool,
) -> SignalOptions:
    allowed = set(_SIGNAL_INPUT_FIELDS)
    if allow_limit:
        allowed.add("limit")
    unknown = sorted(set(legacy_overrides) - allowed)
    if unknown:
        raise TypeError("unexpected signal option keyword arguments: " + ", ".join(unknown))

    resolved = options or SignalOptions()
    if not legacy_overrides:
        return resolved
    return replace(resolved, **legacy_overrides)


def research_signals(
    experiment_dir: Path,
    options: SignalOptions | None = None,
    **legacy_overrides: Any,
) -> list[str]:
    """Return bounded, advisory research signals derived from current artifacts.

    Preserves the original public API: returns list[str] (no priority scores).
    """
    options = _coerce_signal_options(
        options,
        legacy_overrides,
        allow_limit=True,
    )
    inputs = _resolve_signal_inputs(experiment_dir, options)
    candidates: list[tuple[int, str]] = _get_research_signal_observations(
        inputs.results,
        journal_cycles=inputs.journal_cycles,
        external_sources=inputs.external_sources,
        has_error_analysis=inputs.has_error_analysis,
        has_diagnostics=inputs.has_diagnostics,
    )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [signal for _, signal in candidates[: options.limit]]


def _get_research_signal_observations(
    results: list[dict[str, Any]],
    *,
    journal_cycles: int,
    external_sources: int,
    has_error_analysis: bool,
    has_diagnostics: bool,
) -> list[tuple[int, str]]:
    """Return research signal observations as (priority, text) tuples."""
    candidates: list[tuple[int, str]] = []
    for priority, signal in [
        (90, _plateau_signal(results)),
        (80, _repeated_overfit_signal(results)),
        (
            70,
            _missing_analysis_signal(
                results,
                has_error_analysis=has_error_analysis,
                has_diagnostics=has_diagnostics,
            ),
        ),
        (60, _stale_research_signal(journal_cycles, external_sources)),
        (50, _baseline_signal(results)),
    ]:
        if signal:
            candidates.append((priority, signal))
    return candidates


def advisory_signals(
    experiment_dir: Path,
    options: SignalOptions | None = None,
    **legacy_overrides: Any,
) -> list[tuple[int, str]]:
    """Collect, dedup, and return priority-sorted advisory signals from all sources.

    Returns at most ADVISORY_SIGNAL_CAP items.
    """
    options = _coerce_signal_options(
        options,
        legacy_overrides,
        allow_limit=False,
    )
    inputs = _resolve_signal_inputs(experiment_dir, options)

    return merge_observations(
        _get_research_signal_observations(
            inputs.results,
            journal_cycles=inputs.journal_cycles,
            external_sources=inputs.external_sources,
            has_error_analysis=inputs.has_error_analysis,
            has_diagnostics=inputs.has_diagnostics,
        ),
        get_diagnostics_observations(experiment_dir),
        get_evaluation_observations(experiment_dir),
        cap=ADVISORY_SIGNAL_CAP,
    )
