from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from lib.io import load_json

LEDGER_COLUMNS = [
    "cycle_id",
    "started_at",
    "completed_at",
    "result",
    "completion_marker",
    "attempt_count",
    "progress_reasons",
    "candidates_added",
    "candidates_changed",
    "best_candidate_id",
    "best_objective_score",
    "runtime_seconds",
    "cost_usd",
]


def _cycle_summary_paths(experiment_dir: Path) -> list[Path]:
    cycles_dir = experiment_dir / "cycles"
    if not cycles_dir.is_dir():
        return []
    return sorted(cycles_dir.glob("*/cycle_summary.json"))


def _result_items(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = snapshot.get("results_by_id")
    return value if isinstance(value, dict) else {}


def _candidate_delta(summary: dict[str, Any]) -> tuple[list[str], list[str]]:
    before = _result_items(summary.get("before_snapshot", {}))
    after = _result_items(summary.get("after_snapshot", {}))
    before_ids = set(before)
    after_ids = set(after)
    added = sorted(after_ids - before_ids)
    changed = sorted(
        candidate_id
        for candidate_id in before_ids & after_ids
        if before[candidate_id] != after[candidate_id]
    )
    return added, changed


def _best_candidate(summary: dict[str, Any]) -> tuple[str, str]:
    after = _result_items(summary.get("after_snapshot", {}))
    best_id = ""
    best_score = float("-inf")
    for candidate_id, result in after.items():
        score = result.get("objective_score")
        if isinstance(score, int | float) and score > best_score:
            best_id = candidate_id
            best_score = float(score)
    if not best_id:
        return "", ""
    return best_id, str(best_score)


def _sum_candidate_field(
    summary: dict[str, Any],
    candidate_ids: list[str],
    field_name: str,
) -> str:
    after = _result_items(summary.get("after_snapshot", {}))
    total = 0.0
    found = False
    for candidate_id in candidate_ids:
        hyperparameters = after.get(candidate_id, {}).get("hyperparameters", {})
        if not isinstance(hyperparameters, dict):
            continue
        value = hyperparameters.get(field_name)
        if isinstance(value, int | float):
            total += float(value)
            found = True
    return str(round(total, 6)) if found else ""


def row_from_cycle_summary(summary: dict[str, Any]) -> dict[str, str]:
    """Return one CSV row for a persisted cycle summary."""
    added, changed = _candidate_delta(summary)
    touched = sorted({*added, *changed})
    best_candidate_id, best_objective_score = _best_candidate(summary)
    progress_reasons = summary.get("progress_reasons", [])
    if not isinstance(progress_reasons, list):
        progress_reasons = []
    attempts = summary.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    return {
        "cycle_id": str(summary.get("cycle_id", "")),
        "started_at": str(summary.get("started_at", "")),
        "completed_at": str(summary.get("completed_at", "")),
        "result": str(summary.get("result", "")),
        "completion_marker": str(summary.get("completion_marker", "")),
        "attempt_count": str(max(1, len(attempts))),
        "progress_reasons": ";".join(str(reason) for reason in progress_reasons),
        "candidates_added": ";".join(added),
        "candidates_changed": ";".join(changed),
        "best_candidate_id": best_candidate_id,
        "best_objective_score": best_objective_score,
        "runtime_seconds": _sum_candidate_field(summary, touched, "runtime_seconds"),
        "cost_usd": _sum_candidate_field(summary, touched, "cost_usd"),
    }


def refresh_cycle_metrics(experiment_dir: Path) -> Path:
    """Rebuild ``outputs/cycle_metrics.csv`` from persisted cycle summaries."""
    rows: list[dict[str, str]] = []
    for summary_path in _cycle_summary_paths(experiment_dir):
        payload = load_json(summary_path)
        if isinstance(payload, dict):
            rows.append(row_from_cycle_summary(payload))

    output_path = experiment_dir / "outputs" / "cycle_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
