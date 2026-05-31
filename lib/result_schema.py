from __future__ import annotations

import math
from typing import Any


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def non_finite_paths(value: Any, path: str) -> list[str]:
    if isinstance(value, float) and not math.isfinite(value):
        return [path]
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(non_finite_paths(child, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths: list[str] = []
        for idx, child in enumerate(value):
            paths.extend(non_finite_paths(child, f"{path}[{idx}]"))
        return paths
    return []


def validate_result_entry(
    entry: Any,
    location: str,
    *,
    strict_completion: bool = False,
) -> list[str]:
    messages: list[str] = []
    if not isinstance(entry, dict):
        return [f"{location} must be an object"]

    candidate_id = entry.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        messages.append(f"{location} requires non-empty string candidate_id")

    objective_score = entry.get("objective_score")
    if not is_finite_number(objective_score):
        messages.append(f"{location} requires finite numeric objective_score")

    objective_metric = entry.get("objective_metric")
    if not isinstance(objective_metric, str) or not objective_metric.strip():
        msg = f"{location} requires non-empty string objective_metric"
        messages.append(msg if strict_completion else f"warning: {msg}")

    for path in non_finite_paths(entry, location):
        if path != f"{location}.objective_score":
            messages.append(f"{path} must be finite JSON number")

    return messages


def has_minimum_result_fields(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    candidate_id = entry.get("candidate_id")
    return (
        isinstance(candidate_id, str)
        and bool(candidate_id.strip())
        and is_finite_number(entry.get("objective_score"))
    )


def validate_result_entries(
    results: list[Any],
    *,
    strict_completion: bool = False,
    location_prefix: str = "results.json",
) -> list[str]:
    messages: list[str] = []
    valid_entries = 0

    for index, entry in enumerate(results):
        location = f"{location_prefix}[{index}]"
        messages.extend(
            validate_result_entry(
                entry,
                location,
                strict_completion=strict_completion,
            )
        )
        if has_minimum_result_fields(entry):
            valid_entries += 1

    if strict_completion and results and valid_entries == 0:
        messages.append("strict completion requires at least one valid result entry")

    return messages
