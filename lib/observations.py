"""Shared priority-scored observation helpers for advisory signals."""

from __future__ import annotations

from collections.abc import Iterable

Observation = tuple[int, str]

_DEDUP_KEYWORD_GROUPS: list[list[str]] = [
    ["split", "drift"],
    ["split", "concern"],
    ["split", "reliab"],
    ["leakage"],
    ["instability", "leaderboard"],
    ["metric", "fit"],
]


def dedup_observations(observations: list[Observation]) -> list[Observation]:
    """Remove overlapping observations, keeping the higher-priority one."""
    sorted_obs = sorted(observations, key=lambda item: item[0], reverse=True)
    kept: list[Observation] = []
    used_groups: set[int] = set()

    for priority, text in sorted_obs:
        text_lower = text.lower()
        is_duplicate = False
        for group_idx in used_groups:
            if all(kw in text_lower for kw in _DEDUP_KEYWORD_GROUPS[group_idx]):
                is_duplicate = True
                break
        if is_duplicate:
            continue

        for group_idx, keywords in enumerate(_DEDUP_KEYWORD_GROUPS):
            if group_idx not in used_groups and all(kw in text_lower for kw in keywords):
                used_groups.add(group_idx)
                break

        kept.append((priority, text))

    return kept


def merge_observations(
    *sources: Iterable[Observation],
    cap: int,
) -> list[Observation]:
    """Concatenate sources, dedup, sort by priority, and cap."""
    combined: list[Observation] = []
    for source in sources:
        combined.extend(source)
    deduped = dedup_observations(combined)
    deduped.sort(key=lambda item: item[0], reverse=True)
    return deduped[:cap]
