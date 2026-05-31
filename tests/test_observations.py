from __future__ import annotations

from lib.observations import dedup_observations, merge_observations


def test_dedup_keeps_higher_priority_split_drift() -> None:
    observations = [
        (85, "Split drift detected: feature `age` shifted."),
        (60, "Split drift on validation: feature `age` looks unstable."),
    ]
    result = dedup_observations(observations)
    assert len(result) == 1
    assert result[0][0] == 85


def test_merge_observations_caps_and_sorts() -> None:
    merged = merge_observations([(10, "low"), (90, "high"), (50, "mid")], cap=2)
    assert merged == [(90, "high"), (50, "mid")]
