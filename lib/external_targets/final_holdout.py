from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from lib.candidate_result import CandidateResult
from lib.external_targets import box_office_oracle, estate_value_index

FinalHoldoutRunner = Callable[[str, Path], CandidateResult]


def _estate_runner(candidate_id: str, experiment_dir: Path) -> CandidateResult:
    return estate_value_index.run_candidate(
        candidate_id,
        experiment_dir=experiment_dir,
        include_test=True,
    )


def _box_runner(candidate_id: str, experiment_dir: Path) -> CandidateResult:
    return box_office_oracle.run_candidate(
        candidate_id,
        experiment_dir=experiment_dir,
        include_test=True,
    )


FINAL_HOLDOUT_RUNNERS: dict[str, FinalHoldoutRunner] = {
    "estate-value-index": _estate_runner,
    "estate-value-index-phase2": _estate_runner,
    "box-office-oracle": _box_runner,
    "box-office-oracle-phase2": _box_runner,
}


def run_final_holdout_candidates(
    experiment_dir: Path,
    candidate_ids: list[str],
) -> list[dict[str, Any]]:
    """Run registered external-target candidates with final holdout enabled."""
    runner = FINAL_HOLDOUT_RUNNERS.get(experiment_dir.name)
    if runner is None:
        supported = ", ".join(sorted(FINAL_HOLDOUT_RUNNERS))
        raise ValueError(
            f"No final-holdout runner is registered for {experiment_dir.name!r}. "
            f"Supported experiments: {supported}."
        )
    return [runner(candidate_id, experiment_dir).result_payload() for candidate_id in candidate_ids]
