"""Cycle hooks protocol and default implementation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from experiment import get_cross_learnings_enabled, learnings_file
from lib.learnings import extract_and_append_learnings
from lib.referee import CycleScorecard, grade_cycle, write_scorecard

from .artifacts import compute_progress
from .prompts import CyclePrompt
from .prompts import cycle_prompt as make_cycle_prompt

logger = logging.getLogger(__name__)


@dataclass
class PreCycleResult:
    """Payload returned by ``CycleHooks.pre_cycle()``."""

    prompt_text: str
    cycle_prompt: CyclePrompt


@dataclass
class PostCycleResult:
    """Payload returned by ``CycleHooks.post_cycle()``."""

    progress_reasons: list[str] = field(default_factory=list)
    is_stalled: bool = False
    learnings_extracted: bool = False
    scorecard: CycleScorecard | None = None


class CycleHooks(Protocol):
    """Structural protocol for pre/post-cycle extension points."""

    def pre_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        state: dict[str, Any],
    ) -> PreCycleResult: ...

    def post_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        before_snapshot: dict[str, Any],
        after_snapshot: dict[str, Any],
        output: str,
        marker: str,
    ) -> PostCycleResult: ...


class DefaultCycleHooks:
    """Default implementation reproducing current ``run_cycle`` / ``run_loop`` behaviour."""

    def pre_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        state: dict[str, Any],
    ) -> PreCycleResult:
        prompt = make_cycle_prompt(experiment_dir, cycle_id)
        return PreCycleResult(
            prompt_text=prompt.assemble(),
            cycle_prompt=prompt,
        )

    def post_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        before_snapshot: dict[str, Any],
        after_snapshot: dict[str, Any],
        output: str,
        marker: str,
    ) -> PostCycleResult:
        progress_reasons = compute_progress(before_snapshot, after_snapshot)

        learnings_extracted = False
        if marker == "EXPERIMENT_COMPLETE" and get_cross_learnings_enabled(experiment_dir):
            if extract_and_append_learnings(experiment_dir):
                print(f"Learnings appended to {learnings_file()}")
                learnings_extracted = True
            else:
                print("No generalizable learnings extracted.")

        return PostCycleResult(
            progress_reasons=progress_reasons,
            learnings_extracted=learnings_extracted,
        )


class RefereeCycleHooks(DefaultCycleHooks):
    """Default hooks plus an advisory per-cycle research referee scorecard.

    Computes and persists a :class:`~lib.referee.CycleScorecard` after the
    default post-cycle work. The scorecard is advisory — it is attached to the
    result and written to ``cycles/<id>/scorecard.json`` but never blocks the
    loop.
    """

    def post_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        before_snapshot: dict[str, Any],
        after_snapshot: dict[str, Any],
        output: str,
        marker: str,
    ) -> PostCycleResult:
        result = super().post_cycle(
            experiment_dir, cycle_id, before_snapshot, after_snapshot, output, marker
        )
        try:
            scorecard = grade_cycle(
                experiment_dir,
                cycle_id,
                before=before_snapshot,
                after=after_snapshot,
                output_text=output,
            )
            write_scorecard(experiment_dir, scorecard)
            result.scorecard = scorecard
            print(scorecard.summary_line())
        except Exception:
            # The referee is advisory; never let scoring failure abort a cycle.
            logger.warning("Referee scoring failed for cycle %s", cycle_id, exc_info=True)
        return result
