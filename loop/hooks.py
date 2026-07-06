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


@dataclass(frozen=True)
class PostCycleContext:
    experiment_dir: Path
    cycle_id: str
    before_snapshot: dict[str, Any]
    after_snapshot: dict[str, Any]
    output: str
    marker: str


class CycleHooks(Protocol):
    """Structural protocol for pre/post-cycle extension points."""

    def pre_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        state: dict[str, Any],
    ) -> PreCycleResult: ...

    def post_cycle(self, context: PostCycleContext) -> PostCycleResult: ...


def call_post_cycle_hook(
    hooks: CycleHooks,
    context: PostCycleContext,
) -> PostCycleResult:
    """Call a post-cycle hook with the cycle context."""
    return hooks.post_cycle(context)


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
        context: PostCycleContext,
    ) -> PostCycleResult:
        progress_reasons = compute_progress(context.before_snapshot, context.after_snapshot)

        learnings_extracted = False
        if context.marker == "EXPERIMENT_COMPLETE" and get_cross_learnings_enabled(
            context.experiment_dir
        ):
            if extract_and_append_learnings(context.experiment_dir):
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
        context: PostCycleContext,
    ) -> PostCycleResult:
        result = super().post_cycle(context)
        try:
            scorecard = grade_cycle(
                context.experiment_dir,
                context.cycle_id,
                before=context.before_snapshot,
                after=context.after_snapshot,
                output_text=context.output,
            )
            write_scorecard(context.experiment_dir, scorecard)
            result.scorecard = scorecard
            print(scorecard.summary_line())
        except Exception:
            # The referee is advisory; never let scoring failure abort a cycle.
            logger.warning("Referee scoring failed for cycle %s", context.cycle_id, exc_info=True)
        return result
