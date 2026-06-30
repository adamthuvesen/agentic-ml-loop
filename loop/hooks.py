"""Cycle hooks protocol and default implementation."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
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

    def post_cycle(self, *args: Any, **kwargs: Any) -> PostCycleResult: ...


def _post_cycle_context_from_args(
    context_or_experiment_dir: PostCycleContext | Path,
    legacy_args: tuple[Any, ...],
    legacy_kwargs: dict[str, Any],
) -> PostCycleContext:
    if isinstance(context_or_experiment_dir, PostCycleContext):
        if legacy_args or legacy_kwargs:
            raise TypeError("post_cycle() got extra arguments with PostCycleContext")
        return context_or_experiment_dir

    fields = ["cycle_id", "before_snapshot", "after_snapshot", "output", "marker"]
    if len(legacy_args) > len(fields):
        raise TypeError("post_cycle() got too many positional arguments")
    positional_values = dict(zip(fields, legacy_args, strict=False))
    duplicate = sorted(set(positional_values) & set(legacy_kwargs))
    if duplicate:
        raise TypeError("post_cycle() got multiple values for argument(s): " + ", ".join(duplicate))

    unknown = sorted(set(legacy_kwargs) - set(fields))
    if unknown:
        raise TypeError("post_cycle() got unexpected keyword arguments: " + ", ".join(unknown))
    values = {**positional_values, **legacy_kwargs}
    missing = [field for field in fields if field not in values]
    if missing:
        raise TypeError(
            "post_cycle() expected PostCycleContext or legacy arguments "
            "(experiment_dir, cycle_id, before_snapshot, after_snapshot, output, marker); "
            "missing: " + ", ".join(missing)
        )
    return PostCycleContext(
        experiment_dir=context_or_experiment_dir,
        cycle_id=values["cycle_id"],
        before_snapshot=values["before_snapshot"],
        after_snapshot=values["after_snapshot"],
        output=values["output"],
        marker=values["marker"],
    )


def _expects_legacy_post_cycle(
    post_cycle: Callable[..., PostCycleResult],
) -> bool:
    try:
        signature = inspect.signature(post_cycle)
    except (TypeError, ValueError):
        return False
    required_positionals = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is inspect.Parameter.empty
    ]
    return len(required_positionals) > 1


def call_post_cycle_hook(
    hooks: CycleHooks,
    context: PostCycleContext,
) -> PostCycleResult:
    """Call a post-cycle hook, accepting both current and legacy hook shapes."""
    if _expects_legacy_post_cycle(hooks.post_cycle):
        return hooks.post_cycle(
            context.experiment_dir,
            context.cycle_id,
            context.before_snapshot,
            context.after_snapshot,
            context.output,
            context.marker,
        )
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
        context_or_experiment_dir: PostCycleContext | Path,
        *legacy_args: Any,
        **legacy_kwargs: Any,
    ) -> PostCycleResult:
        context = _post_cycle_context_from_args(
            context_or_experiment_dir, legacy_args, legacy_kwargs
        )
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
        context_or_experiment_dir: PostCycleContext | Path,
        *legacy_args: Any,
        **legacy_kwargs: Any,
    ) -> PostCycleResult:
        context = _post_cycle_context_from_args(
            context_or_experiment_dir, legacy_args, legacy_kwargs
        )
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
