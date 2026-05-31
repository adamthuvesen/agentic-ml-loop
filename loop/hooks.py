"""Cycle hooks protocol and default implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from experiment import get_cross_learnings_enabled, learnings_file
from lib.learnings import extract_and_append_learnings

from .artifacts import compute_progress
from .prompts import CyclePrompt, build_cycle_prompt


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
        cycle_prompt = build_cycle_prompt(experiment_dir, cycle_id)
        return PreCycleResult(
            prompt_text=cycle_prompt.assemble(),
            cycle_prompt=cycle_prompt,
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
