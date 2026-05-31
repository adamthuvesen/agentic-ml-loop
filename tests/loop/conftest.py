from __future__ import annotations

import json
from pathlib import Path

from loop import CyclePrompt, PostCycleResult, PreCycleResult


class RecordingHooks:
    def __init__(self) -> None:
        self.pre_calls: list[tuple[Path, str]] = []
        self.post_calls: list[tuple[Path, str, str]] = []

    def pre_cycle(
        self, experiment_dir: Path, cycle_id: str, state: dict[str, object]
    ) -> PreCycleResult:
        self.pre_calls.append((experiment_dir, cycle_id))
        prompt = CyclePrompt(
            static_sections=["# Custom static"],
            dynamic_sections=[f"cycle={cycle_id}", f"state={state['cycle_count']}"],
        )
        return PreCycleResult(prompt_text=prompt.assemble(), cycle_prompt=prompt)

    def post_cycle(
        self,
        experiment_dir: Path,
        cycle_id: str,
        before_snapshot: dict[str, object],
        after_snapshot: dict[str, object],
        output: str,
        marker: str,
    ) -> PostCycleResult:
        self.post_calls.append((experiment_dir, cycle_id, marker))
        return PostCycleResult(progress_reasons=["custom:hook"])


def make_experiment_dir(
    tmp_path: Path, *, results: list | None = None, journal: str = ""
) -> Path:
    """Create a minimal experiment directory for prompt tests."""
    d = tmp_path / "exp"
    d.mkdir(exist_ok=True)
    (d / "experiment.md").write_text("# Experiment\n")
    (d / "research_journal.md").write_text(journal or "# Journal\n")
    (d / "research_sources.md").write_text("# Research Sources\n")
    (d / "results.json").write_text(
        json.dumps(results if results is not None else []) + "\n"
    )
    return d


def _make_experiment(
    tmp_path: Path, *, results: list | None = None, journal: str = ""
) -> Path:
    return make_experiment_dir(tmp_path, results=results, journal=journal)
