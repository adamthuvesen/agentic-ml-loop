from __future__ import annotations

import shlex
from datetime import UTC, datetime
from pathlib import Path

from lib.referee import latest_scorecard_line

from .constants import DEFAULT_MAX_ATTEMPTS_PER_CYCLE
from .loop_state import LoopState
from .prompts import latest_hypothesis, top_results_lines
from .ui import format_elapsed, format_runner_label, iso_to_datetime


def status_markdown(
    experiment_dir: Path,
    state: LoopState,
    *,
    has_active_lock: bool,
) -> str:
    """Return human-readable loop status and top results as Markdown."""
    lines = [
        f"# Loop Status: {experiment_dir.name}",
        "",
        f"- Status: `{state.status}`",
        f"- Runner: `{format_runner_label(state.to_dict())}`",
        f"- Runner command: `{shlex.join(state.runner_command)}`",
        f"- Runner timeout: {state.runner_timeout_seconds}s",
        f"- Cycle count: {state.cycle_count}",
        f"- Last successful cycle: {state.last_successful_cycle_id or 'n/a'}",
        f"- Started at: `{state.started_at}`",
        f"- Updated at: `{state.updated_at}`",
    ]

    if state.active_cycle_id:
        lines.append(f"- Active cycle: `{state.active_cycle_id}`")
    if state.active_attempt is not None:
        lines.append(f"- Active attempt: {state.active_attempt}/{DEFAULT_MAX_ATTEMPTS_PER_CYCLE}")
    if state.active_started_at:
        lines.append(f"- Active started at: `{state.active_started_at}`")
        elapsed = datetime.now(UTC) - iso_to_datetime(state.active_started_at)
        lines.append(f"- Active for: `{format_elapsed(int(elapsed.total_seconds()))}`")

    scorecard_line = latest_scorecard_line(experiment_dir)
    if scorecard_line:
        lines.append(f"- {scorecard_line}")
    if state.stop_reason:
        lines.append(f"- Stop reason: `{state.stop_reason}`")
    if state.selection_frozen:
        frozen_ids = ", ".join(state.frozen_candidate_ids or []) or "n/a"
        lines.append(f"- Selection frozen: `{frozen_ids}`")
        if state.frozen_at_cycle:
            lines.append(f"- Frozen at cycle: `{state.frozen_at_cycle}`")
        if state.freeze_reason:
            lines.append(f"- Freeze reason: {state.freeze_reason}")
    if state.final_holdout_accessed:
        lines.append("- Final holdout: `accessed`")
        if state.final_holdout_at:
            lines.append(f"- Final holdout at: `{state.final_holdout_at}`")
        if state.final_holdout_path:
            lines.append(f"- Final holdout artifact: `{state.final_holdout_path}`")
    if state.enforce_budget_until_limit:
        lines.append(
            "- Budget mode: `run until limit` — ignores EXPERIMENT_COMPLETE until "
            "max cycles or max hours is reached, but still stops after final holdout access"
        )
    if state.status == "running" and not has_active_lock:
        lines.append(
            "- Note: no active loop lock detected; the previous run may have been interrupted."
        )

    lines.extend(["", "## Results", ""])
    lines.extend(top_results_lines(experiment_dir))

    hypothesis = latest_hypothesis(experiment_dir)
    if hypothesis:
        lines.extend(["", "## Current Direction", ""])
        lines.append(f"- {hypothesis}")

    return "\n".join(lines) + "\n"
