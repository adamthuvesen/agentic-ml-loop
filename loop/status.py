from __future__ import annotations

import shlex
from datetime import UTC, datetime
from pathlib import Path

from lib.referee import latest_scorecard_line

from .constants import DEFAULT_MAX_ATTEMPTS_PER_CYCLE
from .loop_state import LoopState
from .prompts import latest_hypothesis, top_results_lines
from .ui import format_elapsed, format_runner_label, iso_to_datetime


def _base_status_lines(experiment_dir: Path, state: LoopState) -> list[str]:
    return [
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


def _active_cycle_lines(state: LoopState) -> list[str]:
    lines: list[str] = []
    if state.active_cycle_id:
        lines.append(f"- Active cycle: `{state.active_cycle_id}`")
    if state.active_attempt is not None:
        lines.append(f"- Active attempt: {state.active_attempt}/{DEFAULT_MAX_ATTEMPTS_PER_CYCLE}")
    if state.active_started_at:
        lines.append(f"- Active started at: `{state.active_started_at}`")
        elapsed = datetime.now(UTC) - iso_to_datetime(state.active_started_at)
        lines.append(f"- Active for: `{format_elapsed(int(elapsed.total_seconds()))}`")
    return lines


def _selection_frozen_lines(state: LoopState) -> list[str]:
    if not state.selection_frozen:
        return []
    frozen_ids = ", ".join(state.frozen_candidate_ids or []) or "n/a"
    lines = [f"- Selection frozen: `{frozen_ids}`"]
    if state.frozen_at_cycle:
        lines.append(f"- Frozen at cycle: `{state.frozen_at_cycle}`")
    if state.freeze_reason:
        lines.append(f"- Freeze reason: {state.freeze_reason}")
    return lines


def _final_holdout_lines(state: LoopState) -> list[str]:
    if not state.final_holdout_accessed:
        return []
    lines = ["- Final holdout: `accessed`"]
    if state.final_holdout_at:
        lines.append(f"- Final holdout at: `{state.final_holdout_at}`")
    if state.final_holdout_path:
        lines.append(f"- Final holdout artifact: `{state.final_holdout_path}`")
    return lines


def _budget_mode_lines(state: LoopState) -> list[str]:
    if not state.enforce_budget_until_limit:
        return []
    return [
        "- Budget mode: `run until limit` — ignores EXPERIMENT_COMPLETE until "
        "max cycles or max hours is reached, but still stops after final holdout access"
    ]


def _interruption_lines(state: LoopState, has_active_lock: bool) -> list[str]:
    if state.status == "running" and not has_active_lock:
        return ["- Note: no active loop lock detected; the previous run may have been interrupted."]
    return []


def status_markdown(
    experiment_dir: Path,
    state: LoopState,
    *,
    has_active_lock: bool,
) -> str:
    """Return human-readable loop status and top results as Markdown."""
    lines = _base_status_lines(experiment_dir, state)
    lines.extend(_active_cycle_lines(state))

    scorecard_line = latest_scorecard_line(experiment_dir)
    if scorecard_line:
        lines.append(f"- {scorecard_line}")
    if state.stop_reason:
        lines.append(f"- Stop reason: `{state.stop_reason}`")
    lines.extend(_selection_frozen_lines(state))
    lines.extend(_final_holdout_lines(state))
    lines.extend(_budget_mode_lines(state))
    lines.extend(_interruption_lines(state, has_active_lock))

    lines.extend(["", "## Results", ""])
    lines.extend(top_results_lines(experiment_dir))

    hypothesis = latest_hypothesis(experiment_dir)
    if hypothesis:
        lines.extend(["", "## Current Direction", ""])
        lines.append(f"- {hypothesis}")

    return "\n".join(lines) + "\n"
