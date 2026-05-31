"""Loop stop-condition evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .constants import DEFAULT_FAILURE_LIMIT, DEFAULT_STALL_LIMIT
from .loop_state import LoopState
from .ui import iso_to_datetime


@dataclass(frozen=True)
class StopDecision:
    should_end: bool
    reason: str | None
    final_status: str


def _budget_stop(state: LoopState) -> StopDecision | None:
    if state.max_cycles is not None and state.cycle_count >= state.max_cycles:
        final_status = "failed" if state.last_cycle_result == "failed" else "completed"
        return StopDecision(True, "max_cycles_reached", final_status)
    if state.max_hours is not None:
        elapsed = datetime.now(timezone.utc) - iso_to_datetime(state.started_at)
        if elapsed.total_seconds() >= float(state.max_hours) * 3600:
            final_status = "failed" if state.last_cycle_result == "failed" else "completed"
            return StopDecision(True, "max_hours_reached", final_status)
    return None


def evaluate_stop(state: LoopState) -> StopDecision:
    """Decide if the supervisor should exit before starting another cycle."""
    if not state.enforce_budget_until_limit and state.last_cycle_result == "complete":
        return StopDecision(True, "experiment_complete", "completed")

    if state.consecutive_no_progress_cycles >= DEFAULT_STALL_LIMIT:
        return StopDecision(True, "slice_stall", "stalled")
    if state.consecutive_failed_cycles >= DEFAULT_FAILURE_LIMIT:
        return StopDecision(True, "too_many_failed_cycles", "failed")

    budget = _budget_stop(state)
    if budget is not None:
        return budget

    return StopDecision(False, None, state.status)


def should_stop(state: LoopState | dict[str, Any]) -> tuple[bool, str | None, str]:
    """Compatibility wrapper returning ``(should_end, reason, final_status)``."""
    if isinstance(state, dict):
        state = LoopState.from_dict(state)
    decision = evaluate_stop(state)
    return decision.should_end, decision.reason, decision.final_status
