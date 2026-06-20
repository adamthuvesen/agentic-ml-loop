"""Loop stop-condition evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .constants import DEFAULT_FAILURE_LIMIT, DEFAULT_STALL_LIMIT
from .enums import CycleResult, LoopStatus, StopReason
from .loop_state import LoopState
from .ui import iso_to_datetime


@dataclass(frozen=True)
class StopDecision:
    should_end: bool
    reason: StopReason | None
    final_status: LoopStatus


def _budget_stop(state: LoopState) -> StopDecision | None:
    budget_failed = state.last_cycle_result == CycleResult.FAILED
    final_status = LoopStatus.FAILED if budget_failed else LoopStatus.COMPLETED
    if state.max_cycles is not None and state.cycle_count >= state.max_cycles:
        return StopDecision(True, StopReason.MAX_CYCLES_REACHED, final_status)
    if state.max_hours is not None:
        elapsed = datetime.now(UTC) - iso_to_datetime(state.started_at)
        if elapsed.total_seconds() >= float(state.max_hours) * 3600:
            return StopDecision(True, StopReason.MAX_HOURS_REACHED, final_status)
    return None


def evaluate_stop(state: LoopState) -> StopDecision:
    """Decide if the supervisor should exit before starting another cycle."""
    if state.final_holdout_accessed:
        return StopDecision(True, StopReason.FINAL_HOLDOUT_ACCESSED, LoopStatus.COMPLETED)

    if not state.enforce_budget_until_limit and state.last_cycle_result == CycleResult.COMPLETE:
        return StopDecision(True, StopReason.EXPERIMENT_COMPLETE, LoopStatus.COMPLETED)

    if state.consecutive_no_progress_cycles >= DEFAULT_STALL_LIMIT:
        return StopDecision(True, StopReason.SLICE_STALL, LoopStatus.STALLED)
    if state.consecutive_failed_cycles >= DEFAULT_FAILURE_LIMIT:
        return StopDecision(True, StopReason.TOO_MANY_FAILED_CYCLES, LoopStatus.FAILED)

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
