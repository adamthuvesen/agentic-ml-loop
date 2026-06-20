"""Named constants for loop supervisor state.

These are ``StrEnum`` members, so each compares equal to its string value and
serializes to that plain string in JSON. Existing ``loop_state.json`` files and
string comparisons keep working; the enums just give the values a single home
and let callers avoid bare string literals.
"""

from __future__ import annotations

from enum import StrEnum


class LoopStatus(StrEnum):
    """Supervisor lifecycle status persisted in ``loop_state.json``."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    STALLED = "stalled"
    FAILED = "failed"


class CycleResult(StrEnum):
    """Outcome of a single research cycle."""

    PROGRESS = "progress"
    NO_PROGRESS = "no_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    TOO_MANY_FAILED_ATTEMPTS = "too_many_failed_attempts"


class AttemptOutcomeKind(StrEnum):
    """Fixed ``last_attempt_outcome`` markers.

    The per-attempt "running" marker is dynamic (it embeds the attempt number),
    so use :func:`attempt_running` for that case; everything else is one of
    these members or a :class:`CycleResult` value.
    """

    EXCEPTION = "exception"
    INTERRUPTED = "interrupted"
    INTERRUPTED_BEFORE_RESUME = "interrupted_before_resume"


def attempt_running(attempt: int) -> str:
    """Return the in-progress ``last_attempt_outcome`` marker for ``attempt``."""
    return f"attempt_{attempt:02d}_running"


class StopReason(StrEnum):
    """Why the supervisor stopped, recorded as ``stop_reason``."""

    MAX_CYCLES_REACHED = "max_cycles_reached"
    MAX_HOURS_REACHED = "max_hours_reached"
    EXPERIMENT_COMPLETE = "experiment_complete"
    FINAL_HOLDOUT_ACCESSED = "final_holdout_accessed"
    SLICE_STALL = "slice_stall"
    TOO_MANY_FAILED_CYCLES = "too_many_failed_cycles"
    INTERRUPTED = "interrupted"
