"""Typed loop supervisor state persisted as ``loop_state.json``."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lib.io import load_json, write_json

from .constants import STATE_PATH_NAME
from .enums import CycleResult, LoopStatus, StopReason


@dataclass
class LoopState:
    experiment_id: str
    status: LoopStatus
    runner_name: str
    runner_model: str | None
    runner_effort: str | None
    runner_command: list[str]
    runner_timeout_seconds: int
    cycle_count: int
    consecutive_no_progress_cycles: int
    consecutive_failed_cycles: int
    last_successful_cycle_id: str | None
    last_cycle_result: CycleResult | None
    max_cycles: int | None
    max_hours: float | None
    enforce_budget_until_limit: bool
    started_at: str
    updated_at: str
    stop_reason: StopReason | None
    active_cycle_id: str | None = None
    active_started_at: str | None = None
    active_objective: str | None = None
    active_attempt: int | None = None
    last_attempt_outcome: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoopState:
        """Build state from on-disk JSON, applying defaults for missing keys.

        String enum fields are coerced from their stored values; legacy strings
        already match the enum values, so older state files load unchanged.
        """
        runner_name = str(data.get("runner_name", "unknown"))
        runner_command = data.get("runner_command") or [runner_name]
        last_cycle_result = data.get("last_cycle_result")
        stop_reason = data.get("stop_reason")
        return cls(
            experiment_id=str(data.get("experiment_id", "")),
            status=LoopStatus(data.get("status", LoopStatus.IDLE)),
            runner_name=runner_name,
            runner_model=data.get("runner_model"),
            runner_effort=data.get("runner_effort"),
            runner_command=[str(part) for part in runner_command],
            runner_timeout_seconds=int(data.get("runner_timeout_seconds", 1800)),
            cycle_count=int(data.get("cycle_count", 0)),
            consecutive_no_progress_cycles=int(data.get("consecutive_no_progress_cycles", 0)),
            consecutive_failed_cycles=int(data.get("consecutive_failed_cycles", 0)),
            last_successful_cycle_id=data.get("last_successful_cycle_id"),
            last_cycle_result=(
                CycleResult(last_cycle_result) if last_cycle_result is not None else None
            ),
            max_cycles=data.get("max_cycles"),
            max_hours=data.get("max_hours"),
            enforce_budget_until_limit=bool(data.get("enforce_budget_until_limit", False)),
            started_at=str(data.get("started_at", "")),
            updated_at=str(data.get("updated_at", "")),
            stop_reason=StopReason(stop_reason) if stop_reason is not None else None,
            active_cycle_id=data.get("active_cycle_id"),
            active_started_at=data.get("active_started_at"),
            active_objective=data.get("active_objective"),
            active_attempt=data.get("active_attempt"),
            last_attempt_outcome=data.get("last_attempt_outcome"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_state(experiment_dir: Path) -> LoopState:
    """Load ``loop_state.json`` for the experiment."""
    return LoopState.from_dict(load_json(experiment_dir / STATE_PATH_NAME))


def write_state_file(experiment_dir: Path, state: LoopState) -> None:
    """Persist loop state JSON (caller sets ``updated_at`` if needed)."""
    write_json(experiment_dir / STATE_PATH_NAME, state.to_dict())
