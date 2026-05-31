from __future__ import annotations

import json

from loop.enums import (
    AttemptOutcomeKind,
    CycleResult,
    LoopStatus,
    StopReason,
    attempt_running,
)
from loop.loop_state import LoopState


class TestStrEnumBackwardCompat:
    def test_members_equal_their_string_values(self) -> None:
        assert LoopStatus.RUNNING == "running"
        assert CycleResult.COMPLETE == "complete"
        assert StopReason.SLICE_STALL == "slice_stall"
        assert AttemptOutcomeKind.EXCEPTION == "exception"

    def test_str_renders_plain_value(self) -> None:
        # StrEnum renders as its value, not "LoopStatus.RUNNING".
        assert f"{LoopStatus.RUNNING}" == "running"
        assert str(CycleResult.PROGRESS) == "progress"

    def test_attempt_running_marker_is_zero_padded(self) -> None:
        assert attempt_running(1) == "attempt_01_running"
        assert attempt_running(12) == "attempt_12_running"


def _state_with(**overrides: object) -> LoopState:
    base: dict[str, object] = {
        "experiment_id": "exp",
        "status": LoopStatus.RUNNING,
        "runner_name": "claude",
        "runner_command": ["claude"],
        "last_cycle_result": CycleResult.PROGRESS,
        "stop_reason": StopReason.SLICE_STALL,
    }
    base.update(overrides)
    return LoopState.from_dict(base)


class TestLoopStateSerialization:
    def test_enum_fields_serialize_to_plain_strings(self) -> None:
        state = _state_with()
        blob = json.loads(json.dumps(state.to_dict()))
        assert blob["status"] == "running"
        assert blob["last_cycle_result"] == "progress"
        assert blob["stop_reason"] == "slice_stall"

    def test_legacy_string_state_loads_into_enums(self) -> None:
        # Older loop_state.json stored bare strings; these must still load.
        state = LoopState.from_dict(
            {
                "experiment_id": "exp",
                "status": "stalled",
                "last_cycle_result": "complete",
                "stop_reason": "interrupted",
            }
        )
        assert state.status is LoopStatus.STALLED
        assert state.last_cycle_result is CycleResult.COMPLETE
        assert state.stop_reason is StopReason.INTERRUPTED

    def test_round_trip_is_stable(self) -> None:
        state = _state_with()
        reloaded = LoopState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert reloaded == state

    def test_none_optionals_survive_round_trip(self) -> None:
        state = _state_with(last_cycle_result=None, stop_reason=None)
        reloaded = LoopState.from_dict(json.loads(json.dumps(state.to_dict())))
        assert reloaded.last_cycle_result is None
        assert reloaded.stop_reason is None
