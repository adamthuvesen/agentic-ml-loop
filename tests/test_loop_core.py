from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from loop import (
    LOCK_PATH_NAME,
    DefaultCycleHooks,
    acquire_lock,
    release_lock,
    resume_command,
)
from loop.core import (
    cli_parser,
    initial_state,
    run_loop,
    should_stop,
)
from tests.loop.conftest import RecordingHooks, _make_experiment


class TestShouldStopBudgetMode:
    def _minimal_state(self, **kwargs: object) -> dict[str, object]:
        base: dict[str, object] = {
            "status": "running",
            "cycle_count": 1,
            "consecutive_no_progress_cycles": 0,
            "consecutive_failed_cycles": 0,
            "last_cycle_result": None,
            "max_cycles": None,
            "max_hours": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "enforce_budget_until_limit": False,
        }
        base.update(kwargs)
        return base

    def test_run_until_limit_ignores_complete_when_cycles_remain(self) -> None:
        should_end, reason, _ = should_stop(
            self._minimal_state(
                last_cycle_result="complete",
                cycle_count=3,
                max_cycles=10,
                enforce_budget_until_limit=True,
            )
        )
        assert not should_end
        assert reason is None

    def test_run_until_limit_stops_at_max_cycles(self) -> None:
        should_end, reason, _ = should_stop(
            self._minimal_state(
                last_cycle_result="complete",
                cycle_count=10,
                max_cycles=10,
                enforce_budget_until_limit=True,
            )
        )
        assert should_end
        assert reason == "max_cycles_reached"

    def test_run_until_limit_still_stops_on_stall(self) -> None:
        should_end, reason, _ = should_stop(
            self._minimal_state(
                last_cycle_result="progress",
                cycle_count=2,
                max_cycles=10,
                consecutive_no_progress_cycles=3,
                enforce_budget_until_limit=True,
            )
        )
        assert should_end
        assert reason == "slice_stall"

    def test_default_mode_stops_on_experiment_complete(self) -> None:
        should_end, reason, _ = should_stop(
            self._minimal_state(
                last_cycle_result="complete",
                cycle_count=1,
                max_cycles=10,
                enforce_budget_until_limit=False,
            )
        )
        assert should_end
        assert reason == "experiment_complete"


class TestAcquireLock:
    def test_reclaims_stale_lock(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        lock_path = d / LOCK_PATH_NAME
        lock_path.write_text('{"pid": 999999, "acquired_at": "2025-01-01T00:00:00+00:00"}\n')

        with patch("loop.core.active_lock_pid", return_value=None):
            acquired = acquire_lock(d)

        try:
            payload = json.loads(lock_path.read_text())
            assert acquired == lock_path
            assert payload["pid"] == os.getpid()
        finally:
            release_lock(lock_path)


class TestBuildParser:
    def test_rejects_non_positive_cycle_budget(self) -> None:
        parser = cli_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["start", "experiments/demo", "--max-cycles", "0"])

    def test_rejects_non_positive_hour_budget(self) -> None:
        parser = cli_parser()

        with pytest.raises(SystemExit):
            parser.parse_args(["resume", "experiments/demo", "--max-hours", "-1"])

    def test_accepts_positive_budgets(self) -> None:
        parser = cli_parser()

        args = parser.parse_args(
            ["start", "experiments/demo", "--max-cycles", "2", "--max-hours", "0.5"]
        )

        assert args.max_cycles == 2
        assert args.max_hours == 0.5


class TestResumeCommand:
    def test_resume_clears_stale_running_state_without_lock(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = {
            "experiment_id": "exp",
            "status": "running",
            "cycle_count": 0,
            "consecutive_no_progress_cycles": 0,
            "consecutive_failed_cycles": 0,
            "last_successful_cycle_id": None,
            "last_cycle_result": None,
            "max_cycles": 5,
            "max_hours": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": None,
            "active_cycle_id": "0001",
            "active_started_at": "2025-01-01T00:00:05+00:00",
            "active_objective": "something",
            "active_attempt": 1,
            "last_attempt_outcome": "attempt_01_running",
        }
        (d / "loop_state.json").write_text(json.dumps(state) + "\n")

        args = argparse.Namespace(
            experiment_path=str(d),
            max_cycles=5,
            max_hours=None,
        )

        captured_state: dict[str, object] = {}

        def fake_run_loop(_experiment_dir, current_state, **_kwargs):
            captured_state.update(current_state.to_dict())
            return current_state

        with (
            patch("loop.core.acquire_lock", return_value=d / LOCK_PATH_NAME),
            patch("loop.core.release_lock"),
            patch("loop.core.run_loop", side_effect=fake_run_loop),
            patch("loop.core.read_text", return_value="# status\n"),
        ):
            exit_code = resume_command(args)

        assert exit_code == 0
        assert captured_state["status"] == "idle"
        assert captured_state["active_cycle_id"] is None
        assert captured_state["active_started_at"] is None
        assert captured_state["active_objective"] is None
        assert captured_state["active_attempt"] is None
        assert captured_state["last_attempt_outcome"] == "interrupted_before_resume"

    def test_resume_clears_early_completion_when_budget_remains(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = {
            "experiment_id": "exp",
            "status": "completed",
            "cycle_count": 1,
            "consecutive_no_progress_cycles": 0,
            "consecutive_failed_cycles": 0,
            "last_successful_cycle_id": "0001",
            "last_cycle_result": "complete",
            "max_cycles": 10,
            "max_hours": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": "experiment_complete",
            "active_cycle_id": None,
            "active_started_at": None,
            "active_objective": None,
            "active_attempt": None,
            "last_attempt_outcome": "complete",
        }
        (d / "loop_state.json").write_text(json.dumps(state) + "\n")

        args = argparse.Namespace(
            experiment_path=str(d),
            max_cycles=10,
            max_hours=None,
        )

        captured_state: dict[str, object] = {}

        def fake_run_loop(_experiment_dir: Path, current_state, **_kwargs):
            captured_state.update(current_state.to_dict())
            return current_state

        with (
            patch("loop.core.acquire_lock", return_value=d / LOCK_PATH_NAME),
            patch("loop.core.release_lock"),
            patch("loop.core.run_loop", side_effect=fake_run_loop),
            patch("loop.core.read_text", return_value="# status\n"),
        ):
            exit_code = resume_command(args)

        assert exit_code == 0
        assert captured_state.get("last_cycle_result") is None

    def test_resume_keeps_completion_when_cycle_budget_exhausted(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = {
            "experiment_id": "exp",
            "status": "completed",
            "cycle_count": 10,
            "consecutive_no_progress_cycles": 0,
            "consecutive_failed_cycles": 0,
            "last_successful_cycle_id": "0010",
            "last_cycle_result": "complete",
            "max_cycles": 10,
            "max_hours": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": "experiment_complete",
            "active_cycle_id": None,
            "active_started_at": None,
            "active_objective": None,
            "active_attempt": None,
            "last_attempt_outcome": "complete",
        }
        (d / "loop_state.json").write_text(json.dumps(state) + "\n")

        args = argparse.Namespace(
            experiment_path=str(d),
            max_cycles=10,
            max_hours=None,
        )

        captured_state: dict[str, object] = {}

        def fake_run_loop(_experiment_dir: Path, current_state, **_kwargs):
            captured_state.update(current_state.to_dict())
            return current_state

        with (
            patch("loop.core.acquire_lock", return_value=d / LOCK_PATH_NAME),
            patch("loop.core.release_lock"),
            patch("loop.core.run_loop", side_effect=fake_run_loop),
            patch("loop.core.read_text", return_value="# status\n"),
        ):
            exit_code = resume_command(args)

        assert exit_code == 0
        assert captured_state.get("last_cycle_result") == "complete"

    def test_resume_clears_completion_when_hours_budget_set(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = {
            "experiment_id": "exp",
            "status": "completed",
            "cycle_count": 3,
            "consecutive_no_progress_cycles": 0,
            "consecutive_failed_cycles": 0,
            "last_successful_cycle_id": "0003",
            "last_cycle_result": "complete",
            "max_cycles": None,
            "max_hours": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": "experiment_complete",
            "active_cycle_id": None,
            "active_started_at": None,
            "active_objective": None,
            "active_attempt": None,
            "last_attempt_outcome": "complete",
        }
        (d / "loop_state.json").write_text(json.dumps(state) + "\n")

        args = argparse.Namespace(
            experiment_path=str(d),
            max_cycles=None,
            max_hours=2.0,
        )

        captured_state: dict[str, object] = {}

        def fake_run_loop(_experiment_dir: Path, current_state, **_kwargs):
            captured_state.update(current_state.to_dict())
            return current_state

        with (
            patch("loop.core.acquire_lock", return_value=d / LOCK_PATH_NAME),
            patch("loop.core.release_lock"),
            patch("loop.core.run_loop", side_effect=fake_run_loop),
            patch("loop.core.read_text", return_value="# status\n"),
        ):
            exit_code = resume_command(args)

        assert exit_code == 0
        assert captured_state.get("last_cycle_result") is None


class TestRunLoopHooks:
    def test_run_loop_uses_default_hooks_when_none_provided(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = initial_state(d, max_cycles=1, max_hours=None)
        captured_hooks: list[object] = []

        def fake_run_cycle(
            experiment_dir: Path,
            current_state,
            hooks: object | None = None,
        ) -> dict[str, object]:
            assert experiment_dir == d
            assert current_state.status == "running"
            captured_hooks.append(hooks)
            return {
                "cycle_id": "0001",
                "started_at": "2025-01-01T00:00:00+00:00",
                "completed_at": "2025-01-01T00:00:01+00:00",
                "result": "progress",
                "completion_marker": "CYCLE_DONE",
                "progress_reasons": ["journal_updated"],
                "attempts": [],
                "before_snapshot": {},
                "after_snapshot": {},
            }

        with (
            patch("loop.core.run_cycle", side_effect=fake_run_cycle),
            patch("loop.core.write_state"),
            patch("loop.core.emit_loop_start"),
            patch("loop.core.emit_loop_stop"),
            patch("loop.core.emit_cycle_result"),
        ):
            final_state = run_loop(d, state)

        assert len(captured_hooks) == 1
        assert isinstance(captured_hooks[0], DefaultCycleHooks)
        assert final_state.status == "completed"
        assert final_state.stop_reason == "max_cycles_reached"

    def test_run_loop_passes_custom_hooks_to_run_cycle(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = initial_state(d, max_cycles=1, max_hours=None)
        hooks = RecordingHooks()
        captured_hooks: list[object] = []

        def fake_run_cycle(
            experiment_dir: Path,
            current_state,
            hooks: object | None = None,
        ) -> dict[str, object]:
            assert experiment_dir == d
            captured_hooks.append(hooks)
            return {
                "cycle_id": "0001",
                "started_at": "2025-01-01T00:00:00+00:00",
                "completed_at": "2025-01-01T00:00:01+00:00",
                "result": "progress",
                "completion_marker": "CYCLE_DONE",
                "progress_reasons": ["custom:hook"],
                "attempts": [],
                "before_snapshot": {},
                "after_snapshot": {},
            }

        with (
            patch("loop.core.run_cycle", side_effect=fake_run_cycle),
            patch("loop.core.write_state"),
            patch("loop.core.emit_loop_start"),
            patch("loop.core.emit_loop_stop"),
            patch("loop.core.emit_cycle_result"),
        ):
            run_loop(d, state, hooks=hooks)

        assert captured_hooks == [hooks]
