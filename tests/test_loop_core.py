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
    final_holdout_command,
    freeze_command,
    ledger_command,
    release_lock,
    resume_command,
)
from loop.core import (
    cli_parser,
    initial_state,
    run_cycle,
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

    def test_final_holdout_access_stops_even_in_run_until_limit(self) -> None:
        should_end, reason, _ = should_stop(
            self._minimal_state(
                last_cycle_result="complete",
                cycle_count=3,
                max_cycles=10,
                enforce_budget_until_limit=True,
                final_holdout_accessed=True,
            )
        )
        assert should_end
        assert reason == "final_holdout_accessed"

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

    def test_exposes_freeze_final_holdout_and_ledger_subcommands(self) -> None:
        parser = cli_parser()

        freeze = parser.parse_args(["freeze", "experiments/demo", "--candidate", "candidate-a"])
        holdout = parser.parse_args(["final-holdout", "experiments/demo"])
        ledger = parser.parse_args(["ledger", "experiments/demo"])

        assert freeze.command == "freeze"
        assert holdout.command == "final-holdout"
        assert ledger.command == "ledger"


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


class TestFreezeAndFinalHoldoutCommands:
    def _write_state(self, experiment_dir: Path, **overrides: object) -> None:
        state = {
            "experiment_id": experiment_dir.name,
            "status": "completed",
            "cycle_count": 3,
            "consecutive_no_progress_cycles": 0,
            "consecutive_failed_cycles": 0,
            "last_successful_cycle_id": "0003",
            "last_cycle_result": "complete",
            "max_cycles": 10,
            "max_hours": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": "experiment_complete",
        }
        state.update(overrides)
        (experiment_dir / "loop_state.json").write_text(json.dumps(state) + "\n")

    def test_freeze_records_frozen_candidates(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "candidate-a", "objective_score": 0.7}],
        )
        self._write_state(d)

        args = argparse.Namespace(
            experiment_path=str(d),
            candidate=["candidate-a"],
            reason="validation winner",
            force=False,
        )

        assert freeze_command(args) == 0
        state = json.loads((d / "loop_state.json").read_text())
        assert state["selection_frozen"] is True
        assert state["frozen_candidate_ids"] == ["candidate-a"]
        assert state["frozen_at_cycle"] == "0003"
        assert state["freeze_reason"] == "validation winner"

    def test_final_holdout_writes_separate_artifact_and_preserves_results(
        self,
        tmp_path: Path,
    ) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "candidate-a", "objective_score": 0.7}],
        )
        self._write_state(
            d,
            selection_frozen=True,
            frozen_candidate_ids=["candidate-a"],
            frozen_at_cycle="0003",
            freeze_reason="validation winner",
            final_holdout_accessed=False,
        )
        before_results = (d / "results.json").read_text()
        args = argparse.Namespace(experiment_path=str(d), candidate=None, force=False)
        final_payload = [
            {
                "candidate_id": "candidate-a",
                "objective_score": 0.7,
                "metrics": {
                    "validation": {"score": 0.7},
                    "test": {"score": 0.65},
                },
                "hyperparameters": {"final_holdout_included": True},
            }
        ]

        with patch("loop.core.run_final_holdout_candidates", return_value=final_payload):
            assert final_holdout_command(args) == 0

        assert (d / "results.json").read_text() == before_results
        artifact = json.loads((d / "outputs" / "final_holdout.json").read_text())
        assert artifact["scored_candidate_ids"] == ["candidate-a"]
        assert artifact["candidates"] == final_payload
        state = json.loads((d / "loop_state.json").read_text())
        assert state["final_holdout_accessed"] is True
        assert state["stop_reason"] == "final_holdout_accessed"
        assert state["final_holdout_path"].endswith("outputs/final_holdout.json")


class TestLedgerCommand:
    def test_ledger_rebuild_is_deterministic(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        cycle_dir = d / "cycles" / "0001"
        cycle_dir.mkdir(parents=True)
        summary = {
            "cycle_id": "0001",
            "started_at": "2025-01-01T00:00:00+00:00",
            "completed_at": "2025-01-01T00:00:02+00:00",
            "result": "progress",
            "completion_marker": "CYCLE_DONE",
            "progress_reasons": ["new_candidates:candidate-a"],
            "attempts": [{"attempt": 1}],
            "before_snapshot": {"results_by_id": {}},
            "after_snapshot": {
                "results_by_id": {
                    "candidate-a": {
                        "candidate_id": "candidate-a",
                        "objective_score": 0.7,
                        "hyperparameters": {"runtime_seconds": 1.25, "cost_usd": 0.0},
                    }
                }
            },
        }
        (cycle_dir / "cycle_summary.json").write_text(json.dumps(summary) + "\n")

        args = argparse.Namespace(experiment_path=str(d))
        assert ledger_command(args) == 0
        first = (d / "outputs" / "cycle_metrics.csv").read_text()
        assert ledger_command(args) == 0
        second = (d / "outputs" / "cycle_metrics.csv").read_text()

        assert first == second
        assert "candidate-a" in first
        assert "1.25" in first


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


class TestRunCycleRollback:
    def test_unexpected_exception_restores_artifacts_and_persists_cleared_state(
        self, tmp_path: Path
    ) -> None:
        d = _make_experiment(tmp_path)
        original_journal = (d / "research_journal.md").read_text()
        state = initial_state(d, max_cycles=1, max_hours=None)

        def crash_after_mutation(**_kwargs: object) -> object:
            (d / "research_journal.md").write_text("partial attempt edit\n")
            raise RuntimeError("post-mutation crash")

        with (
            patch("loop.core.run_cycle_attempt", side_effect=crash_after_mutation),
            patch("loop.core.emit_cycle_start"),
            patch("loop.core.emit_attempt_start"),
            pytest.raises(RuntimeError, match="post-mutation crash"),
        ):
            run_cycle(d, state)

        assert (d / "research_journal.md").read_text() == original_journal
        persisted = json.loads((d / "loop_state.json").read_text())
        assert persisted["active_cycle_id"] is None
        assert persisted["active_started_at"] is None
        assert persisted["active_objective"] is None
        assert persisted["active_attempt"] is None
        assert persisted["last_attempt_outcome"] == "exception"
