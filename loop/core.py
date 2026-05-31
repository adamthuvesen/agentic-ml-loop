from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from experiment import journal_path
from lib.utils import load_json, read_text, utc_now, write_json, write_text

from .artifacts import (
    artifact_snapshot,
    capture_cycle_baselines,
    restore_cycle_baselines,
)
from .artifacts import (
    compute_progress as compute_progress,  # noqa: F401 - compatibility export
)
from .artifacts import (
    restore_artifacts as _restore_artifacts,  # noqa: F401 - re-exported for tests
)
from .constants import DEFAULT_MAX_ATTEMPTS_PER_CYCLE, STATE_PATH_NAME
from .contracts import (  # noqa: F401 - re-exported for tests
    cycle_contract_errors,
    extract_completion_marker,
)
from .cycle_attempt import run_cycle_attempt
from .hooks import CycleHooks, DefaultCycleHooks
from .invoke import (
    RunnerConfig,
    add_runner_arguments,
    build_runner_config,
    default_runner_config,
)
from .loop_state import LoopState, load_state, write_state_file
from .prompts import latest_hypothesis
from .status import build_status_markdown
from .stop_policy import should_stop
from .ui import (
    _LiveTimer,
    emit_attempt_start,
    emit_cycle_result,
    emit_cycle_retry,
    emit_cycle_start,
    emit_loop_start,
    emit_loop_stop,
)

ARTIFACT_FILES = (
    "experiment.md",
    "research_journal.md",
)
MUTABLE_ARTIFACT_FILES = ("research_journal.md",)
LOCK_PATH_NAME = ".loop.lock"
STATUS_PATH_NAME = "status.md"


def ensure_experiment_directory(experiment_dir: Path) -> None:
    """Raise if ``experiment_dir`` is missing required experiment files."""
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    if not (experiment_dir / "experiment.md").exists():
        raise FileNotFoundError(f"Missing experiment.md in {experiment_dir}")
    if not journal_path(experiment_dir).exists():
        raise FileNotFoundError(f"Missing research_journal.md in {experiment_dir}")


def active_lock_pid(lock_path: Path) -> int | None:
    """Return the PID from the lock file if that process is still alive, else ``None``."""
    if not lock_path.exists():
        return None
    try:
        payload = load_json(lock_path)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    pid = payload.get("pid")
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)  # POSIX: signal 0 only checks whether the process exists
    except OSError:
        return None
    return pid


def acquire_lock(experiment_dir: Path) -> Path:
    """Ensure only one loop process runs per experiment directory.

    Creates ``experiment_dir/.loop.lock`` with ``O_CREAT|O_EXCL``. If the file
    already exists, checks whether the recorded PID is still alive; if another
    live process holds the lock, raises; if the lock is stale (crashed runner),
    unlinks and retries a few times. Call `release_lock` in a ``finally`` when
    the supervisor exits.
    """
    lock_path = experiment_dir / LOCK_PATH_NAME
    lock_data = (
        json.dumps(
            {
                "pid": os.getpid(),
                "acquired_at": utc_now(),
                "run_dir": str(experiment_dir),
            },
            indent=2,
        )
        + "\n"
    )
    # Stale lock: prior run may have crashed without unlinking; reclaim if PID is dead.
    for _ in range(3):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, lock_data.encode("utf-8"))
            finally:
                os.close(fd)
            return lock_path
        except FileExistsError as err:
            active_pid = active_lock_pid(lock_path)
            if active_pid and active_pid != os.getpid():
                raise RuntimeError(
                    f"Loop is already running for {experiment_dir} under pid {active_pid}. "
                    f"Use 'status' or stop that process first."
                ) from err
            lock_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"Could not reclaim stale loop lock for {experiment_dir}. "
        "Try again once competing start/resume commands have exited."
    )


def release_lock(lock_path: Path) -> None:
    """Unlink ``lock_path`` only when its stored PID matches this process.

    Avoids deleting a lock written by a different PID (e.g. race or manual edit).
    """
    if not lock_path.exists():
        return
    try:
        payload = load_json(lock_path)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and payload.get("pid") != os.getpid():
        return
    lock_path.unlink(missing_ok=True)


def write_state(experiment_dir: Path, state: LoopState) -> None:
    """Persist loop state and refresh ``status.md``."""
    state.updated_at = utc_now()
    write_state_file(experiment_dir, state)
    write_status_markdown(experiment_dir, state)


def write_status_markdown(experiment_dir: Path, state: LoopState | dict[str, Any]) -> None:
    """Write human-readable loop status and top results to ``status.md``."""
    if isinstance(state, dict):
        state = LoopState.from_dict(state)
    has_active_lock = (
        active_lock_pid(experiment_dir / LOCK_PATH_NAME) is not None
        if state.status == "running"
        else True
    )
    write_text(
        experiment_dir / STATUS_PATH_NAME,
        build_status_markdown(
            experiment_dir,
            state,
            has_active_lock=has_active_lock,
        ),
    )


def initial_state(
    experiment_dir: Path,
    max_cycles: int | None,
    max_hours: float | None,
    *,
    enforce_budget_until_limit: bool = False,
    runner_config: RunnerConfig | None = None,
) -> LoopState:
    """Return default state for a new ``loop start`` (status ``idle``)."""
    now = utc_now()
    runner_config = runner_config or default_runner_config()
    return LoopState(
        experiment_id=experiment_dir.name,
        status="idle",
        runner_name=runner_config.name,
        runner_model=runner_config.model,
        runner_effort=runner_config.effort,
        runner_command=runner_config.command,
        runner_timeout_seconds=runner_config.timeout_seconds,
        cycle_count=0,
        consecutive_no_progress_cycles=0,
        consecutive_failed_cycles=0,
        last_successful_cycle_id=None,
        last_cycle_result=None,
        max_cycles=max_cycles,
        max_hours=max_hours,
        enforce_budget_until_limit=enforce_budget_until_limit,
        started_at=now,
        updated_at=now,
        stop_reason=None,
    )


def _positive_int(value: str) -> int:
    """Parse a positive CLI integer."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    """Parse a positive CLI float."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _runner_config_from_args(args: argparse.Namespace) -> RunnerConfig:
    return build_runner_config(
        runner=getattr(args, "runner", "claude"),
        runner_command=getattr(args, "runner_command", None),
        runner_model=getattr(args, "runner_model", None),
        runner_effort=getattr(args, "runner_effort", None),
        runner_timeout=getattr(args, "runner_timeout", 1800),
    )


def _clear_active_state(state: LoopState, outcome: str) -> None:
    """Clear in-progress cycle fields and set ``last_attempt_outcome``."""
    state.active_cycle_id = None
    state.active_started_at = None
    state.active_objective = None
    state.active_attempt = None
    state.last_attempt_outcome = outcome


def run_cycle(
    experiment_dir: Path,
    state: LoopState,
    hooks: CycleHooks | None = None,
) -> dict[str, Any]:
    """Run a single ML research cycle: prompt, attempts, validation, summary."""
    if hooks is None:
        hooks = DefaultCycleHooks()
    cycle_id = f"{state.cycle_count + 1:04d}"
    current_cycle_dir = experiment_dir / "cycles" / cycle_id
    current_cycle_dir.mkdir(parents=True, exist_ok=True)
    cycle_started_at = utc_now()

    pre = hooks.pre_cycle(experiment_dir, cycle_id, state.to_dict())
    prompt_text = pre.prompt_text
    (current_cycle_dir / "prompt.md").write_text(prompt_text, encoding="utf-8")
    agent_message_path = current_cycle_dir / "agent_last_message.md"

    baselines = capture_cycle_baselines(experiment_dir)
    before_snapshot = baselines.before_snapshot

    state.active_cycle_id = cycle_id
    state.active_started_at = cycle_started_at
    state.active_objective = latest_hypothesis(experiment_dir) or "(fresh start — no prior plan)"
    state.active_attempt = None
    state.last_attempt_outcome = None
    write_state(experiment_dir, state)
    emit_cycle_start(cycle_id, state.active_objective)

    attempt_records: list[dict[str, Any]] = []

    try:
        for attempt in range(1, DEFAULT_MAX_ATTEMPTS_PER_CYCLE + 1):
            restore_cycle_baselines(experiment_dir, baselines)

            state.active_attempt = attempt
            state.last_attempt_outcome = f"attempt_{attempt:02d}_running"
            write_state(experiment_dir, state)
            emit_attempt_start(attempt)

            attempt_stdout = current_cycle_dir / f"attempt_{attempt:02d}_stdout.log"
            attempt_stderr = current_cycle_dir / f"attempt_{attempt:02d}_stderr.log"
            attempt_meta = current_cycle_dir / f"attempt_{attempt:02d}_meta.json"
            attempt_result_path = current_cycle_dir / f"attempt_{attempt:02d}_result.json"

            timer = _LiveTimer("Running")
            timer.start()
            outcome = run_cycle_attempt(
                experiment_dir=experiment_dir,
                cycle_id=cycle_id,
                attempt=attempt,
                prompt_text=prompt_text,
                baselines_journal_hash=before_snapshot["journal_hash"],
                experiment_md_backup=baselines.experiment_md_backup,
                attempt_stdout=attempt_stdout,
                attempt_stderr=attempt_stderr,
                attempt_meta=attempt_meta,
                attempt_result_path=attempt_result_path,
                agent_message_path=agent_message_path,
                runner_config=RunnerConfig(
                    name=state.runner_name,
                    command=state.runner_command,
                    timeout_seconds=state.runner_timeout_seconds,
                    model=state.runner_model,
                    effort=state.runner_effort,
                ),
            )
            timer.stop()

            if not outcome.success:
                attempt_records.append(outcome.attempt_record)
                if attempt < DEFAULT_MAX_ATTEMPTS_PER_CYCLE:
                    emit_cycle_retry(attempt + 1, outcome.attempt_record)
                continue

            assert outcome.summary is not None
            after_snapshot = artifact_snapshot(experiment_dir)
            post = hooks.post_cycle(
                experiment_dir,
                cycle_id,
                before_snapshot,
                after_snapshot,
                outcome.summary["output_text"],
                outcome.marker,
            )
            progress_reasons = post.progress_reasons
            result = "progress" if progress_reasons else "no_progress"
            if outcome.marker == "EXPERIMENT_COMPLETE":
                result = "complete"

            summary = {
                "cycle_id": cycle_id,
                "started_at": cycle_started_at,
                "completed_at": utc_now(),
                "result": result,
                "completion_marker": outcome.marker,
                "progress_reasons": progress_reasons,
                "validation_warnings": outcome.summary["validation_warnings"],
                "attempts": attempt_records,
                "before_snapshot": before_snapshot,
                "after_snapshot": after_snapshot,
            }
            write_json(current_cycle_dir / "cycle_summary.json", summary)
            _clear_active_state(state, result)
            return summary

    except BaseException:
        _clear_active_state(state, "exception")
        raise

    restore_cycle_baselines(experiment_dir, baselines)

    summary = {
        "cycle_id": cycle_id,
        "started_at": cycle_started_at,
        "completed_at": utc_now(),
        "result": "failed",
        "failure_reason": "too_many_failed_attempts",
        "attempts": attempt_records,
        "before_snapshot": before_snapshot,
        "after_snapshot": before_snapshot,
    }
    write_json(current_cycle_dir / "cycle_summary.json", summary)
    _clear_active_state(state, "too_many_failed_attempts")
    return summary


def apply_cycle_result(state: LoopState, summary: dict[str, Any]) -> None:
    """Fold one `run_cycle` summary into persistent loop state.

    Always increments ``cycle_count``. ``progress`` / ``complete`` resets both stall
    and failure streaks; ``no_progress`` increments ``consecutive_no_progress_cycles``;
    ``failed`` increments ``consecutive_failed_cycles`` and resets the stall counter.
    """
    state.cycle_count += 1
    result = summary["result"]
    state.last_cycle_result = result
    if result in {"progress", "complete"}:
        state.consecutive_no_progress_cycles = 0
        state.consecutive_failed_cycles = 0
        state.last_successful_cycle_id = summary["cycle_id"]
    elif result == "no_progress":
        state.consecutive_no_progress_cycles += 1
        state.consecutive_failed_cycles = 0
    else:
        state.consecutive_failed_cycles += 1
        state.consecutive_no_progress_cycles = 0


def run_loop(
    experiment_dir: Path,
    state: LoopState,
    hooks: CycleHooks | None = None,
) -> LoopState:
    """Main supervisor: mark running, loop until `should_stop` or Ctrl+C.

    Each iteration checks stop conditions (limits, completion, stall/failure caps),
    then runs `run_cycle`, applies `apply_cycle_result`, persists state, and prints
    UI. Keyboard interrupt sets status ``stopped`` and clears active-cycle fields.
    """
    if hooks is None:
        hooks = DefaultCycleHooks()
    state.status = "running"
    state.stop_reason = None
    write_state(experiment_dir, state)
    emit_loop_start(experiment_dir, state.to_dict())

    try:
        while True:
            should_end, reason, final_status = should_stop(state)
            if should_end:
                state.status = final_status
                state.stop_reason = reason
                write_state(experiment_dir, state)
                emit_loop_stop(experiment_dir, state.to_dict())
                return state

            summary = run_cycle(experiment_dir, state, hooks=hooks)
            apply_cycle_result(state, summary)
            write_state(experiment_dir, state)
            emit_cycle_result(experiment_dir, summary)

    except KeyboardInterrupt:
        state.status = "stopped"
        state.stop_reason = "interrupted"
        _clear_active_state(state, "interrupted")
        write_state(experiment_dir, state)
        emit_loop_stop(experiment_dir, state.to_dict())
        return state


def print_status(experiment_dir: Path) -> int:
    """Print ``status.md`` for the experiment, or a message if no loop state exists."""
    sp = experiment_dir / STATE_PATH_NAME
    if sp.exists():
        state = load_state(experiment_dir)
        write_status_markdown(experiment_dir, state)
        print(read_text(experiment_dir / STATUS_PATH_NAME))
        return 0
    print(f"No loop state found for {experiment_dir}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    """CLI: ``start``, ``resume``, and ``status`` subcommands."""
    parser = argparse.ArgumentParser(description="Long-running model-search loop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("start", "resume"):
        subparser = subparsers.add_parser(command_name)
        subparser.add_argument("experiment_path")
        subparser.add_argument("--max-cycles", type=_positive_int, default=None)
        subparser.add_argument("--max-hours", type=_positive_float, default=None)
        subparser.add_argument(
            "--run-until-limit",
            action="store_true",
            help=(
                "Keep running until --max-cycles or --max-hours is reached; do not "
                "stop early when the runner emits EXPERIMENT_COMPLETE. Requires at "
                "least one of --max-cycles or --max-hours."
            ),
        )
        add_runner_arguments(subparser)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("experiment_path")
    return parser


def start_command(args: argparse.Namespace) -> int:
    """Begin a new loop run (requires no existing ``loop_state.json``)."""
    experiment_dir = Path(args.experiment_path).resolve()
    ensure_experiment_directory(experiment_dir)

    sp = experiment_dir / STATE_PATH_NAME
    if sp.exists():
        # Fresh start requires no loop_state.json; otherwise use resume to continue.
        print(f"Loop state already exists for {experiment_dir}. Use 'resume' instead.")
        return 1

    if (
        getattr(args, "run_until_limit", False)
        and args.max_cycles is None
        and (args.max_hours is None)
    ):
        print(
            "error: --run-until-limit requires at least one of --max-cycles or --max-hours",
            file=sys.stderr,
        )
        return 1

    state = initial_state(
        experiment_dir,
        args.max_cycles,
        args.max_hours,
        enforce_budget_until_limit=getattr(args, "run_until_limit", False),
        runner_config=_runner_config_from_args(args),
    )

    lock_path = acquire_lock(experiment_dir)
    try:
        run_loop(experiment_dir, state)
    finally:
        release_lock(lock_path)  # always clear lock when run_loop returns or raises

    print(read_text(experiment_dir / STATUS_PATH_NAME))
    return 0


def resume_command(args: argparse.Namespace) -> int:
    """Continue an existing loop from ``loop_state.json``."""
    experiment_dir = Path(args.experiment_path).resolve()
    ensure_experiment_directory(experiment_dir)

    sp = experiment_dir / STATE_PATH_NAME
    if not sp.exists():
        print(f"No loop state found for {experiment_dir}. Use 'start' first.")
        return 1

    state = load_state(experiment_dir)

    if state.status == "running" and active_lock_pid(experiment_dir / LOCK_PATH_NAME) is None:
        state.status = "idle"
        _clear_active_state(state, "interrupted_before_resume")

    if args.max_cycles is not None:
        state.max_cycles = args.max_cycles
    if args.max_hours is not None:
        state.max_hours = args.max_hours
    runner_config = _runner_config_from_args(args)
    state.runner_name = runner_config.name
    state.runner_model = runner_config.model
    state.runner_effort = runner_config.effort
    state.runner_command = runner_config.command
    state.runner_timeout_seconds = runner_config.timeout_seconds

    if getattr(args, "run_until_limit", False):
        effective_mc = args.max_cycles if args.max_cycles is not None else state.max_cycles
        effective_mh = args.max_hours if args.max_hours is not None else state.max_hours
        if effective_mc is None and effective_mh is None:
            print(
                "error: --run-until-limit requires at least one of --max-cycles or "
                "--max-hours (in args or existing state)",
                file=sys.stderr,
            )
            return 1
        state.enforce_budget_until_limit = True

    if state.last_cycle_result == "complete":
        cycle_budget_has_room = (
            state.max_cycles is not None and state.cycle_count < state.max_cycles
        )
        has_time_budget = state.max_hours is not None
        if cycle_budget_has_room or has_time_budget:
            state.last_cycle_result = None

    lock_path = acquire_lock(experiment_dir)
    try:
        run_loop(experiment_dir, state)
    finally:
        release_lock(lock_path)  # always clear lock when run_loop returns or raises

    print(read_text(experiment_dir / STATUS_PATH_NAME))
    return 0


def main() -> int:
    """Entry point: dispatch ``start`` / ``resume`` / ``status``."""
    signal.signal(signal.SIGTERM, signal.default_int_handler)  # align SIGTERM with Ctrl+C
    args = build_parser().parse_args()
    if args.command == "start":
        return start_command(args)
    if args.command == "resume":
        return resume_command(args)
    if args.command == "status":
        return print_status(Path(args.experiment_path).resolve())
    raise ValueError(f"Unknown command: {args.command}")
