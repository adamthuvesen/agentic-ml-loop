from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Any

from .constants import DEFAULT_MAX_ATTEMPTS_PER_CYCLE
from .prompts import latest_hypothesis, results_snapshot

_ANSI_RESET = "\033[0m"
_ANSI_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def _supports_ansi() -> bool:
    """Return True if stdout is a TTY and ``NO_COLOR`` is unset."""
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _style(text: str, *styles: str) -> str:
    """Wrap ``text`` in ANSI style codes (no-op if ANSI disabled)."""
    if not _supports_ansi():
        return text
    prefix = "".join(_ANSI_STYLES[s] for s in styles)
    return f"{prefix}{text}{_ANSI_RESET}"


def _muted(text: str) -> str:
    """Dim style helper."""
    return _style(text, "dim")


def _status_text(kind: str, text: str) -> str:
    """Color ``text`` by semantic kind (info, success, warning, error)."""
    palette: dict[str, tuple[str, ...]] = {
        "info": ("cyan", "bold"),
        "success": ("green", "bold"),
        "warning": ("yellow", "bold"),
        "error": ("red", "bold"),
    }
    return _style(text, *palette[kind])


def _print_header(title: str) -> None:
    """Print a boxed Unicode header for the given ``title``."""
    width = max(len(title) + 6, 34)
    bar = "━" * width
    pad = max((width - len(title)) // 2, 1)
    title_line = f"{' ' * pad}{title}".ljust(width)
    print()
    print(f"  {_muted('┏')}{_muted(bar)}{_muted('┓')}")
    print(f"  {_muted('┃')}{_style(title_line, 'bold')}{_muted('┃')}")
    print(f"  {_muted('┗')}{_muted(bar)}{_muted('┛')}")
    print()


def _print_section(label: str) -> None:
    """Print a section divider with ``label``."""
    rule_len = max(50 - len(label) - 2, 4)
    print()
    print(f"  {_muted('──')} {_style(label, 'bold')} {_muted('─' * rule_len)}")
    print()


def _print_kv(label: str, value: str) -> None:
    """Print a two-column label/value line (14-char label column)."""
    print(f"  {_muted(f'{label:<14}')}{value}")


def format_runner_label(state: dict[str, Any]) -> str:
    """Return a compact human-readable runner identity from loop state."""
    runner_label = str(state.get("runner_name") or "unknown")
    if state.get("runner_model"):
        runner_label = f"{runner_label} ({state['runner_model']}"
        if state.get("runner_effort"):
            runner_label = f"{runner_label}, effort={state['runner_effort']}"
        runner_label = f"{runner_label})"
    return runner_label


def format_elapsed(seconds: int) -> str:
    """Human-readable duration from a second count."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def iso_to_datetime(value: str) -> datetime:
    """Parse ISO format timestamp string (used for loop state times)."""
    return datetime.fromisoformat(value)


def _elapsed_str(started_at: str, completed_at: str) -> str:
    """Format elapsed time between two ISO timestamp strings."""
    secs = int((iso_to_datetime(completed_at) - iso_to_datetime(started_at)).total_seconds())
    return format_elapsed(secs)


class _LiveTimer:
    """Daemon thread that displays a live elapsed-time counter on the terminal."""

    def __init__(self, label: str = "Running") -> None:
        """Configure the counter label and internal thread state."""
        self._label = label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def start(self) -> None:
        """Start the daemon thread that redraws elapsed time every second."""
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Thread target: update one-line elapsed display until stopped."""
        while not self._stop_event.wait(1.0):
            elapsed = int(time.monotonic() - self._start_time)
            if _supports_ansi():
                text = (
                    f"  {_style(f'{self._label:<14}', 'dim')}"
                    f"{_style(format_elapsed(elapsed), 'cyan', 'bold')}"
                )
                sys.stdout.write(f"\r\033[2K{text}")
                sys.stdout.flush()

    def stop(self) -> int:
        """Stop the thread, clear the status line, return elapsed seconds."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        elapsed = int(time.monotonic() - self._start_time)
        if _supports_ansi():
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
        return elapsed


def emit_loop_start(experiment_dir: Any, state: dict[str, Any]) -> None:
    """Print config banner when a loop run begins (after state is ``running``)."""
    # First console output once run_loop has persisted status=running (see loop/core.py).
    _print_header("ML Research")
    _print_section("Config")
    _print_kv("Experiment", experiment_dir.name)
    _print_kv("Runner", format_runner_label(state))
    _print_kv(
        "Max cycles",
        "∞" if state.get("max_cycles") is None else str(state["max_cycles"]),
    )
    if state.get("max_hours"):
        _print_kv("Max hours", f"{state['max_hours']}h")
    if state.get("enforce_budget_until_limit"):
        _print_kv(
            "Budget mode",
            "run until limit (ignores EXPERIMENT_COMPLETE until cap)",
        )
    _print_kv("Status", f"{_status_text('info', 'Starting')} research loop")


def emit_cycle_start(cycle_id: str, prior_plan: str) -> None:
    """Print cycle header and the handoff context (the prior cycle's plan).

    The current cycle's *own* hypothesis is unknown at start time — it lands in
    the journal during execution and is printed by ``emit_cycle_result``.
    Labeling this line "Prior plan" prevents the off-by-one confusion where
    cycle N's start banner shows cycle N-1's heading.
    """
    _print_section(f"Cycle {cycle_id}")
    _print_kv("Prior plan", prior_plan)


def emit_attempt_start(attempt: int) -> None:
    """Print current attempt number within the cycle."""
    _print_kv("Attempt", f"{attempt}/{DEFAULT_MAX_ATTEMPTS_PER_CYCLE}")


def emit_cycle_retry(next_attempt: int, attempt_record: dict[str, Any]) -> None:
    """Print why the previous attempt failed before retrying."""
    reasons: list[str] = []
    if attempt_record.get("contract_errors"):
        reasons.extend(str(reason) for reason in attempt_record["contract_errors"][:3])
    elif attempt_record.get("validation_errors"):
        reasons.append("validation errors")
    elif attempt_record.get("journal_updated") is False:
        reasons.append("journal not updated")
    elif not attempt_record.get("marker"):
        reasons.append("missing marker")
    reason_str = ", ".join(reasons) if reasons else attempt_record.get("failure_reason", "unknown")
    failed = next_attempt - 1
    _print_kv("Status", f"{_status_text('error', 'x')} attempt {failed} failed: {reason_str}")


def emit_cycle_result(experiment_dir: Any, summary: dict[str, Any]) -> None:
    """Print cycle outcome, timing, new candidates, and validation errors if any."""
    result = summary["result"]
    elapsed = _elapsed_str(summary["started_at"], summary["completed_at"])

    if result in {"progress", "complete"}:
        hypothesis = latest_hypothesis(experiment_dir)
        if hypothesis:
            _print_kv("Hypothesis", hypothesis)
        label = "experiment complete" if result == "complete" else "progress"
        _print_kv(
            "Result",
            f"{_status_text('success', '✓')} {_status_text('success', label)}  {_muted(f'({elapsed})')}",
        )
        lb = summary.get("after_snapshot", {}).get("results_by_id", {})
        before_scores = [
            c["objective_score"]
            for c in summary.get("before_snapshot", {}).get("results_by_id", {}).values()
            if isinstance(c.get("objective_score"), (int, float))
        ]
        after_scores = [
            c["objective_score"]
            for c in lb.values()
            if isinstance(c.get("objective_score"), (int, float))
        ]
        before_best = max(before_scores) if before_scores else None
        after_best = max(after_scores) if after_scores else None
        if before_best is not None and after_best is not None:
            if after_best > before_best + 0.001:
                delta_icon = " ↑"
            elif after_best < before_best - 0.001:
                delta_icon = " ↓"
            else:
                delta_icon = ""
        else:
            delta_icon = ""
        new_lines: list[str] = []
        for reason in summary.get("progress_reasons", []):
            if reason.startswith("new_candidates:"):
                for cid in reason.split(":", 1)[1].split(", "):
                    cid = cid.strip()
                    entry = lb.get(cid, {})
                    score = entry.get("objective_score")
                    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
                    new_lines.append(
                        f"{cid}  {entry.get('model_family', '?')}"
                        f"  {entry.get('objective_metric', '?')} {score_str}{delta_icon}"
                    )
        if new_lines:
            _print_kv("New", new_lines[0])
            for line in new_lines[1:]:
                _print_kv("", line)
        other_reasons = [
            r for r in summary.get("progress_reasons", []) if not r.startswith("new_candidates:")
        ]
        if other_reasons:
            _print_kv("Changes", ", ".join(other_reasons))

    elif result == "no_progress":
        _print_kv(
            "Result",
            f"{_status_text('warning', '!')} {_status_text('warning', 'no progress')}  {_muted(f'({elapsed})')}",
        )

    else:  # failed
        _print_kv("Result", f"{_status_text('error', 'x')} {_status_text('error', 'failed')}")
        for attempt in summary.get("attempts", [])[-1:]:
            for err in (attempt.get("validation_errors") or [])[:2]:
                _print_kv("Error", _muted(err))


def emit_loop_stop(experiment_dir: Any, state: dict[str, Any]) -> None:
    """Print final summary when the loop exits (stop reason, best score)."""
    reason = state.get("stop_reason", "unknown")
    cycle_count = state.get("cycle_count", 0)
    status = state.get("status", "")
    _print_section("Summary")
    _print_kv("Cycles", str(cycle_count))
    kind = "success" if status == "completed" else ("warning" if status == "stalled" else "error")
    _print_kv("Stop reason", f"{_status_text(kind, reason)}")
    results = results_snapshot(experiment_dir)
    if results:
        best = max(results, key=lambda x: x.get("objective_score", float("-inf")))
        score = best.get("objective_score")
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "?"
        _print_kv(
            "Best",
            f"{best.get('candidate_id', '?')}  {best.get('objective_metric', '?')} {score_str}",
        )
    print()
