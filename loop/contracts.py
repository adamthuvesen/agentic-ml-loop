from __future__ import annotations

import json
from pathlib import Path

VALID_COMPLETION_MARKERS = frozenset({"CYCLE_DONE", "EXPERIMENT_COMPLETE"})


def read_completion_marker(status_path: Path) -> tuple[str, list[str]]:
    """Return the cycle's completion marker plus validation errors.

    The marker is a file the agent writes, not a token matched out of its prose.
    Scanning stdout meant an agent that quoted the marker while reasoning about
    it failed its own cycle, and the old "exactly one match" rule existed only
    to blunt that. The supervisor clears this file before every attempt, so its
    contents can only come from the attempt being judged.
    """
    if not status_path.exists():
        return "", [f"missing completion status file: {status_path.name}"]
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "", [f"{status_path.name} is not readable JSON: {exc}"]
    if not isinstance(payload, dict):
        return "", [f"{status_path.name} must hold a JSON object"]

    marker = str(payload.get("status", "")).strip()
    if not marker:
        return "", [f"{status_path.name} needs a non-empty `status`"]
    if marker not in VALID_COMPLETION_MARKERS:
        return marker, [f"unknown completion marker: {marker}"]
    return marker, []


def actionable_validation_errors(validation_errors: list[str]) -> list[str]:
    """Return validation messages that should fail the cycle contract."""
    return [error for error in validation_errors if not error.startswith("warning:")]


def validation_warnings(validation_errors: list[str]) -> list[str]:
    """Return soft validation warnings from a mixed error list."""
    return [error for error in validation_errors if error.startswith("warning:")]


def cycle_contract_errors(
    *,
    returncode: int,
    marker_errors: list[str],
    validation_errors: list[str],
    journal_updated: bool,
    experiment_md_changed: bool,
) -> list[str]:
    """Return user-facing contract errors for a failed cycle attempt."""
    errors: list[str] = []
    seen: set[str] = set()

    def add(error: str) -> None:
        if error not in seen:
            errors.append(error)
            seen.add(error)

    if returncode != 0:
        add(f"runner exited with return code {returncode}")
    for error in marker_errors:
        add(error)
    for error in actionable_validation_errors(validation_errors):
        add(error)
    if not journal_updated:
        add("research_journal.md was not updated")
    if experiment_md_changed:
        add("experiment.md changed during the cycle")
    return errors
