from __future__ import annotations

import re


VALID_COMPLETION_MARKERS = frozenset({"CYCLE_DONE", "EXPERIMENT_COMPLETE"})


def extract_completion_marker(output_text: str) -> tuple[str, list[str]]:
    """Return the sole valid completion marker plus validation errors."""
    markers = [
        marker.strip()
        for marker in re.findall(r"<promise>([^<]+)</promise>", output_text)
    ]
    if not markers:
        return "", ["missing completion marker"]
    if len(markers) > 1:
        return "", [f"expected exactly one completion marker, found {len(markers)}"]
    marker = markers[0]
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
