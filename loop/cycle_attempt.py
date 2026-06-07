"""Single cycle attempt: runner invoke, validation, and contract checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from experiment import (
    count_journal_cycles,
    get_min_cycles_before_complete,
    journal_path,
    validate_experiment,
)
from lib.io import read_text, utc_now, write_json

from .artifacts import sha256_text
from .contracts import (
    cycle_contract_errors,
    extract_completion_marker,
    validation_warnings,
)
from .invoke import RunnerConfig, default_runner_config, invoke_runner

logger = logging.getLogger(__name__)


class CycleSummary(TypedDict):
    """Successful-attempt payload consumed by ``run_cycle``."""

    output_text: str
    validation_warnings: list[str]
    journal_updated: bool
    experiment_md_changed: bool
    returncode: int


class AttemptRecord(TypedDict, total=False):
    """Failure record persisted per attempt (keys vary by failure kind)."""

    attempt: int
    returncode: int
    failure_reason: str
    error_message: str
    contract_errors: list[str]
    marker_errors: list[str]
    validation_errors: list[str]
    validation_warnings: list[str]
    journal_updated: bool
    experiment_md_changed: bool
    marker: str


@dataclass
class AttemptOutcome:
    success: bool
    marker: str
    summary: CycleSummary | None
    attempt_record: AttemptRecord


def run_cycle_attempt(
    *,
    experiment_dir: Path,
    cycle_id: str,
    attempt: int,
    prompt_text: str,
    baselines_journal_hash: str,
    experiment_md_backup: str,
    attempt_stdout: Path,
    attempt_stderr: Path,
    attempt_meta: Path,
    attempt_result_path: Path,
    agent_message_path: Path,
    runner_config: RunnerConfig | None = None,
) -> AttemptOutcome:
    """Run one runner attempt and return success or a failure record."""
    runner_config = runner_config or default_runner_config()
    agent_message_path.unlink(missing_ok=True)

    write_json(
        attempt_meta,
        {
            "attempt": attempt,
            "cycle_id": cycle_id,
            "started_at": utc_now(),
            "stdout_path": str(attempt_stdout.resolve()),
            "stderr_path": str(attempt_stderr.resolve()),
            "agent_message_path": str(agent_message_path.resolve()),
            "runner_name": runner_config.name,
            "runner_command": runner_config.command,
            "timeout_seconds": runner_config.timeout_seconds,
        },
    )

    try:
        runner_result = invoke_runner(
            prompt_text,
            attempt_stdout,
            attempt_stderr,
            agent_message_path,
            runner_config,
        )
    except Exception as exc:
        logger.warning(
            "Runner invocation failed for cycle %s attempt %s: %s",
            cycle_id,
            attempt,
            exc,
            exc_info=True,
        )
        attempt_record: AttemptRecord = {
            "attempt": attempt,
            "failure_reason": f"runner_invocation_error:{type(exc).__name__}",
            "error_message": str(exc),
        }
        write_json(attempt_result_path, attempt_record)
        return AttemptOutcome(
            success=False,
            marker="",
            summary=None,
            attempt_record=attempt_record,
        )

    try:
        output_text = (
            read_text(Path(runner_result["agent_message_path"]))
            if runner_result["agent_message_path"]
            else read_text(attempt_stdout)
        )
    except FileNotFoundError:
        output_text = ""

    marker, marker_errors = extract_completion_marker(output_text)

    validation_errors = validate_experiment(
        experiment_dir,
        strict_completion=(marker == "EXPERIMENT_COMPLETE"),
    )
    min_before_complete = get_min_cycles_before_complete(experiment_dir)
    if marker == "EXPERIMENT_COMPLETE" and min_before_complete is not None:
        jc = count_journal_cycles(experiment_dir)
        if jc < min_before_complete:
            validation_errors = [
                *validation_errors,
                (
                    f"experiment.md requires at least {min_before_complete} "
                    f"`## Cycle NNNN:` journal entries before EXPERIMENT_COMPLETE "
                    f"(found {jc})."
                ),
            ]

    journal_file = journal_path(experiment_dir)
    current_journal_text = read_text(journal_file) if journal_file.exists() else ""
    journal_updated = sha256_text(current_journal_text) != baselines_journal_hash
    experiment_md_changed = read_text(experiment_dir / "experiment.md") != experiment_md_backup

    warnings = validation_warnings(validation_errors)
    contract_errors = cycle_contract_errors(
        returncode=runner_result["returncode"],
        marker_errors=marker_errors,
        validation_errors=validation_errors,
        journal_updated=journal_updated,
        experiment_md_changed=experiment_md_changed,
    )

    if contract_errors:
        failure_record: AttemptRecord = {
            "attempt": attempt,
            "returncode": runner_result["returncode"],
            "failure_reason": "cycle_validation_failed",
            "contract_errors": contract_errors,
            "marker_errors": marker_errors,
            "validation_errors": [e for e in validation_errors if not e.startswith("warning:")],
            "validation_warnings": warnings,
            "journal_updated": journal_updated,
            "experiment_md_changed": experiment_md_changed,
            "marker": marker,
        }
        write_json(attempt_result_path, failure_record)
        return AttemptOutcome(
            success=False,
            marker=marker,
            summary=None,
            attempt_record=failure_record,
        )

    summary: CycleSummary = {
        "output_text": output_text,
        "validation_warnings": warnings,
        "journal_updated": journal_updated,
        "experiment_md_changed": experiment_md_changed,
        "returncode": runner_result["returncode"],
    }
    return AttemptOutcome(
        success=True,
        marker=marker,
        summary=summary,
        attempt_record={},
    )
