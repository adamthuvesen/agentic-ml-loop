from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from loop.cycle_attempt import run_cycle_attempt
from loop.invoke import RunnerConfig
from tests.loop.conftest import make_experiment_dir


def test_runner_invocation_error_writes_attempt_result(tmp_path: Path) -> None:
    experiment_dir = make_experiment_dir(tmp_path)
    cycle_dir = experiment_dir / "cycles" / "0001"
    cycle_dir.mkdir(parents=True)

    attempt_result_path = cycle_dir / "attempt_01_result.json"

    with patch(
        "loop.cycle_attempt.invoke_runner",
        side_effect=RuntimeError("runner unavailable"),
    ):
        outcome = run_cycle_attempt(
            experiment_dir=experiment_dir,
            cycle_id="0001",
            attempt=1,
            prompt_text="test prompt",
            baselines_journal_hash="abc",
            experiment_md_backup=(experiment_dir / "experiment.md").read_text(),
            attempt_stdout=cycle_dir / "attempt_01_stdout.log",
            attempt_stderr=cycle_dir / "attempt_01_stderr.log",
            attempt_meta=cycle_dir / "attempt_01_meta.json",
            attempt_result_path=attempt_result_path,
            agent_message_path=cycle_dir / "agent_last_message.md",
        )

    assert not outcome.success
    assert outcome.attempt_record["failure_reason"] == "runner_invocation_error:RuntimeError"
    assert attempt_result_path.exists()
    assert "runner unavailable" in attempt_result_path.read_text()


def test_contract_failure_record_matches_persisted_json(tmp_path: Path) -> None:
    """The returned attempt_record and the on-disk attempt_result.json are identical.

    Guards the de-duplicated failure record: both consumers read one dict.
    """
    experiment_dir = make_experiment_dir(tmp_path)
    cycle_dir = experiment_dir / "cycles" / "0001"
    cycle_dir.mkdir(parents=True)
    attempt_result_path = cycle_dir / "attempt_01_result.json"

    runner_result = {
        "returncode": 0,
        "stdout_path": str(cycle_dir / "attempt_01_stdout.log"),
        "stderr_path": str(cycle_dir / "attempt_01_stderr.log"),
        "agent_message_path": None,  # no assistant text -> missing completion marker
    }

    with patch("loop.cycle_attempt.invoke_runner", return_value=runner_result):
        outcome = run_cycle_attempt(
            experiment_dir=experiment_dir,
            cycle_id="0001",
            attempt=1,
            prompt_text="test prompt",
            baselines_journal_hash="abc",
            experiment_md_backup=(experiment_dir / "experiment.md").read_text(),
            attempt_stdout=cycle_dir / "attempt_01_stdout.log",
            attempt_stderr=cycle_dir / "attempt_01_stderr.log",
            attempt_meta=cycle_dir / "attempt_01_meta.json",
            attempt_result_path=attempt_result_path,
            agent_message_path=cycle_dir / "agent_last_message.md",
        )

    assert not outcome.success
    assert outcome.attempt_record["failure_reason"] == "cycle_validation_failed"
    assert "missing completion marker" in outcome.attempt_record["marker_errors"]
    assert json.loads(attempt_result_path.read_text()) == dict(outcome.attempt_record)


def test_runner_config_is_written_to_attempt_metadata(tmp_path: Path) -> None:
    experiment_dir = make_experiment_dir(tmp_path)
    cycle_dir = experiment_dir / "cycles" / "0001"
    cycle_dir.mkdir(parents=True)
    attempt_meta = cycle_dir / "attempt_01_meta.json"

    config = RunnerConfig(
        name="codex",
        command=["codex", "exec", "--model", "gpt-5"],
        timeout_seconds=42,
        model="gpt-5",
    )

    with patch("loop.cycle_attempt.invoke_runner", side_effect=RuntimeError("stop")):
        run_cycle_attempt(
            experiment_dir=experiment_dir,
            cycle_id="0001",
            attempt=1,
            prompt_text="test prompt",
            baselines_journal_hash="abc",
            experiment_md_backup=(experiment_dir / "experiment.md").read_text(),
            attempt_stdout=cycle_dir / "attempt_01_stdout.log",
            attempt_stderr=cycle_dir / "attempt_01_stderr.log",
            attempt_meta=attempt_meta,
            attempt_result_path=cycle_dir / "attempt_01_result.json",
            agent_message_path=cycle_dir / "agent_last_message.md",
            runner_config=config,
        )

    meta = json.loads(attempt_meta.read_text())
    assert meta["runner_name"] == "codex"
    assert meta["runner_command"] == ["codex", "exec", "--model", "gpt-5"]
    assert meta["runner_model"] == "gpt-5"
    assert meta["runner_resolved_model"] is None
    assert meta["timeout_seconds"] == 42
