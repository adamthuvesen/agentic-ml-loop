from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop.invoke import (
    extract_agent_text,
    extract_agent_text_from_jsonl,
    extract_agent_text_from_stream_json,
    runner_config_from_args,
)


def test_extract_agent_text_collects_assistant_text_blocks() -> None:
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "text", "text": "First note."},
                            {"type": "tool_use", "name": "x"},
                            {"type": "text", "text": "<promise>CYCLE_DONE</promise>"},
                        ]
                    },
                }
            )
        ]
    )

    assert extract_agent_text_from_stream_json(raw) == (
        "First note.\n<promise>CYCLE_DONE</promise>"
    )


def test_extract_agent_text_uses_result_event_when_assistant_text_missing() -> None:
    raw = json.dumps(
        {
            "type": "result",
            "result": "Final answer.\n<promise>EXPERIMENT_COMPLETE</promise>",
        }
    )

    assert "<promise>EXPERIMENT_COMPLETE</promise>" in (extract_agent_text_from_stream_json(raw))


def test_extract_agent_text_ignores_malformed_ndjson_lines(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        "{not json}\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        + "\n"
    )

    assert extract_agent_text(stdout) == "ok"


def test_extract_agent_text_falls_back_to_raw_stdout() -> None:
    assert extract_agent_text_from_stream_json("plain text marker") == "plain text marker"


def test_extract_agent_text_handles_string_message_jsonl() -> None:
    raw = json.dumps(
        {
            "type": "agent_message",
            "message": "Done.\n<promise>CYCLE_DONE</promise>",
        }
    )

    assert extract_agent_text_from_jsonl(raw) == "Done.\n<promise>CYCLE_DONE</promise>"


@pytest.mark.parametrize(
    ("runner", "expected_command"),
    [
        (
            "claude",
            [
                "claude",
                "--print",
                "--verbose",
                "--output-format",
                "stream-json",
                "--permission-mode",
                "bypassPermissions",
                "--model",
                "claude-opus-4-8-high",
            ],
        ),
        (
            "codex",
            [
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--model",
                "gpt-5.5-high",
            ],
        ),
        (
            "cursor",
            [
                "cursor-agent",
                "--print",
                "--trust",
                "--force",
                "--sandbox",
                "disabled",
                "--model",
                "composer-2.5",
            ],
        ),
    ],
)
def test_runner_config_from_args_uses_builtin_runner_presets(
    runner: str,
    expected_command: list[str],
) -> None:
    config = runner_config_from_args(runner=runner)

    assert config.name == runner
    assert config.command == expected_command


@pytest.mark.parametrize("runner", ["claude", "codex", "cursor"])
def test_runner_config_from_args_appends_model_for_builtin_runners(runner: str) -> None:
    config = runner_config_from_args(runner=runner, runner_model="test-model")

    assert config.model == "test-model"
    assert config.command[-2:] == ["--model", "test-model"]


def test_runner_config_from_args_appends_effort_for_claude() -> None:
    config = runner_config_from_args(runner="claude", runner_effort="high")

    assert config.model == "claude-opus-4-8-high"
    assert config.effort == "high"
    assert config.command[-2:] == ["--effort", "high"]


def test_runner_config_from_args_passes_effort_to_codex_config() -> None:
    config = runner_config_from_args(runner="codex", runner_effort="high")

    assert config.model == "gpt-5.5-high"
    assert config.effort == "high"
    assert config.command[-2:] == ["-c", "model_reasoning_effort=high"]


def test_runner_config_from_args_rejects_cursor_effort() -> None:
    with pytest.raises(ValueError, match="not available for cursor"):
        runner_config_from_args(runner="cursor", runner_effort="high")


def test_runner_config_from_args_uses_custom_command() -> None:
    config = runner_config_from_args(
        runner="codex",
        runner_command="codex exec --model gpt-5",
        runner_timeout=42,
    )

    assert config.name == "codex"
    assert config.command == ["codex", "exec", "--model", "gpt-5"]
    assert config.model == "gpt-5"
    assert config.timeout_seconds == 42


def test_runner_config_from_args_infers_custom_runner_name_from_command() -> None:
    config = runner_config_from_args(
        runner="claude",
        runner_command="cursor-agent --print --model gpt-5",
    )

    assert config.name == "cursor"
    assert config.command == ["cursor-agent", "--print", "--model", "gpt-5"]
