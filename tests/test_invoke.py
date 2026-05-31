from __future__ import annotations

import json
from pathlib import Path

from loop.invoke import (
    build_runner_config,
    extract_agent_text,
    extract_agent_text_from_stream_json,
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

    assert "<promise>EXPERIMENT_COMPLETE</promise>" in (
        extract_agent_text_from_stream_json(raw)
    )


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
    assert (
        extract_agent_text_from_stream_json("plain text marker") == "plain text marker"
    )


def test_build_runner_config_uses_custom_command() -> None:
    config = build_runner_config(
        runner="codex",
        runner_command="codex exec --model gpt-5",
        runner_timeout=42,
    )

    assert config.name == "codex"
    assert config.command == ["codex", "exec", "--model", "gpt-5"]
    assert config.model == "gpt-5"
    assert config.timeout_seconds == 42
