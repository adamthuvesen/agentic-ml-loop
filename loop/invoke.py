"""Runner invocation and transcript parsing for the loop supervisor."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

BUILTIN_RUNNER_COMMANDS = {
    "claude": ["claude", "--print", "--verbose", "--output-format", "stream-json"],
    "codex": ["codex", "exec"],
    "cursor": ["cursor-agent", "--print"],
}
RUNNER_TIMEOUT = 1800


@dataclass(frozen=True)
class RunnerConfig:
    """Resolved runner command and display metadata for loop attempts."""

    name: str
    command: list[str]
    timeout_seconds: int = RUNNER_TIMEOUT
    model: str | None = None
    effort: str | None = None

    @property
    def display_command(self) -> str:
        return shlex.join(self.command)


def default_runner_config() -> RunnerConfig:
    """Resolve runner config from environment variables or built-in defaults."""
    return build_runner_config(
        runner=os.environ.get("AGENTIC_ML_LOOP_RUNNER", "claude"),
        runner_command=os.environ.get("AGENTIC_ML_LOOP_RUNNER_COMMAND"),
        runner_model=os.environ.get("AGENTIC_ML_LOOP_RUNNER_MODEL"),
        runner_effort=os.environ.get("AGENTIC_ML_LOOP_RUNNER_EFFORT"),
        runner_timeout=_env_int("AGENTIC_ML_LOOP_RUNNER_TIMEOUT", RUNNER_TIMEOUT),
    )


def build_runner_config(
    *,
    runner: str,
    runner_command: str | None = None,
    runner_model: str | None = None,
    runner_effort: str | None = None,
    runner_timeout: int = RUNNER_TIMEOUT,
) -> RunnerConfig:
    """Build a runner config from CLI/env values."""
    if runner_command:
        command = shlex.split(runner_command)
        if not command:
            raise ValueError("--runner-command cannot be empty")
        name = runner or command[0]
    else:
        if runner not in BUILTIN_RUNNER_COMMANDS:
            supported = ", ".join(sorted(BUILTIN_RUNNER_COMMANDS))
            raise ValueError(
                f"Unsupported runner {runner!r}; expected one of {supported}."
            )
        command = list(BUILTIN_RUNNER_COMMANDS[runner])
        name = runner
        if runner == "claude":
            if runner_model:
                command.extend(["--model", runner_model])
            if runner_effort:
                command.extend(["--effort", runner_effort])

    return RunnerConfig(
        name=name,
        command=command,
        timeout_seconds=runner_timeout,
        model=runner_model or _arg_after(command, "--model"),
        effort=runner_effort or _arg_after(command, "--effort"),
    )


def add_runner_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared runner configuration flags to a subcommand parser."""
    parser.add_argument(
        "--runner",
        choices=sorted(BUILTIN_RUNNER_COMMANDS),
        default=os.environ.get("AGENTIC_ML_LOOP_RUNNER", "claude"),
        help="Built-in runner command to use when --runner-command is not set.",
    )
    parser.add_argument(
        "--runner-command",
        default=os.environ.get("AGENTIC_ML_LOOP_RUNNER_COMMAND"),
        help="Shell-style command string that receives the cycle prompt on stdin.",
    )
    parser.add_argument(
        "--runner-model",
        default=os.environ.get("AGENTIC_ML_LOOP_RUNNER_MODEL"),
        help="Optional model metadata; appended to built-in Claude commands.",
    )
    parser.add_argument(
        "--runner-effort",
        default=os.environ.get("AGENTIC_ML_LOOP_RUNNER_EFFORT"),
        help="Optional effort metadata; appended to built-in Claude commands.",
    )
    parser.add_argument(
        "--runner-timeout",
        type=_positive_int,
        default=_env_int("AGENTIC_ML_LOOP_RUNNER_TIMEOUT", RUNNER_TIMEOUT),
        help="Seconds before a runner attempt is treated as timed out.",
    )


def extract_agent_text_from_stream_json(raw: str) -> str:
    """Extract all assistant text from stream-json output.

    stream-json emits newline-delimited JSON events. Assistant text lives in
    events with type "assistant" that contain content blocks of type "text".
    The final "result" event also has a .result field with the last text.

    We collect ALL assistant text blocks across the entire session so the
    completion marker is captured even if the agent does tool calls after it.
    """
    text_parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "assistant" and "message" in event:
            for block in event["message"].get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
        elif event.get("type") == "result" and isinstance(event.get("result"), str):
            text_parts.append(event["result"])

    return "\n".join(text_parts) if text_parts else raw.strip()


def extract_agent_text(stdout_path: Path) -> str:
    """Extract agent text from stdout (always stream-json format)."""
    try:
        raw = stdout_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    if not raw.strip():
        return ""
    return extract_agent_text_from_stream_json(raw)


def invoke_runner(
    prompt_text: str,
    stdout_path: Path,
    stderr_path: Path,
    agent_message_path: Path,
    runner_config: RunnerConfig | None = None,
) -> dict[str, Any]:
    """Run the configured agent CLI once, streaming output to ``stdout_path``.

    ``cwd`` is the repo root so paths in the prompt resolve consistently. On success,
    assistant text is extracted from stream-json into ``agent_message_path`` when
    missing, so downstream code can read markers like ``<promise>CYCLE_DONE</promise>``.

    Return dict keys: ``returncode`` (``-1`` if timed out), ``stdout_path``,
    ``stderr_path``, ``agent_message_path`` (may be ``None``), and ``timeout``
    (``True`` only on :exc:`subprocess.TimeoutExpired`; partial stdout may still be
    parsed). Successful completion omits ``timeout``.
    """
    runner_config = runner_config or default_runner_config()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    # Repo cwd so paths in the prompt match how tools resolve; prompt is stdin.
    kwargs: dict[str, Any] = {
        "cwd": ROOT,
        "env": os.environ.copy(),
        "text": True,
        "timeout": runner_config.timeout_seconds,
        "shell": False,
        "input": prompt_text,
    }

    with (
        open(stdout_path, "w", encoding="utf-8") as out_f,
        open(stderr_path, "w", encoding="utf-8") as err_f,
    ):
        try:
            completed = subprocess.run(
                runner_config.command, stdout=out_f, stderr=err_f, **kwargs
            )
        except subprocess.TimeoutExpired:
            # Still persist parsed stdout so timeouts leave a debuggable transcript.
            agent_text = extract_agent_text(stdout_path)
            if agent_text.strip() and not agent_message_path.exists():
                agent_message_path.write_text(agent_text, encoding="utf-8")
            return {
                "returncode": -1,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "agent_message_path": (
                    str(agent_message_path) if agent_message_path.exists() else None
                ),
                "timeout": True,
            }

    # Assistant text lives in NDJSON on stdout; extract for <promise> marker parsing.
    if not agent_message_path.exists():
        agent_text = extract_agent_text(stdout_path)
        if agent_text.strip():
            agent_message_path.write_text(agent_text, encoding="utf-8")

    return {
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "agent_message_path": (
            str(agent_message_path) if agent_message_path.exists() else None
        ),
    }


def _arg_after(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    return argv[idx + 1] if idx + 1 < len(argv) else None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return _positive_int(raw)


def _positive_int(value: str | int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed
