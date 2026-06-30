"""Runner invocation and transcript parsing for the loop supervisor."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

RUNNER_TIMEOUT = 1800
ROOT = Path(__file__).resolve().parent.parent


class RunnerResult(TypedDict):
    """Outcome of one runner subprocess invocation.

    ``returncode`` is ``-1`` on timeout. ``agent_message_path`` is ``None`` when
    no assistant text could be extracted. ``timeout`` is present and ``True``
    only when the subprocess timed out.
    """

    returncode: int
    stdout_path: str
    stderr_path: str
    agent_message_path: str | None
    timeout: NotRequired[bool]


@dataclass(frozen=True)
class RunnerPreset:
    """Built-in CLI command shape."""

    command: tuple[str, ...]
    default_model: str


BUILTIN_RUNNER_PRESETS = {
    "claude": RunnerPreset(
        command=(
            "claude",
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--permission-mode",
            "bypassPermissions",
        ),
        default_model="claude-opus-4-8-high",
    ),
    "codex": RunnerPreset(
        command=("codex", "exec", "--dangerously-bypass-approvals-and-sandbox"),
        default_model="gpt-5.5-high",
    ),
    "cursor": RunnerPreset(
        command=(
            "cursor-agent",
            "--print",
            "--trust",
            "--force",
            "--sandbox",
            "disabled",
        ),
        default_model="composer-2.5",
    ),
}
BUILTIN_RUNNER_COMMANDS = {
    name: list(preset.command) for name, preset in BUILTIN_RUNNER_PRESETS.items()
}

RUNNER_MODEL_ALIASES = {
    "claude": {
        # The local Claude CLI accepts this stable family alias while some
        # fully-qualified Opus ids can lag behind or return model_not_found.
        "claude-opus-4-8-high": "opus",
    }
}


@dataclass(frozen=True)
class RunnerConfig:
    """Resolved runner command and display metadata for loop attempts."""

    name: str
    command: list[str]
    timeout_seconds: int = RUNNER_TIMEOUT
    model: str | None = None
    resolved_model: str | None = None
    effort: str | None = None

    @property
    def display_command(self) -> str:
        return shlex.join(self.command)


def default_runner_config() -> RunnerConfig:
    """Resolve runner config from environment variables or built-in defaults."""
    return runner_config_from_args(
        runner=os.environ.get("AGENTIC_ML_LOOP_RUNNER", "claude"),
        runner_command=os.environ.get("AGENTIC_ML_LOOP_RUNNER_COMMAND"),
        runner_model=os.environ.get("AGENTIC_ML_LOOP_RUNNER_MODEL"),
        runner_effort=os.environ.get("AGENTIC_ML_LOOP_RUNNER_EFFORT"),
        runner_timeout=_env_int("AGENTIC_ML_LOOP_RUNNER_TIMEOUT", RUNNER_TIMEOUT),
    )


def runner_config_from_args(
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
        name = _runner_name_from_command(command[0])
    else:
        if runner not in BUILTIN_RUNNER_PRESETS:
            supported = ", ".join(sorted(BUILTIN_RUNNER_PRESETS))
            raise ValueError(f"Unsupported runner {runner!r}; expected one of {supported}.")
        preset = BUILTIN_RUNNER_PRESETS[runner]
        command = list(preset.command)
        name = runner
        runner_model = runner_model or preset.default_model
        resolved_runner_model = resolve_runner_model(runner, runner_model)
        _append_builtin_runner_options(
            runner=runner,
            command=command,
            runner_model=resolved_runner_model,
            runner_effort=runner_effort,
        )
    if runner_command:
        resolved_runner_model = _arg_after(command, "--model")

    return RunnerConfig(
        name=name,
        command=command,
        timeout_seconds=runner_timeout,
        model=runner_model or _arg_after(command, "--model"),
        resolved_model=resolved_runner_model,
        effort=runner_effort or _arg_after(command, "--effort"),
    )


def resolve_runner_model(runner: str, requested_model: str | None) -> str | None:
    """Return the model id to pass to a runner CLI."""
    if requested_model is None:
        return None
    return RUNNER_MODEL_ALIASES.get(runner, {}).get(requested_model, requested_model)


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
        help="Optional model name appended to built-in runners that support --model.",
    )
    parser.add_argument(
        "--runner-effort",
        default=os.environ.get("AGENTIC_ML_LOOP_RUNNER_EFFORT"),
        help="Optional effort level passed through each built-in runner's mechanism.",
    )
    parser.add_argument(
        "--runner-timeout",
        type=_positive_int,
        default=_env_int("AGENTIC_ML_LOOP_RUNNER_TIMEOUT", RUNNER_TIMEOUT),
        help="Seconds before a runner attempt is treated as timed out.",
    )


def _jsonl_events(raw: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _assistant_message_parts(message: object) -> list[str]:
    if isinstance(message, str):
        return [message]
    if not isinstance(message, dict):
        return []

    parts: list[str] = []
    for block in message.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return parts


def _event_text_parts(event: dict[str, Any]) -> list[str]:
    event_type = event.get("type")
    if event_type == "assistant" and "message" in event:
        return _assistant_message_parts(event["message"])
    if event_type == "result" and isinstance(event.get("result"), str):
        return [event["result"]]
    if event_type in {"agent_message", "assistant_message"}:
        return _assistant_message_parts(event.get("message"))
    return []


def extract_agent_text_from_jsonl(raw: str) -> str:
    """Extract assistant text from known JSONL output formats.

    Claude stream-json emits assistant message content blocks and result events.
    Other runners may emit JSONL events with plain string message/result fields.
    Unknown JSONL falls back to raw stdout so text-mode Codex/Cursor still works.

    We collect ALL assistant text blocks across the entire session so the
    completion marker is captured even if the agent does tool calls after it.
    """
    text_parts: list[str] = []
    for event in _jsonl_events(raw):
        text_parts.extend(_event_text_parts(event))

    return "\n".join(text_parts) if text_parts else raw.strip()


extract_agent_text_from_stream_json = extract_agent_text_from_jsonl


def extract_agent_text(stdout_path: Path) -> str:
    """Extract agent text from stdout JSONL, or return plain text stdout."""
    try:
        raw = stdout_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    if not raw.strip():
        return ""
    return extract_agent_text_from_jsonl(raw)


def invoke_runner(
    prompt_text: str,
    stdout_path: Path,
    stderr_path: Path,
    agent_message_path: Path,
    runner_config: RunnerConfig | None = None,
) -> RunnerResult:
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
            completed = subprocess.run(runner_config.command, stdout=out_f, stderr=err_f, **kwargs)
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

    # Extract assistant text for <promise> marker parsing.
    if not agent_message_path.exists():
        agent_text = extract_agent_text(stdout_path)
        if agent_text.strip():
            agent_message_path.write_text(agent_text, encoding="utf-8")

    return {
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "agent_message_path": (str(agent_message_path) if agent_message_path.exists() else None),
    }


def _arg_after(argv: list[str], flag: str) -> str | None:
    try:
        idx = argv.index(flag)
    except ValueError:
        return None
    return argv[idx + 1] if idx + 1 < len(argv) else None


def _runner_name_from_command(executable: str) -> str:
    name = Path(executable).name
    if name == "cursor-agent":
        return "cursor"
    if name in BUILTIN_RUNNER_PRESETS:
        return name
    return name


def _append_builtin_runner_options(
    *,
    runner: str,
    command: list[str],
    runner_model: str | None,
    runner_effort: str | None,
) -> None:
    if runner == "cursor":
        if runner_effort:
            raise ValueError(
                "--runner-effort is not available for cursor; choose a Cursor "
                "model id that already encodes the desired effort."
            )
        command.extend(["--model", runner_model])
        return

    command.extend(["--model", runner_model])
    if runner == "claude" and runner_effort:
        command.extend(["--effort", runner_effort])
    elif runner == "codex" and runner_effort:
        command.extend(["-c", f"model_reasoning_effort={runner_effort}"])


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
