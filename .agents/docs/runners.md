# Runner Configuration

The loop accepts `--runner claude`, `--runner codex`, `--runner cursor`, or a
custom `--runner-command`. Runner defaults can also come from
`AGENTIC_ML_LOOP_RUNNER`, `AGENTIC_ML_LOOP_RUNNER_COMMAND`,
`AGENTIC_ML_LOOP_RUNNER_MODEL`, `AGENTIC_ML_LOOP_RUNNER_EFFORT`, and
`AGENTIC_ML_LOOP_RUNNER_TIMEOUT`.

Built-in commands:

- `claude --print --verbose --output-format stream-json --permission-mode acceptEdits --allowedTools <CLAUDE_ALLOWED_TOOLS> --model opus`
- `codex exec --full-auto -c sandbox_workspace_write.network_access=true --model gpt-5.5-high`
- `cursor-agent --print --force --model composer-2.5`

## Permissions

Cycles are unattended, so a runner that stops to ask is a runner that hangs.
Every preset therefore grants shell access up front — the agent has to run
training scripts. What the presets do *not* do is switch enforcement off
entirely:

- **Claude** takes an explicit tool allowlist (`CLAUDE_ALLOWED_TOOLS` in
  `loop/invoke.py`) instead of `bypassPermissions`. It covers the built-in file,
  shell, and web tools plus `mcp__context7`, which the first-cycle prompt asks
  for by name. Any other MCP server your setup exposes is denied — add it with
  `--runner-command` if a cycle needs it.
- **Codex** runs `--full-auto`, which keeps the workspace-write sandbox rather
  than `--dangerously-bypass-approvals-and-sandbox`. Network access is enabled
  explicitly because cycles do live research; without it the first-cycle
  research phase fails.
- **Cursor** keeps `--force` for non-interactive approval but leaves the sandbox
  on.

This is a smaller blast radius, not an isolated one. An agent with `Bash` can
still do anything the shell can. If you need real isolation, run the loop in a
container.

The Claude preset records `claude-opus-4-8-high` as the requested model and
resolves it to the local CLI's accepted `opus` alias. `--runner-model` overrides
those defaults.

## Effort flags

Claude receives effort through `--effort`; Codex receives effort through
`-c model_reasoning_effort=<effort>`. Cursor does not expose a separate effort
flag, so choose a Cursor model id that already encodes the desired effort.
