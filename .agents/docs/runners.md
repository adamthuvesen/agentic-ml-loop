# Runner Configuration

The loop accepts `--runner claude`, `--runner codex`, `--runner cursor`, or a
custom `--runner-command`. Runner defaults can also come from
`AGENTIC_ML_LOOP_RUNNER`, `AGENTIC_ML_LOOP_RUNNER_COMMAND`,
`AGENTIC_ML_LOOP_RUNNER_MODEL`, `AGENTIC_ML_LOOP_RUNNER_EFFORT`, and
`AGENTIC_ML_LOOP_RUNNER_TIMEOUT`.

Built-in commands run unattended with full workspace permissions:
`claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model opus`,
`codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high`,
and `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5`.

The Claude preset records `claude-opus-4-8-high` as the requested model and
resolves it to the local CLI's accepted `opus` alias. `--runner-model` overrides
those defaults.

## Effort flags

Claude receives effort through `--effort`; Codex receives effort through
`-c model_reasoning_effort=<effort>`. Cursor does not expose a separate effort
flag, so choose a Cursor model id that already encodes the desired effort.
