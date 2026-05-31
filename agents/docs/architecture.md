# Agentic ML Loop Architecture

Agentic ML Loop repeatedly invokes a configured CLI runner to run bounded ML
research cycles on a local experiment directory. The loop builds a prompt from
the experiment files, captures the runner transcript, validates artifacts, and
updates persisted state.

## Layout

```text
loop/                  Supervisor, prompts, state, status, runner invocation
lib/                   Shared framework helpers and demo experiment packages
runners/               Runner CLIs for bundled demos
experiments/           Experiment specs, journals, results, outputs, work
agents/skills/         Portable skills for creating experiments and notebooks
viz/                   Optional replay generation and renderer
tests/                 Framework and demo tests
```

Framework modules should not import concrete experiment packages directly.
Experiment-specific behavior belongs under `lib/<experiment_id>/` and is reached
through runner registries.

## Cycle Flow

1. Read `experiment.md`, `research_journal.md`, `research_sources.md`, and
   `results.json`.
2. Build a cycle prompt from static program guidance and dynamic experiment
   state.
3. Invoke the configured runner command with the prompt on stdin.
4. Extract assistant text from stream JSON when available, or raw stdout
   otherwise.
5. Validate the experiment contract and completion marker.
6. Snapshot artifacts and update `loop_state.json` plus `status.md`.

## Runner Configuration

Built-in presets exist for Claude, Codex, and Cursor. A custom command can be
provided with `--runner-command`; it receives the cycle prompt on stdin and may
return stream JSON or plain text. The resolved command and timeout are persisted
in loop state and attempt metadata for reproducibility.

The built-in presets use
`claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model claude-opus-4-8-high`,
`codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high`,
and `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5`.
Model names are persisted in loop state and can be overridden with
`--runner-model`. Effort maps to `--effort` for Claude and `-c
model_reasoning_effort=<effort>` for Codex. Cursor has no separate effort flag;
use a Cursor model id that already encodes effort when needed.

## Experiment Structure

```text
experiments/<experiment_id>/
  experiment.md
  research_journal.md
  research_sources.md
  results.json
  outputs/
  work/
  scripts/
```

Use `lib.paths.outputs_dir(exp_dir)`, `work_dir(exp_dir)`, and
`scripts_dir(exp_dir)` instead of hard-coded paths.

## Demos

The public repo includes only deterministic synthetic demos:

- `demo_bootstrap`
- `demo_classification`
- `demo_regression`

Synthetic data is generated in memory when local CSVs are absent. Generated data
files remain local artifacts and are ignored by git.
