# AGENTS.md

## Repo Purpose

This repo is `agentic-ml-loop`, a local offline model-search harness for
agent-driven ML experiments. The main unit of work is an experiment under
`experiments/<experiment_id>/`.

`AGENTS.md` is the canonical agent instruction file. `CLAUDE.md` should remain a
symlink to `AGENTS.md`.

## Core Files

- `program.md` — research principles and ML craft guidance read by each cycle.
- `guidelines.md` — operational rules injected into cycle prompts.
- `experiment.py` — experiment validation and shared helpers.
- `lib/paths.py` — helpers for `outputs/`, `work/`, and `scripts/`.
- `loop/` — long-running supervisor and runner invocation.
- `runners/` — demo runner entrypoints.
- `agents/skills/` — portable local skills for experiment setup and notebooks.

## Operating Principles

- Optimize for reproducible offline evaluation.
- Prefer clear baselines and trustworthy comparisons over tiny score chasing.
- Respect the split policy and leakage constraints in `experiment.md`.
- One loop cycle should complete one bounded hypothesis test.
- Read `research_journal.md`, `research_sources.md`, `experiment.md`, and
  `results.json` before deciding what to do next.
- Write the cycle hypothesis to `research_journal.md` before running code, then
  record the result and implications.

## Loop Contract

Each cycle must end with exactly one marker:

- `<promise>CYCLE_DONE</promise>` — continue the experiment.
- `<promise>EXPERIMENT_COMPLETE</promise>` — genuinely done.

Do not mutate `experiment.md` unless the human explicitly asks for a spec
rewrite. Write new scripts under `experiments/<experiment_id>/scripts/`; keep
long-lived modules in `lib/<experiment_id>/`.

## Commands

```bash
uv run python experiment.py validate experiments/<experiment_id>
uv run python experiment.py validate --strict-completion experiments/<experiment_id>

uv run python -m loop start experiments/<experiment_id>
uv run python -m loop start experiments/<experiment_id> --max-cycles 5
uv run python -m loop resume experiments/<experiment_id>
uv run python -m loop status experiments/<experiment_id>

uv run --extra models python runners/<experiment_id>_runner.py run-candidate --experiment experiments/<experiment_id> --candidate <candidate_id>
uv run --extra models python runners/<experiment_id>_runner.py list-candidates
uv run --extra models python runners/demo_bootstrap_runner.py init-demo --force
```

## Runner Configuration

The loop accepts `--runner claude`, `--runner codex`, `--runner cursor`, or a
custom `--runner-command`. Runner defaults can also come from
`AGENTIC_ML_LOOP_RUNNER`, `AGENTIC_ML_LOOP_RUNNER_COMMAND`,
`AGENTIC_ML_LOOP_RUNNER_MODEL`, `AGENTIC_ML_LOOP_RUNNER_EFFORT`, and
`AGENTIC_ML_LOOP_RUNNER_TIMEOUT`.

Built-in commands run unattended with full workspace permissions:
`claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model claude-opus-4-8-high`,
`codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high`,
and `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5`.
`--runner-model` overrides those defaults. Claude receives effort through
`--effort`; Codex receives effort through `-c model_reasoning_effort=<effort>`.
Cursor does not expose a separate effort flag, so choose a Cursor model id that
already encodes the desired effort.

## Code Conduct

- Use `pathlib.Path` over `os.path`.
- Add Python type hints for new code.
- Prefer explicit, human names.
- Match neighboring style.
- Keep changes scoped to the requested work.
- Do not commit secrets, credentials, `.env`, generated notebooks, or local data.

## Demos

Before using the loop on real data, prove the setup against a bundled demo:

- `experiments/demo_bootstrap/`
- `experiments/demo_classification/`
- `experiments/demo_regression/`
