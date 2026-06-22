# AGENTS.md — Agentic ML Loop

`agentic-ml-loop` is a local, offline model-search harness for agent-driven ML
experiments. The main unit of work is an experiment under `experiments/<experiment_id>/`.

User-level guidance (tone, principles, git etiquette, code conduct) lives in
`~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated
here. This file is for project-specific facts.

## Layout

```
experiment.py     Experiment validation + shared helpers
program.md        Research principles + ML craft, read by each cycle
guidelines.md     Operational rules injected into cycle prompts
loop/             Long-running supervisor and runner invocation
runners/          Demo runner entrypoints
lib/              Shared modules (paths.py → outputs/work/scripts; io, eval, schemas)
experiments/      One dir per experiment; <experiment_id>/ holds all its files
.agents/skills/   Portable local skills for experiment setup and notebooks
.agents/docs/     Subsystem docs — see Index
```

## Quickstart

```bash
uv run python experiment.py validate experiments/<experiment_id>   # add --strict-completion to gate done
uv run python -m loop start experiments/<experiment_id>            # also: resume, status; --max-cycles N
uv run --extra models python runners/<experiment_id>_runner.py run-candidate --experiment experiments/<experiment_id> --candidate <id>
uv run --extra models python runners/demo_bootstrap_runner.py init-demo --force

# CI gate (all must pass):
uv run ruff check . && uv run ruff format --check . && uv run --extra models --extra deep pytest
```

## Critical Conventions

- **`CLAUDE.md` is a symlink to `AGENTS.md`.** `AGENTS.md` is the canonical agent
  instruction file; never replace the symlink with a copy.
- **`experiment.md` is a spec, not scratch.** Do not mutate
  `experiments/<experiment_id>/experiment.md` unless the human explicitly asks for
  a spec rewrite.
- **New code lands per-experiment.** Write new scripts under
  `experiments/<experiment_id>/scripts/`; keep long-lived modules in
  `lib/<experiment_id>/`.
- **Runner presets bypass all sandboxing** and run with full workspace
  permissions — see [runners.md](.agents/docs/runners.md) before changing them.
- **Never commit secrets, `.env`, generated notebooks, or local data.**

## Operating Principles

- Optimize for reproducible offline evaluation.
- Prefer clear baselines and trustworthy comparisons over tiny score chasing.
- Respect the split policy and leakage constraints in
  `experiments/<experiment_id>/experiment.md`.
- One loop cycle should complete one bounded hypothesis test.
- Read `experiments/<experiment_id>/research_journal.md`,
  `research_sources.md`, `experiment.md`, and `results.json` before deciding what
  to do next.
- Write the cycle hypothesis to `research_journal.md` before running code, then
  record the result and implications.

## Loop Contract

Each cycle must end with exactly one marker:

- `<promise>CYCLE_DONE</promise>` — continue the experiment.
- `<promise>EXPERIMENT_COMPLETE</promise>` — genuinely done.

## Read The Docs First

Before editing a subsystem, read the matching doc:

- **Architecture / cycle anatomy** → [architecture.md](.agents/docs/architecture.md)
- **Runner config / model aliases / effort flags** → [runners.md](.agents/docs/runners.md)

If a doc disagrees with code, fix the doc in the same change.

## Index

Start in [architecture.md](.agents/docs/architecture.md). Before using the loop
on real data, prove the setup against a bundled demo:
`experiments/demo_bootstrap/`, `experiments/demo_classification/`,
`experiments/demo_regression/`, `experiments/demo_deep/` (requires `--extra deep`).
