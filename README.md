# Agentic ML Loop

Agentic ML Loop is a local, offline harness for agent-driven model search.
Define an experiment, point a runner at it, and let the loop run bounded
hypothesis-test cycles while keeping a research journal and leaderboard.

The repo ships with synthetic demos only. No external data source is required.

A few things shape how the loop works:

- **The cycle contract is checked in code.** Each cycle must update the research
  journal, leave `experiment.md` unchanged, emit one completion marker, and pass
  result-schema and metric checks. Cycles that don't are rolled back and retried
  ([`loop/cycle_attempt.py`](loop/cycle_attempt.py), [`loop/contracts.py`](loop/contracts.py)).
- **Each cycle gets an advisory scorecard.** A small rubric grades hypothesis,
  evidence, uncertainty, leakage/split, and error analysis. It's advisory and
  never blocks the loop. See [Research referee](#research-referee).
- **Runners are interchangeable.** The same spec can run under Claude, Codex, or
  Cursor, and you can compare them. See [Benchmark](#benchmark).

There's a worked example in [`examples/`](examples/).

## Setup

Requires Python 3.12+ and `uv`.

```bash
uv sync
uv sync --extra models
uv sync --extra models --group dev
uv run pre-commit install
```

## How It Works

Each experiment lives under `experiments/<experiment_id>/`:

| Path | Purpose |
| --- | --- |
| `experiment.md` | Objective, dataset, target, split policy, and constraints |
| `research_journal.md` | Cycle-by-cycle hypotheses, results, and implications |
| `research_sources.md` | Reusable research notes for the experiment |
| `results.json` | Candidate scores and metrics |
| `outputs/` | Human-facing reports and artifacts |
| `work/` | Intermediate artifacts reused by later cycles |
| `scripts/` | One-shot scripts written during cycles |

The supervisor in `loop/` starts a fresh runner process for each cycle. The
runner receives the cycle prompt on stdin and must end with exactly one marker:

- `<promise>CYCLE_DONE</promise>`
- `<promise>EXPERIMENT_COMPLETE</promise>`

## Research referee

After each cycle the loop computes an advisory scorecard
([`lib/referee.py`](lib/referee.py)) grading the cycle on a small, transparent
rubric:

| Criterion | What it checks |
| --- | --- |
| `journal_updated` | The cycle was actually recorded (evidence discipline) |
| `hypothesis_framed` | A falsifiable hypothesis / expectation was stated |
| `evidence_or_understanding` | New results were produced, or understanding was logged |
| `noise_awareness` | New candidates were judged against uncertainty (CI / bootstrap / significance) |
| `leakage_split_clean` | No unaddressed leakage or split-reliability concern |
| `analysis_when_saturating` | Error analysis happened before declaring a ceiling |

The scorecard is written to `cycles/<id>/scorecard.json`, the latest grade shows
up in `status.md`, and it is **advisory only** — it never blocks a cycle. Turn
it off with `--no-referee` or `AGENTIC_ML_LOOP_REFEREE=0`.

## Runners

The loop has built-in runner presets for Claude, Codex, and Cursor, plus a fully
custom command escape hatch. Each runner receives the cycle prompt on stdin and
must print a final response containing the completion marker.

```bash
uv run python -m loop start experiments/demo_bootstrap --runner claude
uv run python -m loop start experiments/demo_bootstrap --runner codex
uv run python -m loop start experiments/demo_bootstrap --runner cursor
uv run python -m loop start experiments/demo_bootstrap --runner codex --runner-model gpt-5.5-high --runner-effort high
uv run python -m loop start experiments/demo_bootstrap --runner-command "claude --print --verbose --output-format stream-json"
```

Built-in command presets:

| Runner | Command |
| --- | --- |
| Claude | `claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model claude-opus-4-8-high` |
| Codex | `codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high` |
| Cursor | `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5` |

The built-in presets run without approval prompts and with full workspace
permissions. If no model is specified, Claude defaults to
`claude-opus-4-8-high`, Codex defaults to `gpt-5.5-high`, and Cursor defaults to
`composer-2.5`. Claude receives effort through `--effort`; Codex receives effort
through `-c model_reasoning_effort=<effort>`. Cursor does not expose a separate
effort flag, so choose a Cursor model id that already encodes the desired effort.

Useful environment defaults:

```bash
export AGENTIC_ML_LOOP_RUNNER=claude
export AGENTIC_ML_LOOP_RUNNER_COMMAND="claude --print --verbose --output-format stream-json"
export AGENTIC_ML_LOOP_RUNNER_MODEL=claude-sonnet-4-5
export AGENTIC_ML_LOOP_RUNNER_EFFORT=high
export AGENTIC_ML_LOOP_RUNNER_TIMEOUT=1800
```

### Troubleshooting runners

The built-in presets shell out to the `claude`, `codex`, or `cursor-agent`
binaries, so each must be installed and on your `PATH`. If a runner exits
immediately or the loop reports it could not start, check that the CLI runs on
its own first:

```bash
claude --version    # or: codex --version / cursor-agent --version
which claude         # confirm it resolves on PATH
```

If the CLI lives outside `PATH`, point at it with an absolute path via
`--runner-command` (or `AGENTIC_ML_LOOP_RUNNER_COMMAND`). The cross-experiment
learnings step also calls `claude`; when it is absent that step is skipped
silently rather than failing the loop.

## Benchmark

Run the same experiment spec across several runners and compare them:

```bash
uv run python -m loop bench experiments/demo_bootstrap --runners claude,codex,cursor --max-cycles 6
```

Each runner gets an isolated copy of the spec and runs the loop to the shared
budget with the referee on. Results are written to
`bench/<id>-<timestamp>/comparison.md` and `comparison.csv`, ordered by mean
referee score, then best validation score. A runner that errors is recorded and
doesn't stop the others.

## Examples

[`examples/demo_bootstrap_replay.html`](examples/demo_bootstrap_replay.html) is a
self-contained replay of the `demo_bootstrap` experiment — open it in a browser
to walk through all five research cycles (hypothesis → research → training →
scoreboard → journal) with no server or dependencies. It is generated from the
committed [`research_journal.md`](experiments/demo_bootstrap/research_journal.md)
and [`results.json`](experiments/demo_bootstrap/results.json) via `viz/`.

## Demos

Three deterministic demos exercise the framework:

- `demo_bootstrap` — tiny classification smoke test
- `demo_classification` — synthetic binary classification
- `demo_regression` — synthetic zero-inflated revenue regression

```bash
uv run --extra models python runners/demo_bootstrap_runner.py init-demo --force
uv run --extra models python runners/demo_bootstrap_runner.py list-candidates
uv run python experiment.py validate experiments/demo_bootstrap
uv run python -m loop status experiments/demo_bootstrap
```

Synthetic datasets are generated in memory when missing. Running the generator
scripts can materialize local CSVs, but those files are ignored by git.

## Portable Notebooks

Experiments can declare a portable notebook recipe in
`experiments/<id>/notebook.yaml`.

```bash
uv run python -m lib.notebook_export experiments/<id>
```

Generated notebooks are written under `experiments/<id>/outputs/` and are
ignored by git unless you intentionally choose to keep them.

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run --extra models pytest tests/ -v --tb=short
```
