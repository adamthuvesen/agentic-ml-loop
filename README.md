# Agentic ML Loop

A local, offline harness for agent-driven model search (inspired by
autoresearch). Define an experiment, point a runner at it, and the loop runs
bounded hypothesis-test cycles, keeping a research journal and leaderboard.
The cycle contract is enforced in code ([`loop/cycle_attempt.py`](loop/cycle_attempt.py),
[`loop/contracts.py`](loop/contracts.py)): each cycle must update the journal,
leave `experiment.md` unchanged, emit one completion marker, and pass
result-schema and metric checks — or it is rolled back and retried.

Ships with synthetic demos only; no external data source required. Worked
example in [`examples/`](examples/). Internals:
[`.agents/docs/architecture.md`](.agents/docs/architecture.md).

## Setup

Requires Python 3.12+ and `uv`.

```bash
uv sync --extra models --extra deep   # extras optional; dev group installs by default
uv run pre-commit install
```

## Experiment layout

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

The supervisor in `loop/` starts a fresh runner process per cycle. The runner
receives the cycle prompt on stdin and must end with exactly one marker:
`<promise>CYCLE_DONE</promise>` or `<promise>EXPERIMENT_COMPLETE</promise>`.

## Runners

Built-in presets for Claude, Codex, and Cursor, plus a custom command escape
hatch. Presets run without approval prompts and with full workspace permissions.

```bash
uv run python -m loop start experiments/demo_bootstrap --runner claude
uv run python -m loop start experiments/demo_bootstrap --runner codex --runner-model gpt-5.5-high --runner-effort high
uv run python -m loop start experiments/demo_bootstrap --runner-command "claude --print --verbose --output-format stream-json"
```

| Runner | Default command |
| --- | --- |
| Claude | `claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model claude-opus-4-8-high` |
| Codex | `codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high` |
| Cursor | `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5` |

`--runner-effort` maps to `--effort` for Claude and
`-c model_reasoning_effort=<effort>` for Codex; Cursor has no effort flag, so
pick a model id that encodes it. Environment defaults: `AGENTIC_ML_LOOP_RUNNER`,
`AGENTIC_ML_LOOP_RUNNER_COMMAND`, `AGENTIC_ML_LOOP_RUNNER_MODEL`,
`AGENTIC_ML_LOOP_RUNNER_EFFORT`, `AGENTIC_ML_LOOP_RUNNER_TIMEOUT` (seconds).

Troubleshooting: presets shell out to the `claude` / `codex` / `cursor-agent`
binaries, which must be on `PATH` — check with `claude --version` and
`which claude`. Otherwise pass an absolute path via `--runner-command` (or
`AGENTIC_ML_LOOP_RUNNER_COMMAND`). The cross-experiment learnings step also
calls `claude`; when absent it is skipped silently rather than failing the loop.

## Research referee

After each cycle the loop computes an advisory scorecard
([`lib/referee.py`](lib/referee.py)) — it never blocks a cycle:

| Criterion | What it checks |
| --- | --- |
| `journal_updated` | The cycle was actually recorded (evidence discipline) |
| `hypothesis_framed` | A falsifiable hypothesis / expectation was stated |
| `evidence_or_understanding` | New results were produced, or understanding was logged |
| `noise_awareness` | New candidates were judged against uncertainty (CI / bootstrap / significance) |
| `leakage_split_clean` | No unaddressed leakage or split-reliability concern |
| `analysis_when_saturating` | Error analysis happened before declaring a ceiling |

Scorecards land in `cycles/<id>/scorecard.json`; the latest grade shows in
`status.md`. Disable with `--no-referee` or `AGENTIC_ML_LOOP_REFEREE=0`.

## Benchmark

Run the same spec across several runners and compare:

```bash
uv run python -m loop bench experiments/demo_bootstrap --runners claude,codex,cursor --max-cycles 6
```

Each runner gets an isolated copy of the spec and runs to the shared budget with
the referee on. Results go to `bench/<id>-<timestamp>/comparison.md` and
`comparison.csv`, ordered by mean referee score, then best validation score. A
runner that errors is recorded and doesn't stop the others.

## Demos

Four deterministic demos exercise the framework:

- `demo_bootstrap` — tiny classification smoke test
- `demo_classification` — synthetic binary classification
- `demo_regression` — synthetic zero-inflated revenue regression
- `demo_deep` — synthetic nonlinear tabular classification with PyTorch MLPs

```bash
uv run --extra models python runners/demo_bootstrap_runner.py init-demo --force
uv run --extra deep python runners/demo_deep_runner.py init-demo --force
uv run --extra deep python runners/demo_deep_runner.py list-candidates
uv run --extra deep python runners/demo_deep_runner.py run-candidate \
  --experiment experiments/demo_deep --candidate mlp-deep
uv run python experiment.py validate experiments/demo_bootstrap
uv run python -m loop status experiments/demo_bootstrap
```

Synthetic datasets are generated in memory when missing; generator scripts can
materialize local CSVs, which git ignores.

[`examples/demo_bootstrap_replay.html`](examples/demo_bootstrap_replay.html) is
a self-contained browser replay of all five `demo_bootstrap` research cycles,
generated from the committed
[`research_journal.md`](experiments/demo_bootstrap/research_journal.md) and
[`results.json`](experiments/demo_bootstrap/results.json) via `viz/`.

## Portable notebooks

Experiments can declare a notebook recipe in `experiments/<id>/notebook.yaml`:

```bash
uv run python -m lib.notebook_export experiments/<id>
```

Generated notebooks land in `experiments/<id>/outputs/`, ignored by git.

## Warehouse ingestion (optional)

Pull a frozen snapshot from Snowflake, BigQuery, Redshift, Databricks, or
Postgres, then run the loop offline against it. The warehouse is touched once,
at the edge; cycles never query it, so reproducibility and cross-runner
comparison hold. Clients are optional extras; the default install stays
`numpy` + `pandas`.

```bash
uv sync --extra postgres              # or: snowflake | bigquery | databricks | redshift
python -m lib.sources sources list    # see available sources + setup docs
python -m lib.sources pull \
  --experiment experiments/<id> --source duckdb \
  --database path/to.duckdb --query "SELECT * FROM events ORDER BY id"
```

A pull writes `experiments/<id>/data/snapshot.parquet` plus a
`dataset_manifest.json` (source, query, as-of, row count, schema hash);
`experiment.py validate` checks the parquet against the manifest. The
experiment's `data.py` loads it offline:

```python
from lib.sources import read_snapshot

def load_dataset():
    return read_snapshot(EXPERIMENT_DIR)  # verifies integrity, returns a DataFrame
```

Each source has a `SETUP.md` (keyless auth + required read-only grant) under
[`lib/sources/bundles/`](lib/sources/bundles/); see
[`examples/warehouse/postgres/`](examples/warehouse/postgres/) for a creds-free
local Postgres demo. Reproducible as-of uses warehouse time travel
(Snowflake/BigQuery/Databricks) or a modeled time predicate (Redshift/Postgres).
The DuckDB and local-Postgres paths are tested end to end; the four proprietary
connectors are verified against mocked clients, not live warehouses.

## Tests

```bash
uv run ruff check .
uv run ruff format --check .
uv run --extra models pytest tests/ -v --tb=short
```
