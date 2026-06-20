# Agentic ML Loop

`agentic-ml-loop` runs local, repeatable ML experiment cycles. You write a spec,
choose a runner, and the loop asks for one bounded hypothesis test at a time
while maintaining a journal and leaderboard.

Cycles are checked in code ([`loop/cycle_attempt.py`](loop/cycle_attempt.py),
[`loop/contracts.py`](loop/contracts.py)): the journal must be updated,
`experiment.md` must stay unchanged, exactly one completion marker must appear,
and results must match the expected schema and metrics. Failed cycles roll back
and retry.

The repo ships with deterministic synthetic demos only; no external data source
is required. Worked example in [`examples/`](examples/). Internals:
[`.agents/docs/architecture.md`](.agents/docs/architecture.md).

## Setup

Requires Python 3.12+ and `uv`.

```bash
uv sync --extra models --extra deep   # extras optional; dev group installs by default
uv run pre-commit install
```

## Fastest demo

Use `demo_bootstrap` for the quickest trust check. It is a tiny synthetic
classification experiment with committed results.

```bash
uv run --extra models python runners/demo_bootstrap_runner.py list-candidates
uv run python experiment.py validate experiments/demo_bootstrap
```

Expected output includes six candidate ids, ending with `lgbm-conservative`,
then `Validation passed: experiments/demo_bootstrap`. In this checkout, both
commands finished in under 10 seconds after dependencies were present. The first
`uv` run may take longer while it prepares the environment.

Open [`examples/demo_bootstrap_replay.html`](examples/demo_bootstrap_replay.html)
in a browser to replay the five committed research cycles. It is generated from
[`experiments/demo_bootstrap/research_journal.md`](experiments/demo_bootstrap/research_journal.md)
and [`experiments/demo_bootstrap/results.json`](experiments/demo_bootstrap/results.json).

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

`loop/` starts a fresh runner process per cycle. The runner receives the cycle
prompt on stdin and must end with exactly one marker:
`<promise>CYCLE_DONE</promise>` or `<promise>EXPERIMENT_COMPLETE</promise>`.

## Runners

Presets cover Claude, Codex, and Cursor; custom commands work too. The presets
run unattended, without approval prompts, and with full workspace permissions.

```bash
uv run python -m loop start experiments/demo_bootstrap --runner claude
uv run python -m loop start experiments/demo_bootstrap --runner codex --runner-model gpt-5.5-high --runner-effort high
uv run python -m loop start experiments/demo_bootstrap --runner-command "claude --print --verbose --output-format stream-json"
```

| Runner | Default command |
| --- | --- |
| Claude | `claude --print --verbose --output-format stream-json --permission-mode bypassPermissions --model opus` (`claude-opus-4-8-high` is recorded as the requested model) |
| Codex | `codex exec --dangerously-bypass-approvals-and-sandbox --model gpt-5.5-high` |
| Cursor | `cursor-agent --print --trust --force --sandbox disabled --model composer-2.5` |

`--runner-effort` maps to `--effort` for Claude and
`-c model_reasoning_effort=<effort>` for Codex; Cursor has no effort flag, so
pick a model id that encodes it. Environment defaults: `AGENTIC_ML_LOOP_RUNNER`,
`AGENTIC_ML_LOOP_RUNNER_COMMAND`, `AGENTIC_ML_LOOP_RUNNER_MODEL`,
`AGENTIC_ML_LOOP_RUNNER_EFFORT`, `AGENTIC_ML_LOOP_RUNNER_TIMEOUT` (seconds).
Built-in runners record the requested model and the resolved CLI model
separately; this lets the Claude preset keep the intended Opus model while using
the local CLI's accepted `opus` alias.

Troubleshooting: presets shell out to `claude`, `codex`, or `cursor-agent`,
which must be on `PATH`. Otherwise pass an absolute path via `--runner-command`
or `AGENTIC_ML_LOOP_RUNNER_COMMAND`. The cross-experiment learnings step also
calls `claude`; when absent, it is skipped rather than failing the loop.

## Research referee

After each cycle the loop writes an advisory scorecard
([`lib/referee.py`](lib/referee.py)); it never blocks a cycle:

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

Run the same spec across several runners:

```bash
uv run python -m loop bench experiments/demo_bootstrap --runners claude,codex,cursor --max-cycles 6
```

Each runner gets an isolated copy of the spec and the same cycle budget, with
the referee on. Results go to `bench/<id>-<timestamp>/comparison.md` and
`comparison.csv`, ordered by mean referee score, then best validation score. A
runner error is recorded and does not stop the others.

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

Synthetic datasets are generated in memory when missing. Generator scripts can
materialize local CSVs, which git ignores.

`loop status` needs an initialized loop state; before a loop run it reports
`No loop state found`.

## Selection freeze and final holdout

For experiments with a locked test/holdout split, freeze model selection before
test access:

```bash
uv run python -m loop freeze experiments/<id> --candidate <validation-winner> \
  --reason "validation winner"
uv run python -m loop final-holdout experiments/<id>
uv run python -m loop ledger experiments/<id>
```

`freeze` records the validation-selected candidate ids in `loop_state.json`.
After freeze, shared candidate runners refuse normal `results.json` writes.
`final-holdout` is only available after freeze; it writes test metrics to
`outputs/final_holdout.json`, marks `final_holdout_accessed=true`, and the loop
will not continue even under `--run-until-limit`. `ledger` rebuilds
`outputs/cycle_metrics.csv` deterministically from `cycles/*/cycle_summary.json`.

External targets that depend on private or gitignored snapshots should add a
small committed fixture manifest based on
[`experiments/templates/external-target-fixture-contract.json`](experiments/templates/external-target-fixture-contract.json).
Validate a manifest with:

```bash
uv run python -m lib.external_targets.fixtures path/to/fixture-contract.json
```

## Portable notebooks

Experiments can declare a notebook recipe in `experiments/<id>/notebook.yaml`:

```bash
uv run python -m lib.notebook_export experiments/<id>
```

Generated notebooks land in `experiments/<id>/outputs/`, which git ignores.

## Warehouse ingestion (optional)

Pull a frozen snapshot from Snowflake, BigQuery, Redshift, Databricks, or
Postgres, then run the loop offline. The warehouse is touched once; cycles never
query it. Clients are optional extras, and the default install stays `numpy` +
`pandas`.

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
