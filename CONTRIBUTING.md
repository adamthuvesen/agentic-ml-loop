# Contributing

Thanks for taking a look at Agentic ML Loop.

## Local Setup

```bash
uv sync --extra models --group dev
uv run pre-commit install
```

## Before Opening A Pull Request

Run the same checks used by CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --extra models pytest tests/ -v --tb=short
uv run python experiment.py validate experiments/demo_bootstrap
```

Keep example data synthetic and reproducible. Do not commit local datasets,
generated notebooks, experiment logs, credentials, or private business context.

## Runner Changes

Runner commands must accept the cycle prompt on stdin and write either plain text
or newline-delimited stream JSON to stdout. The final assistant text must include
exactly one completion marker:

- `<promise>CYCLE_DONE</promise>`
- `<promise>EXPERIMENT_COMPLETE</promise>`
