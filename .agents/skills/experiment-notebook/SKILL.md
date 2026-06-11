---
name: experiment-notebook
description: Export an agentic-ml-loop experiment into a portable runnable Jupyter notebook from its notebook.yaml recipe. Use when the user asks to create, regenerate, or inspect a shareable experiment notebook that should run from a parquet file without importing repo-local experiment modules.
---

# Experiment Notebook

Export an experiment into a **portable** Jupyter notebook — one that runs from a single
parquet via `DATA_PATH` with **no repo-local imports** (`lib.*`, `experiments.*`), so it can
be shared and run outside this repo. The `notebook.yaml` recipe is the source of truth; the
exporter renders the successful start-to-end path, not the full loop history.

## Workflow

1. Confirm the target experiment path, usually `experiments/<experiment_id>`.
2. Check for `notebook.yaml` in that experiment.
   - If present, use it as the source of truth.
   - If missing, inspect `experiment.md`, `outputs/`, `results.json`, and `scripts/` enough to draft the smallest recipe; ask only if the runnable path is unclear.
3. Generate the notebook:

```bash
uv run python -m lib.notebook_export experiments/<experiment_id>
```

Use `--include-sensitive` only after the user explicitly asks to include row-level or sensitive outputs:

```bash
uv run python -m lib.notebook_export experiments/<experiment_id> --include-sensitive
```

4. Verify the notebook JSON and check that code cells do not import repo-local experiment modules:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("experiments/<experiment_id>/outputs/<notebook>.ipynb")
nb = json.loads(path.read_text())
source = "\n".join(
    "".join(cell.get("source", []))
    for cell in nb["cells"]
    if cell["cell_type"] == "code"
)
assert "from lib." not in source
assert "import lib." not in source
assert "from experiments." not in source
print(path)
PY
```

## Notes

- The notebook runs with `DATA_PATH` pointed at the source parquet — confirm that's the only input it needs.
- Default exports exclude recipe-declared sensitive outputs; pass `--include-sensitive` only on explicit request.
- Generated notebooks land in `experiments/<id>/outputs/` and are gitignored unless intentionally kept.
- Keep them focused on the successful start-to-end path, not every loop cycle or failed candidate branch.
