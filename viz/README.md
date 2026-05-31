# Experiment replay (`viz/`)

Turns experiment artifacts into a **standalone HTML replay**: a small canvas animation that walks through research cycles (plan → research → lab → scoreboard → journal → finale).

## How it works

1. **`generate.py`** reads `experiments/<id>/experiment.md`, `results.json`, and `research_journal.md`, normalizes candidate metrics, builds a list of **scenes**, and writes **`output/<id>/script.json`**.
2. **`bundle.py`** embeds that JSON (HTML-safe) and inlines all **`renderer/*.js`** into **`output/<id>/replay.html`** — one file you can open in a browser (no server required).

Development: open `renderer/index.html` with separate script tags; bundled replays use the single-file output.

## Metric support

Replay generation adapts to the objective metric family:

- Captured-at-K / value capture: preserves `at_10`, `at_20`, `at_30`, `at_50`, and related validation metrics when present.
- Classification: surfaces validation metrics such as AUC, precision, recall, average precision, log loss, or Brier score without inventing captured-at-K fields.
- Regression: surfaces validation metrics such as R2, RMSE, MAE, and MAPE without inventing captured-at-K fields.
- Unknown metrics: keeps candidate id, objective metric, objective score, notes, and available validation metrics.

Candidate summaries always include common fields: `id`, `family`, `objective_metric`, `objective_score`, `score`, `primary_metric`, `metrics`, and `notes`.

## Layout

| Path          | Role                                                                                  |
| ------------- | ------------------------------------------------------------------------------------- |
| `generate.py` | Parse experiment files → `script.json`                                                |
| `bundle.py`   | `script.json` + `renderer/` → single `replay.html`                                    |
| `renderer/`   | `index.html`, canvas + keyboard UI, plain JS (engine, scenes, room, character, sound) |
| `templates/`  | Static data for the room layout (`room_layout.json`)                                  |
| `output/`     | Generated `script.json` and `replay.html` per experiment                              |

## Commands

From the repo root:

```bash
uv run python viz/generate.py experiments/<experiment_id>
uv run python viz/bundle.py viz/output/<experiment_id>/script.json
```

Then open `viz/output/<experiment_id>/replay.html`.

Optional second argument to bundle sets the output path:

```bash
uv run python viz/bundle.py viz/output/<experiment_id>/script.json /path/to/replay.html
```

## Tests

`tests/test_viz.py` covers JSON escaping, validation-total helpers, and metric-adaptive result parsing for captured-at-K, classification, regression, and unknown metrics.
