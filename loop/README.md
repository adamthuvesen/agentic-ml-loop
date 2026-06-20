# Autonomous loop (`loop/`)

Supervisor that runs **repeatable cycles** on an experiment: each cycle builds a prompt from the experiment's files, invokes the configured **runner** CLI, then updates persisted state and terminal/`status.md` output.

## How it works

1. **`start`** — requires a clean experiment (no `loop_state.json` yet). Initializes state, acquires a **file lock** (`.loop.lock`), and enters **`run_loop`**.
2. **`resume`** — loads `loop_state.json` and continues until limits or completion.
3. Each **cycle** (`run_cycle` in `core.py`): capture baselines via **`artifacts.py`** → build prompt in **`prompts.py`** → per-attempt **`cycle_attempt.py`** (invoke + **`contracts.py`**) → progress via **`hooks.py`** → retry up to **`DEFAULT_MAX_ATTEMPTS_PER_CYCLE`**.
4. **Stop** via **`stop_policy.py`**: `max_cycles`, `max_hours`, **`EXPERIMENT_COMPLETE`** (unless `--run-until-limit`), stall/failure caps, or Ctrl+C.
   Final-holdout access is stronger than budget mode: once
   `final_holdout_accessed=true`, the supervisor stops instead of prompting for
   another search cycle.

State and human-readable status live under the experiment directory: **`loop_state.json`**, **`status.md`**, plus the lock file.

## Layout

| File              | Role                                                                                                                        |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `core.py`          | Locking, `run_cycle` / `run_loop` orchestration, CLI (`start` / `resume` / `status`)                                       |
| `loop_state.py`    | Typed `LoopState` persisted as `loop_state.json`                                                                            |
| `stop_policy.py`   | Ordered stop reasons (`evaluate_stop` / `should_stop`)                                                                      |
| `cycle_attempt.py` | Single runner attempt: invoke, validate, contract check                                                                     |
| `artifacts.py`     | `CycleBaselines`, snapshots, progress reasons, advisory rollback                                                            |
| `contracts.py`     | Completion markers and actionable validation errors for failed attempts                                                     |
| `prompts.py`       | Cycle prompt assembly, token-budget truncation, completion markers                                                          |
| `hooks.py`         | `CycleHooks` protocol; default pre/post cycle behavior                                                                      |
| `status.py`        | `status.md` Markdown rendering                                                                                              |
| `ui.py`            | Terminal banners and `emit_*` hooks                                                                                         |
| `__main__.py`      | `python -m loop` → `main()`                                                                                                 |

## Commands

From the repo root:

```bash
uv run python -m loop start experiments/<experiment_id>
uv run python -m loop start experiments/<experiment_id> --max-cycles 5 --max-hours 2
uv run python -m loop start experiments/<experiment_id> --max-cycles 10 --run-until-limit
uv run python -m loop resume experiments/<experiment_id>
uv run python -m loop status experiments/<experiment_id>
uv run python -m loop freeze experiments/<experiment_id> --candidate <candidate-id>
uv run python -m loop final-holdout experiments/<experiment_id>
uv run python -m loop ledger experiments/<experiment_id>
```

Use **`resume`** if `loop_state.json` already exists; **`start`** is for a new run only.
Use **`freeze`** to lock validation-time selection before final test access.
Use **`final-holdout`** to score only frozen external-target candidates and write
`outputs/final_holdout.json` without touching `results.json`. Use **`ledger`**
to rebuild `outputs/cycle_metrics.csv` from cycle summaries.

## Contract with the runner

The loop expects the runner to edit experiment artifacts (especially `research_journal.md` and `results.json`) and to end cycles with exactly one marker: `<promise>CYCLE_DONE</promise>` or `<promise>EXPERIMENT_COMPLETE</promise>` (see `prompts.py`). See repo **`AGENTS.md`** for the full experiment contract.

When an experiment has `diagnostics/summary.md`, the prompt builder may surface it
as advisory context. Diagnostics remain optional and do not change the loop's
validation contract.

When an experiment has `evaluation_review.md`, the prompt builder may surface the
latest evaluation concerns and proposed amendments. This review is advisory only;
the loop still rejects successful cycles that mutate `experiment.md`.

When `learnings.md` exists and cross-learnings are enabled, the prompt builder
retrieves only a bounded set of relevant learnings excerpts and renders a short
warm-start note. If no strong match exists, the learnings block is omitted.
