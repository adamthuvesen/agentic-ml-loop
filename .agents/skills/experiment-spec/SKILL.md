---
name: experiment-spec
description: Design and scaffold an agentic-ml-loop experiment from raw data + context. Profiles the data (local file or frozen warehouse snapshot), mines prior learnings and external references, proposes a rigorous, falsifiable experiment spec, generates the data/modeling/runner code, seeds research sources, runs trivial baselines, and validates. Use when a user wants to turn a dataset + problem description into a runnable experiment for the loop.
argument-hint: "<experiment-slug>"
---

# Experiment Spec

Design a new agentic-ml-loop experiment from raw inputs and scaffold everything the
loop needs to run it. The user has dropped data + a `context.md` into `lib/{slug}/`
(or pulled a warehouse snapshot into `experiments/{slug}/data/`). Your job: profile the
data, discover what's known about this problem shape, propose a falsifiable experiment
design, and — once confirmed — generate all the files and run the baselines.

**Experiment slug**: $ARGUMENTS

The deliverable `experiment.md` is **the loop's immutable north star**: the supervisor
injects it into every cycle prompt, the cycle contract rejects any cycle that mutates it,
and `experiment.py` + `lib/` parse specific fields out of it. So the spec must be two
things at once — a rigorous research plan *and* a machine-readable contract. The sections
below tell you how to make it both.

## NEVER

- Generate complex models (XGBoost, neural nets, ensembles) as baselines — establishing
  the trivial anchor is the skill's job; climbing the complexity ladder is the loop's job.
- Invent column names, row counts, or metrics that aren't in the actual data. Every number
  in the spec must come from profiling you ran.
- Write a hypothesis you can't ground in evidence (a profiling result, a `learnings.md`
  pattern, or a cited source) **and** can't falsify. Fewer real hypotheses beat padded ones.
- Re-implement what the framework already gives you. The runner is ~30 lines that import
  `init_experiment_dir` / `run_runner_main` from `lib.runner`; `CandidateResult` comes from
  `lib.candidate_result`. Do not hand-roll file locking, CLI parsing, or a result dataclass.
- Rename the machine-parsed `experiment.md` headings (see *Loop contracts* below). Add
  sections freely; never rename or drop the parsed ones.
- Skip the Phase 2 → Phase 3 confirmation. Stop and get the human's sign-off on the design.
- Modify any existing experiment, `lib/` module, or the shared framework. You only add new
  files under `lib/{slug}/`, `experiments/templates/{slug}*.md`, and `runners/{slug}_runner.py`.
- Generate more than 5 research hypotheses.

## Loop contracts — what `experiment.py` and `lib/` parse out of the spec

A spec that reads well but breaks these silently degrades the loop. Honor all of them; the
self-check in Phase 4 verifies most.

| Contract | Rule | Why |
| --- | --- | --- |
| **Objective-metric token** | `## Objective Metric` must contain a parenthesized identifier with a `val_`/`validation_` prefix, e.g. `(val_auc)`. | `experiment.py` extracts it and warns (errors under `--strict-completion`) if any `results.json` entry's `objective_metric` disagrees or the prefix is missing. The loop also strips the prefix to find the bare metric (`auc`) inside each candidate's `metrics["validation"]`. |
| **Parsed headings** | Keep these headings verbatim: `## Goal`, `## Problem Type`, `## Objective Metric`, `## Split Strategy`, `## Data Profile`, `## Constraints`, `## Known Risks`. | `lib/learnings.py` reads them to infer tags (problem type, split, model family, imbalance, …) that drive cross-experiment learnings retrieval. Wrong headings = no warm-start priors. |
| **Data-profile counts** | In `## Data Profile`, write literal `Row count: N` and `Feature count: N` lines. | Regex-extracted into size buckets (`small`/`medium`/`large-data`, `*-feature-set`) for learnings retrieval. |
| **Baseline recognizability** | The trivial baseline candidate must have `model_family` in `{"constant", "rule_based"}` **or** `candidate_id` containing `baseline`. | The "baseline still competitive" advisory signal keys off this to nudge the loop. |
| **Result shape** | Each candidate writes `metrics = {"train": {...}, "validation": {...}, "test": {...}}` with the bare metric name as the key; `objective_score` = the validation value; `objective_metric` = the `val_*` token. | Overfit/plateau signals read train-vs-validation and the bare metric name. `objective_score` must be a finite number or validation fails. |
| **Uncertainty convention** | When the loop bootstraps a CI, it stores `hyperparameters["{objective_metric}_ci_95"] = [lo, hi]` (e.g. `val_auc_ci_95`). State this in Evaluation Strategy so cycles follow it. | The plateau signal reads `*_ci_95` to tell real gaps from noise. |
| **Completion floor** (optional) | A line `Minimum loop cycles before EXPERIMENT_COMPLETE: N` anywhere forces ≥ N journal cycles before `EXPERIMENT_COMPLETE` is accepted. | Stops the loop from declaring victory after one cycle. Set it to a small honest number (e.g. 4). |
| **Cross-learnings toggle** (optional) | `cross_learnings: false` on its own line disables warm-start retrieval for this experiment. | Use only when prior learnings would mislead (genuinely novel problem shape). |

## Prerequisites

This skill supports two data sources. Detect which applies and verify before starting.

**Mode A — local file (default).** Verify:
1. `lib/{slug}/` exists and holds at least one data file (`.csv`, `.parquet`, `.json`, `.tsv`).
2. `lib/{slug}/context.md` exists (if not, ask the user to describe the problem — target,
   how predictions are used, any known leakage/constraints). See `lib/demo_bootstrap/context.md`
   for the shape.
3. `experiments/{slug}/` does NOT already exist (if it does, confirm before overwriting).

**Mode B — frozen warehouse snapshot.** The data was pulled once at the edge with
`python -m lib.sources pull --experiment experiments/{slug} --source <type> --query @q.sql`,
producing `experiments/{slug}/data/snapshot.parquet` + `dataset_manifest.json`. Verify:
1. The snapshot + manifest exist (`experiment.py validate` checks the parquet against the manifest).
2. A `context.md` or equivalent problem description exists (ask if missing).
3. The loop only ever reads the frozen snapshot — never the live warehouse. Record the
   `## Data Source` provenance (source, query, as-of) from the manifest in the spec.

**Discovering the snapshot (Mode B, if it doesn't exist yet).** Keep this simple. Use a
connected **read-only** warehouse MCP — or the source's own CLI — to browse the schema and
profile the candidate table(s): row count, columns, a few sample rows, target prevalence.
Then write the pull `SELECT` to a `.sql` file with **explicit columns**, an `ORDER BY` on a
stable key, and either an `{as_of}` placeholder (Snowflake/BigQuery/Databricks time travel)
or a literal time predicate (Redshift/Postgres/DuckDB). Dry-run to confirm it's reproducible,
then pull:

```bash
python -m lib.sources pull --experiment experiments/{slug} --source <type> \
  --query @q.sql --as-of '<ts>' --dry-run   # checks determinism, prints the rewritten SQL
python -m lib.sources pull --experiment experiments/{slug} --source <type> \
  --query @q.sql --as-of '<ts>'             # writes data/snapshot.parquet + manifest
```

The pull refuses unreproducible queries (random ordering/sampling, or a row cap without
`ORDER BY`) and warns on moving-time functions like `now()`/`current_date` — pin the moment
with `--as-of` instead. Discovery is a one-time edge step; never query the warehouse from a
cycle. See each source's `lib/sources/bundles/<type>/SETUP.md` for keyless auth and the
read-only grant it needs.

Read `context.md` and all data; also read any notebooks (`.ipynb`) or markdown in the
input dir — they often carry baseline code, feature definitions, or domain context.

---

## Phase 1: Profile and Read Context

### 1a. Profile the data

Write a throwaway profiling script and run it against the real data — local file via pandas,
or the snapshot via `from lib.sources import read_snapshot` (the experiment `scripts/` folder
doesn't exist until Phase 3, so keep this script out of the way). Extract and record:

- Row count, column count.
- Per-column dtype (numeric / categorical / datetime / boolean / text) and cardinality.
- Missing-value % per column (flag > 10%).
- **Target**: class balance (classification) or distribution stats — mean/median/skew, zero
  fraction, right-tail share (regression). Note rare-event or zero-inflated regimes.
- **Effective sample size & feature budget**: for classification, the positive count; the
  events-per-variable (EPV) ceiling is ~N_positives / 10–20 features. State the safe feature
  count so the spec doesn't invite an over-parameterized search.
- Time column range and frequency (if any) → candidate temporal split cutoff.
- Top correlations with target (flag any suspiciously high single-feature correlation — a
  leakage smell).
- **Cheap leakage / integrity probes**: exact-duplicate rows; near-constant features;
  IDs that correlate with the target; for time data, whether the target rate drifts across
  the proposed train/val/test windows.

### 1b. Read all context

From `context.md` and any notebooks, extract: explicit target; problem type; split
preferences (temporal / grouped / stratified); known leakage columns or constraints;
existing baseline code or heuristic; business context (how predictions are used, cost
asymmetry of FP vs FN); metric preferences.

---

## Phase 2: Discover and Propose (interactive — WAIT for user confirmation)

### 2a. Cross-experiment learnings

Read `learnings.md` at the repo root. Retrieval in the loop is **tag-based**: the spec's
`Problem Type` / `Objective Metric` / `Split Strategy` / `Data Profile` / `Goal` /
`Constraints` / `Known Risks` sections (plus row/feature-count buckets) are matched against
each learning section's tags. So two things follow: (1) you get better warm-start priors by
writing those sections with honest, specific keywords; (2) read the matching sections now and
pull anything relevant into your priors. Look for patterns like:

- "Pre-aggregated features impose a hard ceiling" → informs candidate families and hypotheses.
- "Linear baselines match trees on small-data classification" → informs hypothesis ordering.
- "Class weighting often hurts AUC ranking" → informs known risks.

Hold findings for step 2d.

### 2b. Web discovery

Search for 2–4 references matched to *this* problem shape, using the profiling results to
build targeted queries (similar Kaggle competitions; known techniques for the regime, e.g.
"two-stage hurdle model zero-inflated regression"; common pitfalls, e.g. "temporal leakage
tabular"). Prefer seeding **known-good reference approaches** for the data type (tabular →
regularized linear, then gradient-boosted trees with proper CV) so the loop starts from sane
priors rather than inventing fragile pipelines. For each reference capture: title + URL; the
key insight (1–2 sentences); what transfers here and what doesn't.

### 2c. Orientation summary

Give the user a brief orientation: what prior experiments reveal (from `learnings.md`); what
the web suggests; the data characteristics that shape the design (from profiling); and the
risks/confounders found so far.

### 2d. Enhanced proposal

Present a structured proposal. Every populated field must trace to profiling, `learnings.md`,
or a cited source — no generic filler.

```
## Experiment Proposal: {slug}

**Problem type:** binary classification / regression / ranking / …
**Target column:** {column}
**Split strategy:** {temporal cutoff / grouped by {key} / stratified} — {rationale}.
  Temporal data: train on the past, test on the future; never random k-fold.
  Repeated-entity data: keep each entity wholly in one split.
**Objective metric:** {metric} with the (val_{metric}) token — {why this metric fits the task}
**Operating metric:** {precision@k / F-beta / Brier / MAE} — {the deployment decision it reflects}
**Baselines:** {trivial anchor — majority-class or median} {+ user heuristic / human baseline if any}
**Leakage controls (ruled out, with reason):**
  - clean split, preprocessing fit on train only, no train/test duplicates
  - each feature legitimately available at prediction time (flag any proxy-for-target)
  - split respects time order and entity grouping
**Constraints / non-goals:** {interpretability, latency, feature availability, compute budget}
**Data profile summary:** {row/feature counts, missingness, class balance or target dist,
  N positives & EPV feature budget, key correlations}
**Known risks:** {imbalance quantified, scale confounders, drift indicators, label noise}
**Research pointers:** {2–4 concrete references from 2b}
**Evaluation strategy:** {split sizes; single final-test rule; significance via paired
  bootstrap (CI on the metric AND on the delta vs baseline); treat deltas within the CI as
  no-change; slices to report}
**Success definition:** {specific & checkable — e.g. "Beat the constant baseline AUC with a
  bootstrap CI on the delta excluding 0; determine whether trees add value over linear given
  N positives." } {+ Minimum loop cycles before EXPERIMENT_COMPLETE: N}

**Research hypotheses (ranked, most promising first):**
  H1: …
  H2: …
```

### Research hypothesis format

Each hypothesis is a falsifiable claim the loop will treat as a starting prior. Give it:

- **Claim** — what to try or investigate, as one bounded change (one variable per test).
- **Why it might work** — grounded in a specific profiling result, `learnings.md` pattern, or cited source.
- **Expected effect & noise** — rough direction *and* magnitude vs the validation noise floor.
  If the predicted gain is smaller than one bootstrap-CI width, say so — it may not be worth a cycle.
- **Falsifier** — the specific metric/observation that would disconfirm it (e.g. "val R² gain
  < 0.02, or the train-val gap widens").
- **Confidence** — high / medium / speculative, with a one-word read on feasibility within a cycle.

Design rules: 3–5 hypotheses, ordered by prior probability; each falsifiable; grounded, not
speculative; focused on the *what* and *why*, not the *how* (the loop chooses implementation).
If you can't ground one in evidence, drop it.

**STOP HERE and ask the user to confirm or adjust.** Do not proceed to Phase 3 until approved.

---

## Phase 3: Generate (after user confirms)

Read these reference files first and follow their patterns exactly:

- **Rich filled spec (the quality bar):** `experiments/demo_regression/experiment.md` —
  Data Profile, Known Risks, Evaluation Strategy, ranked hypotheses, all evidence-backed.
- **Seeded sources example:** `experiments/demo_regression/research_sources.md`.
- **Data module pattern:** `lib/demo_classification/data.py` (local) — and the warehouse
  loader snippet in `README.md` (`read_snapshot`).
- **Modeling pattern:** `lib/demo_classification/modeling.py` — copy the *structure*
  (preprocessor, per-split evaluation, `CandidateResult`, `CANDIDATE_RUNNERS`), but generate
  **only the trivial baseline**, not the demo's trees.
- **Runner pattern (tiny):** `runners/demo_bootstrap_runner.py`.
- **Spec template / fallback:** `experiments/templates/model-search.md`.
- **Scaffolder:** `lib/runner.py` (`init_experiment_dir`, `run_runner_main`) and
  `lib/candidate_result.py` (`CandidateResult`).

The slug uses hyphens in prose but Python modules use underscores. Convert consistently
(`my-experiment` → `lib/my_experiment/`).

### Step 1 — `lib/{slug}/__init__.py`

Empty file.

### Step 2 — `lib/{slug}/data.py`

Mirror `lib/demo_classification/data.py`:

- `TARGET_COLUMN`, `TIME_COLUMN` (if temporal), column-list constants (numeric / categorical /
  boolean), and an `EXCLUDE_COLUMNS` set (IDs, target, and any leakage columns).
- A frozen `@dataclass` for splits (`train` / `validation` / `test`).
- `load_dataset()`:
  - **Mode A:** `DATA_PATH = Path(__file__).parent / "<file>"`, read with the right reader and dtypes.
  - **Mode B (warehouse):**
    ```python
    from lib.sources import read_snapshot
    EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "experiments" / "{slug}"
    def load_dataset() -> pd.DataFrame:
        return read_snapshot(EXPERIMENT_DIR)  # verifies integrity, returns a DataFrame
    ```
- `split_dataset(df)`: implement the confirmed strategy — temporal (sort by `TIME_COLUMN`,
  cut by time), grouped (keep entities whole), or stratified. Validate it produces non-empty
  splits and, for classification, ≥ 2 classes in each (a single-class validation split makes
  AUC undefined and fails the finite-`objective_score` contract).
- Feature-getter helpers as needed.

### Step 3 — `lib/{slug}/modeling.py`

Mirror the *structure* of `lib/demo_classification/modeling.py`:

- `from lib.candidate_result import CandidateResult` (do not redefine it).
- `OBJECTIVE_METRIC = "val_{metric}"` (the `val_*` token) and `SPLIT_STRATEGY = "..."`; seed a `RANDOM_STATE`.
- A per-split evaluation helper returning `{"train": {...}, "validation": {...}, "test": {...}}`
  where each dict is keyed by the bare metric name (e.g. `"auc"`). `objective_score` is the
  validation value.
- **Baseline candidate(s) only — trivial, no learned model:**
  - Classification: majority-class probability — `np.full(len(y), train_positive_rate)`,
    `model_family="constant"`.
  - Regression: predict the train median — `np.full(len(y), train_median)`, `model_family="constant"`.
  - If the user supplied a heuristic/rule baseline, adapt it too (`model_family="rule_based"`).
  - Name it so the baseline signal recognizes it (`candidate_id="majority-baseline"` etc.).
- `CANDIDATE_RUNNERS = {candidate_id: runner_fn, ...}`.

### Step 4 — `experiments/templates/{slug}.md` (the filled spec)

Write the confirmed proposal as a full `experiment.md`, following the
`experiments/demo_regression/experiment.md` structure and the *Loop contracts* table.
Required, in this order where they map to parsed headings:
`## Title`, `## Goal`, `## Baseline`, `## Dataset`, `## Data Source` (Mode B only),
`## Data Profile` (with `Row count:` / `Feature count:` lines), `## Target`,
`## Problem Type`, `## Split Strategy`, `## Objective Metric` (with the `(val_*)` token),
`## Candidate Families`, `## Constraints`, `## Known Risks` (leakage controls ruled out,
drift, label noise), `## Evaluation Strategy`, `## Research Pointers`,
`## Research Hypotheses`, `## Success Definition` (with the optional `Minimum loop cycles…`
line), `## Deliverables` (every path under `outputs/…`).

This file is the seed the runner's `init-demo` copies into `experiments/{slug}/experiment.md`.

### Step 5 — `experiments/templates/{slug}_research_sources.md` (seeded discovery)

Write the Phase 2b findings as proper source cards, following
`experiments/demo_regression/research_sources.md`:

- **Reusable Takeaways** — 2–4 scoped heuristics (with caveats) synthesizing what the
  references suggest: techniques that tend to work, pitfalls to avoid, metrics that matter.
- **Source Cards** — one per reference: Type (kaggle / paper / blog / docs); URL; why
  relevant; key takeaways; applicability & caveats; ideas it suggests; Status. Use a real
  title in the heading (`### Source 001: <real title>`) — the loop counts cards whose heading
  still contains the literal `<title>` placeholder as empty.

### Step 6 — `runners/{slug}_runner.py` (tiny)

Follow `runners/demo_bootstrap_runner.py` exactly — no hand-rolled CLI or locking:

```python
from __future__ import annotations

from pathlib import Path

from lib.{slug}.data import load_dataset, split_dataset
from lib.{slug}.modeling import CANDIDATE_RUNNERS
from lib.runner import init_experiment_dir, run_runner_main

EXPERIMENT_ID = "{slug}"
_TEMPLATES = Path(__file__).resolve().parents[1] / "experiments" / "templates"
TEMPLATE_PATH = _TEMPLATES / "{slug}.md"
RESEARCH_SOURCES_TEMPLATE_PATH = _TEMPLATES / "{slug}_research_sources.md"


def _load_splits():
    return split_dataset(load_dataset())


def init_demo(force: bool = False) -> Path:
    return init_experiment_dir(
        EXPERIMENT_ID, TEMPLATE_PATH, RESEARCH_SOURCES_TEMPLATE_PATH, force=force
    )


if __name__ == "__main__":
    raise SystemExit(
        run_runner_main(
            EXPERIMENT_ID,
            CANDIDATE_RUNNERS,
            _load_splits,
            TEMPLATE_PATH,
            RESEARCH_SOURCES_TEMPLATE_PATH,
        )
    )
```

### Step 7 — Scaffold the experiment directory

`init_experiment_dir` (via the runner's `init-demo`) seeds `experiment.md`,
`research_journal.md`, `research_sources.md`, `results.json` (`[]`), and the `outputs/` /
`work/` / `scripts/` folders idempotently — do not hand-create them.

```bash
uv run python runners/{slug}_runner.py init-demo
```

(Mode B: do **not** pass `--force` after a snapshot is pulled — `--force` deletes
loop-managed dirs but preserves `data/`; still, prefer a plain `init-demo` here.)

### Step 8 — Run baselines and validate

```bash
uv run python runners/{slug}_runner.py list-candidates
uv run python runners/{slug}_runner.py run-candidate --experiment experiments/{slug} --candidate {baseline_id}
uv run python experiment.py validate experiments/{slug}
```

If anything fails, fix it and re-run. Do not leave a broken experiment.

---

## Phase 4: Verify (self-check) and report

Before declaring the experiment ready, confirm every item:

- [ ] `experiment.py validate experiments/{slug}` passes (warnings OK; no errors, no stray-root files).
- [ ] `## Objective Metric` contains a `(val_*)` token, and each `results.json` entry's
      `objective_metric` matches it.
- [ ] `results.json` has ≥ 1 baseline entry with a **finite** `objective_score`, and the
      baseline is recognizable (`model_family` ∈ {`constant`, `rule_based`} or id contains `baseline`).
- [ ] `## Data Profile` has literal `Row count:` / `Feature count:` lines, and every number
      in the spec traces to profiling you ran (no invented values).
- [ ] Parsed headings are present and unrenamed; Mode B has a `## Data Source` block.
- [ ] ≥ 1 hypothesis with a real falsifier; none ungrounded.
- [ ] Known Risks rules out each leakage type with a reason; the split matches the data
      structure (temporal / grouped).
- [ ] Deliverables live under `outputs/…`.

Then print:

```
## Experiment Ready: {slug}

**Mode:** local file | warehouse snapshot ({source})
**Files created:**
- lib/{slug}/__init__.py, data.py, modeling.py
- experiments/templates/{slug}.md, {slug}_research_sources.md
- runners/{slug}_runner.py
- experiments/{slug}/ (experiment.md, research_journal.md, research_sources.md [N cards], results.json)

**Problem:** {type} · target `{target}` · {row}×{feat} · {split strategy}
**Objective:** {val_metric}  ·  Operating: {operating_metric}
**Baseline:** {baseline_id} {val_metric}={score}
**Research hypotheses:** {count}

**Start the loop:**
  uv run python -m loop start experiments/{slug}
```
