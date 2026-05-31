---
name: experiment-spec
description: Design and scaffold a agentic-ml-loop experiment from raw data + context. Profiles data, discovers prior learnings and external references, proposes a rich experiment spec with research hypotheses, generates all files, runs baselines.
argument-hint: "<experiment-slug>"
---

# Experiment Spec

Design a new agentic-ml-loop experiment from raw inputs. The user has dropped data files and a
`context.md` into `lib/{slug}/`. Your job: profile the data, discover what's known
about this problem shape, propose an experiment design with research hypotheses, and —
once confirmed — generate all the files needed to run the loop.

**Experiment slug**: $ARGUMENTS

## NEVER

- Generate complex models (XGBoost, neural nets, ensembles) as baselines — the loop's job is to be clever
- Invent column names that don't exist in the data
- Skip the user confirmation step between Phase 2 and Phase 3
- Deviate from the existing file patterns without calling it out
- Modify any existing experiments or lib modules
- Generate more than 5 research hypotheses
- Write speculative hypotheses without citing evidence from profiling, learnings.md, or web discovery

## Prerequisites

Before starting, verify:
1. `lib/{slug}/` exists and contains at least one data file (`.csv`, `.parquet`, `.json`, `.tsv`)
2. `lib/{slug}/context.md` exists (if not, ask the user to describe the problem)
3. `experiments/{slug}/` does NOT already exist (if it does, confirm with user before overwriting)

Read the user's `context.md` and all data files. Also read notebooks (`.ipynb`) or
other markdown files in `lib/{slug}/` — they may contain baseline code, feature
descriptions, or domain context.

---

## Phase 1: Profile and Read Context

### 1a. Profile the data

Write and run a Python profiling script against the data. Extract:
- Row count, column count
- Column dtypes (numeric, categorical, datetime, boolean, text)
- Missing value percentages per column (flag columns with >10% missing)
- Target column distribution (class balance for classification, stats for regression)
- Time column range and frequency (if applicable)
- Top correlations with target
- Cardinality of categorical columns
- Basic outlier detection

### 1b. Read all context

Read `context.md` and any notebooks/markdown in `lib/{slug}/`. Look for:
- Explicit target column
- Problem type (classification, regression, ranking)
- Split strategy preferences (temporal, stratified, etc.)
- Known leakage columns or constraints
- Existing baseline code or heuristics
- Business context (cost asymmetry, how predictions are used)
- Metric preferences

---

## Phase 2: Discover and Propose (interactive — WAIT for user confirmation)

### 2a. Cross-experiment learnings

Read `learnings.md` at the repo root. Match sections by tag overlap with what you
learned from profiling — problem type, data size bucket, metric family, model
families, split type, and flags like imbalance or zero-inflated.

Extract relevant patterns. Examples of what to look for:
- "Pre-aggregated features impose a hard ceiling" → informs candidate families and hypotheses
- "Linear baselines match trees on small-data classification" → informs hypothesis ordering
- "Class weighting often hurts AUC ranking" → informs known risks

Hold findings in memory for step 2d.

### 2b. Web discovery

Search for 2-4 references relevant to this specific problem shape. Use data
profiling results to build targeted queries:
- Similar Kaggle competitions (e.g., "churn prediction tabular imbalanced kaggle")
- Known techniques for this problem shape (e.g., "best models for zero-inflated regression")
- Common pitfalls (e.g., "temporal leakage tabular classification")

For each reference, capture:
- Title and URL
- Key insight (1-2 sentences)
- Applicability to this experiment (what transfers, what doesn't)

### 2c. Orientation summary

Present a brief orientation to the user summarizing what you discovered:
- What prior agentic-ml-loop experiments reveal about this problem shape (from learnings.md)
- What web research suggests about effective approaches
- Key data characteristics that will shape the experiment (from profiling)
- Potential risks and confounders identified so far

### 2d. Enhanced proposal

Present a structured proposal covering:

```
## Experiment Proposal: {slug}

**Problem type:** binary classification / regression / etc.
**Target column:** {column_name}
**Split strategy:** {temporal / stratified / etc.} — {rationale}
**Objective metric:** {metric} — {why this metric}
**Leakage columns:** {list} — {why they leak}
**Constraints:** {list}
**Baseline approach:** {description of the trivial baseline}
**Data profile summary:** {key stats}

**Candidate families:** {informed by discovery — e.g., "Logistic regression
  (learnings.md: linear models match trees on pre-aggregated data at this
  sample size), LightGBM (if nonlinear signal exists)"}

**Known risks:** {populated from profiling — class imbalance quantified,
  missing data patterns, temporal drift indicators, effective sample size}

**Research pointers:** {2-4 concrete references from web discovery}

**Evaluation strategy:** {CV scheme, secondary metrics, significance
  requirements — informed by problem characteristics}

**Success definition:** {specific and checkable — e.g., "Beat constant
  baseline AUC significantly (bootstrap p < 0.05); determine if tree models
  add value over linear given N positives"}

**Research hypotheses:**
  H1: {title} — {why, signal, confidence}
  H2: ...
  H3: ...
```

### Research hypothesis format

Each hypothesis should have:
- **Title** — what to try or investigate
- **Why it might work** — grounded in evidence (profiling result, learnings.md pattern, or web reference)
- **Signal to look for** — specific metric or observation that confirms/disconfirms
- **Prior confidence** — high / medium / speculative

Design principles:
- 3-5 hypotheses, ordered by prior probability (most promising first)
- Each must be falsifiable — "Signal to look for" defines what success/failure looks like
- Grounded in evidence, not speculation
- Focused on the *what* and *why*, not the *how* — the loop agent decides implementation
- If you can't ground a hypothesis in evidence, have fewer hypotheses rather than padding

**STOP HERE and ask the user to confirm or adjust.** Do not proceed to Phase 3
until the user approves.

---

## Phase 3: Generate (after user confirms)

Read these reference files BEFORE generating anything — follow their patterns exactly:

- **Data module pattern:** `lib/demo_classification/data.py`
- **Modeling pattern:** `lib/demo_classification/modeling.py`
- **Runner pattern:** `runners/demo_classification_runner.py`
- **Experiment template:** `experiments/templates/model-search.md`
- **Filled experiment example:** `experiments/demo_classification/experiment.md`

### Step 1: Create `lib/{slug}/__init__.py`

Empty file.

### Step 2: Generate `lib/{slug}/data.py`

Follow the exact pattern from `lib/demo_classification/data.py`:

- `TARGET_COLUMN` constant
- `TIME_COLUMN` constant (if applicable, else omit)
- Column list constants (numeric, categorical, boolean — as appropriate for the data)
- `EXCLUDE_COLUMNS` set (leakage columns, IDs, targets that shouldn't be features)
- `DATA_PATH` pointing to the data file in `lib/{slug}/`
- A frozen `@dataclass` for splits (e.g., `ExperimentSplits` with train/validation/test)
- `load_dataset()` function: reads the file, handles dtypes, validates
- `split_dataset(df)` function: implements the confirmed split strategy, returns the dataclass
- Feature getter helpers as needed

### Step 3: Generate `lib/{slug}/modeling.py`

Follow the exact pattern from `lib/demo_classification/modeling.py`:

- `OBJECTIVE_METRIC` and `SPLIT_STRATEGY` constants
- `CandidateResult` frozen dataclass with `result_payload()` method
- Evaluation helper functions appropriate for the problem type
- **Baseline candidates:**
  - If user provided a baseline (in context.md, notebook, or code) → adapt it to the `CandidateResult` pattern
  - If no baseline provided → generate the simplest possible heuristic:
    - **Classification:** majority class predictor (literally `np.full(len(y), majority_class_probability)`)
    - **Regression:** predict median (literally `np.full(len(y), train_median)`)
  - The baseline must be trivial — no learned models, no feature engineering
- `CANDIDATE_RUNNERS` dict mapping candidate IDs to runner functions

### Step 4: Generate `runners/{slug}_runner.py`

Follow the exact pattern from `runners/demo_classification_runner.py`:

- Run runners via `uv run python runners/<slug>_runner.py` from repo root (editable install; no `sys.path` hacks)
- Import from `lib.{module}.data` and `lib.{module}.modeling`
- `save_candidate_result()` with file locking (fcntl)
- `list_candidates()` function
- `build_parser()` with subcommands: `list-candidates`, `run-candidate`, `init-demo`
- `init_demo(force)` function that creates/refreshes the experiment directory
- `EXPERIMENT_MD` string constant (copy of the generated experiment.md content)

Note: the slug uses hyphens (`my-experiment`) but Python modules use underscores
(`my_experiment`). Handle this conversion.

### Step 5: Scaffold experiment directory

Create `experiments/{slug}/` with:
- `experiment.md` — filled from the confirmed proposal (using the template structure, all sections populated including Research Hypotheses). Deliverable paths in this spec MUST live under `outputs/<filename>` (never flat at the experiment root).
- `research_journal.md` — use the template from `experiment.py`'s `research_journal_template()`
- `research_sources.md` — seeded with discovery findings (see Step 6)
- `results.json` — empty `[]`
- `feedback.json` — default structure: `{"usefulness_score": null, "accepted_candidate_ids": [], "rejected_candidate_ids": [], "missing_checks": [], "notes": ""}`
- `outputs/.gitkeep`, `work/.gitkeep`, `scripts/.gitkeep` — the three universal output folders. `init_experiment_dir()` creates these idempotently; this is the layout every cycle writes into via `lib.paths.outputs_dir / work_dir / scripts_dir`. Do not let cycle scripts write CSVs or summaries directly to the experiment root — the validator surfaces that as a warning.

### Step 6: Seed research_sources.md

Write the discovery findings from Phase 2b into `research_sources.md` as proper
source cards. Follow the existing source card format:

- **Reusable Takeaways** — synthesis of what the references suggest for this problem
  (2-4 bullet points: what techniques tend to work, what pitfalls to avoid, what
  metrics matter most)
- **Source Cards** — one per web reference found in 2b, each with:
  - Type (kaggle / paper / blog / documentation)
  - URL
  - Why relevant
  - Key takeaways
  - Applicability and caveats
  - Ideas to try
  - Status: reviewed

These sources are grounded in data profiling — they are targeted references, not
generic pre-fills. The loop agent will add more sources during its own research
cycles.

### Step 7: Run baselines and validate

1. Run each baseline candidate via the runner:
   ```
   uv run python runners/{slug}_runner.py run-candidate \
     --experiment experiments/{slug} --candidate {id}
   ```
2. Validate the experiment:
   ```
   uv run python experiment.py validate experiments/{slug}
   ```
3. Verify `results.json` has baseline entries

If anything fails, fix it and re-run. Do not leave a broken experiment.

## Output

After everything succeeds, print a summary:

```
## Experiment Ready: {slug}

**Files created:**
- lib/{slug}/data.py
- lib/{slug}/modeling.py
- runners/{slug}_runner.py
- experiments/{slug}/experiment.md
- experiments/{slug}/research_sources.md (seeded with {N} source cards)
- experiments/{slug}/results.json (baseline results)

**Research hypotheses:** {count}
**Baseline results:**
- {candidate_id}: {objective_metric} = {score}

**To start the loop:**
  uv run python -m loop start experiments/{slug}
```
