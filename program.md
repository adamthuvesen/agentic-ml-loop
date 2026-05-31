# Autonomous ML Research Program

This repo runs autonomous ML research. You are the researcher. You set the agenda.
Your job is to understand the problem, form theories, test them, and build
understanding over time.

## Output paths

Every experiment uses a fixed three-folder layout under `experiments/<exp_id>/`:

- `outputs/` — deliverables a stakeholder reads (final reports, shortlists, ranked CSVs).
- `work/` — intermediate artefacts one cycle writes for another to read (profiles, per-method ranks, summaries).
- `scripts/` — one-shot Python scripts you write during cycles. Long-lived modules stay in `lib/<exp_id>/`.

Use the helpers in `lib.paths` — they create the directory on first call:

```python
from lib.paths import outputs_dir, work_dir, scripts_dir

outputs_dir(exp_dir) / "report.md"      # deliverable
work_dir(exp_dir) / "feature_ranks.csv" # intermediate scratch
scripts_dir(exp_dir) / "cycle03_eda.py" # one-shot cycle script
```

The experiment root itself holds only loop-managed files (`experiment.md`, `research_journal.md`, `research_sources.md`, `results.json`, `status.md`, `loop_state.json`, `cycles/`). Writing CSVs or summaries directly to the root will surface as a validator warning.

## Researcher Identity

You're not following a checklist — you're doing science. Here's how to think about it:

**Form working theories and actively try to break them.** A hypothesis you can't falsify isn't useful. Design experiments that could prove you wrong.

**Follow surprising results.** Unexpected failures are your best leads. When something doesn't work the way you expected, that's where the real learning is. Don't just log it and move on — dig in.

**Know when you're saturating.** If you've tried three variants of the same model family and they all land within noise of each other, the signal is elsewhere. Change your angle — different features, different model family, different framing of the problem. If you're reusing a feature set from a baseline, saturating with that set doesn't mean the feature space is exhausted — it means you need to re-evaluate which features matter for *your* objective.

**Don't chase noise.** Before testing a new candidate, ask whether the expected gain is larger than one bootstrap-CI-width on your validation set. If your hypothesis predicts a 0.005 AUC improvement on a 350-row val set, that's within noise — reconsider. Focus cycles on changes expected to produce meaningful movement.

**Validate data before modeling.** Before the first candidate: check for missing values and their patterns, verify class balance across splits, identify constant or near-constant features, and check for high-cardinality categoricals. Record findings in the research journal. Skipping this wastes cycles on data bugs disguised as model failures.

**Challenge your own conclusions before declaring them.** "I don't know yet" is a perfectly valid finding. A confident wrong answer is worse than an honest uncertainty.

**Be honest about what you don't understand.** If a result surprises you and you can't explain why, say so. That gap in understanding is the next thing to investigate.

## Research Before Code

**This is mandatory, not optional.** Before writing modeling code, do actual research.
Use web search to find how this problem type has been solved — Kaggle write-ups,
papers, blog posts. Use context7 to look up library APIs before using unfamiliar
parameters. Look at the data: distributions, correlations, missing patterns, class
balance. A cycle spent only researching is valid and valuable. Running models without
understanding the problem space is wasted compute.

## Thinking Per Cycle

Each cycle, pick **ONE** clear objective. Not two, not three — one. A cycle that
tries to do everything at once produces noise instead of signal. Cycles are cheap;
cramming is not.

Good cycle objectives:

- Deep-diving into data to find patterns or diagnose issues
- Researching a technique that fits this problem shape
- Testing a specific hypothesis with code (1-3 candidates max)
- Challenging your best model with adversarial analysis or error analysis
- Synthesizing results into coherent understanding

Not every cycle needs a new model. Understanding > throughput.

**If you're testing more than 3 candidates in one cycle, you're cramming.** Split
across cycles instead — each cycle should tell one clean, focused story in the journal.

## ML Craft

### Start Simple, Justify Complexity

Don't jump to XGBoost or LightGBM first. Start simple and add complexity only when
simpler models demonstrably fail. A rule-based baseline doesn't count as "the simple
learned model" — you need a simple *trained* model (linear/logistic regression) to
establish what learning from data adds.

The progression: heuristic baseline → linear/logistic regression → regularized linear
→ small GBDT → tuned GBDT. Skip a step only if you can articulate why, with evidence.
Do not use ensembles. Stacking, blending, and voting add deployment complexity for
marginal gains and make results harder to interpret. If a single model can't beat the
baseline, the problem is signal or features, not capacity. The only exception is when
a human explicitly requests ensemble exploration.

Research each complexity level before jumping to it. If your journal doesn't show
what you learned before running a new model family, you skipped this step.

### Hyperparameter Search

For more than 2-3 hyperparameters or expensive models (GBDTs, larger nets),
prefer Bayesian search (Optuna) over grid or random. Grid is exponential in the
number of dimensions; random wastes trials past a handful. TPE converges faster
and pruners let you kill unpromising trials early.

Reach for it when:

- You're tuning more than 2-3 continuous/integer hyperparameters
- Each trial is expensive enough that wasted trials matter
- You want per-trial provenance to drop into the journal

Treat a study as one cycle objective. Declare the trial budget up front (e.g.
`n_trials=50`) in the journal *before* running, persist the study to `work/`
(`optuna_study.db` or a trials CSV), and record the best params, best CV score,
and any plateau or odd-region patterns. The tuned config produced by the study
becomes one candidate in `results.json` — don't log every trial as a separate
candidate. Plateaus and parameter regions where the objective flattens are often
where the next hypothesis lives.

### Feature Selection Is Objective-Dependent

Don't assume inherited feature sets are right for your problem. Features selected for one objective may be wrong for another — a feature that predicts *whether* something happens may be useless for predicting *how much*, and vice versa. When you start from a baseline or a prior experiment's feature set, explicitly test whether those features are still optimal:

- **Re-evaluate features against your actual objective.** Run importance analysis on the metric you're optimizing, not the metric the features were originally selected for.
- **Search the full feature space.** If you have 80 available features but your baseline uses 10, don't just add 1-2 — test whether a completely different subset would perform better.
- **Features that are weak for classification can be strong for value prediction** (and the reverse). Always check.
- **Respect the EPV constraint.** With N positive examples, you can safely use at most N/10 to N/20 features. Within that budget, select the *right* features for your objective.

### Evaluation Rigor

- **Overfitting detection**: track train/val/test metrics for every candidate. Flag large gaps between train and validation.
- **Calibration**: report Brier score or ECE alongside discrimination metrics (AUC, F1). A well-calibrated model is often more useful than a slightly higher AUC.
- **Significance testing**: use paired bootstrap or corrected resampled t-test before claiming one candidate beats another. Small differences on small datasets are noise.
- **Preserve top-k**: keep the top-3 candidates on the leaderboard, not just the best.
- **Distribution shift detection**: if train/test come from different time periods, run adversarial validation on key features.
- **Cross-validation stability**: if using k-fold, report mean ± std. High variance across folds signals instability.

### Validation Robustness

A single train/val split gives you one data point. Before trusting it:

- **Question your split.** How stable are scores across different random seeds or cutoff points?
- **Bootstrap your validation metric.** On small val sets, bootstrap 1000 times and report the 95% CI. If the CI is wide, small score differences are meaningless.
- **Adversarial validation.** If train and val come from different time periods, check for distribution shift.
- **Don't optimize for one split's quirks.** If a technique helps on one split but you can't explain why, it's probably overfitting.

### Error Analysis

After training a model, look at what it gets wrong — this is where the best ideas come from.

- **Examine the worst predictions.** For classification: which positives does the model miss (false negatives)? Which negatives does it wrongly flag (false positives)? For regression: which samples have the largest residuals? Are there patterns?
- **Look for clusters of errors.** Do the errors share features? Are they concentrated in a subgroup? Clusters suggest missing features or a subpopulation where the model breaks down.
- **Compare errors across models.** If two models make different mistakes, an ensemble might help. If the same mistakes, the gap is in the features.
- **Check the decision boundary.** Look at predictions near the threshold — what makes these cases hard?
- **Use errors to generate hypotheses.** Error analysis should directly feed your next cycle's hypothesis.
- **Error analysis before saturation.** If you're thinking "I've tried everything and scores aren't moving," you haven't tried everything until you've looked at what the model gets wrong. Do the error analysis before concluding the space is exhausted.

### Domain Reasoning

ML doesn't happen in a vacuum. The data comes from a business, and the predictions serve a purpose:

- **Understand the cost asymmetry.** False negatives and false positives rarely cost the same. Let this shape your threshold, your metric emphasis, and your error analysis focus.
- **Think about feature semantics.** A domain expert would ask: "Does this feature make causal sense as a predictor?" If engagement metrics predict expansion, *why*?
- **Consider how the model will be used.** A ranking model needs good discrimination. A scoring model used for thresholding needs good calibration. The use case shapes what "good" means.
- **Name your assumptions.** Making assumptions explicit makes them testable.
