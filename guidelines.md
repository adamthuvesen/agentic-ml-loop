# Guidelines

Hard invariants for all experiments. Violating these is a serious error.

For research methodology and ML craft principles, see `program.md`.
This file covers enforceable guardrails, not philosophy.

---

## Evaluation Integrity

1. **`objective_score` = validation metric, same metric and split for all candidates.**
   Not test, not holdout, not tuning-CV. Report test metrics separately — they
   must never flow into `objective_score`.

2. **Never use test/holdout data to choose features, hyperparameters, thresholds, calibration, or stopping criteria.**
   The test set exists for one purpose: final unbiased evaluation after all
   decisions are locked.

3. **Split first; fit every learned preprocessing step on train only, then apply unchanged to val and test.**
   Scalers, encoders, imputers — all fitted on train. Fitting per-split makes
   metrics incomparable.

4. **Supervised transforms — target encoding, feature selection, calibration — must use train-only or out-of-fold data.**
   `df.groupby('col')['target'].mean()` on the full dataset is target leakage.
   Calibrators fitted on the same data they predict on are overfit. Use
   `Pipeline` or compute per-fold.

5. **Respect data structure in evaluation.**
   Use temporal splits for time-ordered data, group-aware splits for repeated
   entities. Random splitting time-series or grouped data produces optimistic,
   unreliable estimates.

6. **CV for tuning stays inside training data; never report tuning-CV as final performance.**
   If you use cross-validation to select hyperparameters, those CV scores are
   selection-biased. The validation split is the arbiter.

## Statistical Rigor

7. **Before claiming a winner, report uncertainty.**
   Use paired bootstrap (>= 1000 resamples), repeated CV, or permutation
   testing. Report the confidence interval alongside the point estimate. On
   small val sets (< 5K rows), a 0.01 AUC difference is noise.

8. **Track validation-set erosion from repeated selection.**
   After every 5 candidates on the same val split, report the bootstrap 95% CI
   of the best candidate's score. If it overlaps with the runner-up, the ranking
   is not trustworthy. After 20+ candidates, treat the leaderboard with
   increasing skepticism.

9. **Flag train-val gaps exceeding 0.05 AUC or 10% relative.**
   Every candidate must report both train and validation metrics. A large gap
   signals overfitting — investigate before accepting a candidate as "best."

10. **Lock hyperparameter search budgets up front; tune on train-only data; persist the study.**
    Declare the trial count (e.g. `n_trials=50`) in the research journal before
    starting the search, fit on CV or a tuning slice carved from train, and
    save the study artefact to `work/` (`optuna_study.db` or a trials CSV).
    Extending the budget mid-search after peeking at scores reintroduces
    selection bias on the tuning-CV scores. The tuned config's *validation*
    score — not its best tuning-CV score — is what enters `objective_score`.

## Reporting

11. **Report both discrimination and task-appropriate operating metrics.**
    Every candidate needs a ranking metric (AUC) and a metric that reflects the
    deployment decision (precision@k, F-beta, Brier, etc.). Which operating
    metric depends on the task — `experiment.md` should specify it.

## Data Handling

12. **Class rebalancing on train only.**
    Prefer `class_weight='balanced'` or sample weights over resampling. Never
    resample across split boundaries.

## Candidate Design

13. **One variable per candidate; lock the pipeline within a comparison group.**
    Each candidate differs on exactly one axis: features, model family, or
    hyperparameters. The preprocessing pipeline must be identical across compared
    candidates. If you change the pipeline, re-run the baseline and start a new
    comparison group.

14. **Tune decision thresholds on validation only, after model selection.**
    Never bake threshold-dependent metrics into `objective_score`. Threshold
    tuning is a separate step that happens after the best model is identified.

## Reproducibility

15. **Seed every stochastic component; record seed, features, hyperparameters, and train metric in candidate metadata.**
    Results that can't be reproduced can't be trusted. Missing metadata makes
    debugging impossible.

16. **`results.json` must remain valid and atomic.**
    Either write a complete candidate entry or fail cleanly. Don't catch
    exceptions just to write partial entries — they break artifact trust.
