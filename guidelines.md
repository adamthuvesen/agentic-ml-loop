# Guidelines

Hard invariants for all experiments. Violating these is a serious error.

For research methodology and ML craft principles, see `program.md`.
This file covers enforceable guardrails, not philosophy.

---

## Evaluation Integrity

1. **Start every cycle with a written objective, success criterion, expected direction, and falsifier.**
   If the result cannot prove the hypothesis wrong, the cycle is too vague.
   Record this in `research_journal.md` before running code or launching a
   search.

2. **Every material claim must point to evidence.**
   Link journal conclusions to a command, script, output file, source, or
   `results.json` entry. External research that changes the plan belongs in
   `research_sources.md` with a reusable takeaway, not as invisible prompt
   residue.

3. **Recover deliberately after failures.**
   After a failed tool call, script, or model run, record the failure mode before
   retrying. After two failures with the same cause, simplify the approach or
   change the hypothesis instead of stacking patches.

4. **`objective_score` = validation metric, same metric and split for all candidates.**
   Not test, not holdout, not tuning-CV. Report test metrics separately — they
   must never flow into `objective_score`.

5. **Keep train/val/test roles separate and explicit.**
   Train fits parameters and train-only transforms. Validation selects features,
   hyperparameters, model families, thresholds, calibration, and stopping
   criteria. Test/holdout is touched once, after decisions are locked, for final
   unbiased evaluation.

6. **Never use test/holdout data to choose features, hyperparameters, thresholds, calibration, or stopping criteria.**
   Test feedback is not research feedback. If a test result changes the plan,
   promote a new holdout before continuing.

7. **Split first; fit every learned preprocessing step on train only, then apply unchanged to val and test.**
   Scalers, encoders, imputers — all fitted on train. Fitting per-split makes
   metrics incomparable.

8. **Supervised transforms — target encoding, feature selection, calibration — must use train-only or out-of-fold data.**
   `df.groupby('col')['target'].mean()` on the full dataset is target leakage.
   Calibrators fitted on the same data they predict on are overfit. Use
   `Pipeline` or compute per-fold.

9. **Respect data structure in evaluation.**
   Use temporal splits for time-ordered data, group-aware splits for repeated
   entities. Random splitting time-series or grouped data produces optimistic,
   unreliable estimates.

10. **CV for tuning stays inside training data; never report tuning-CV as final performance.**
   If you use cross-validation to select hyperparameters, those CV scores are
   selection-biased. The validation split is the arbiter.

11. **Audit feature availability at prediction time.**
    For every new feature family, ask: would this value exist, with the same
    definition and freshness, at the prediction timestamp? If not, it is leakage
    even if it lives in the raw dataset.

12. **Run a sanity probe when performance looks too good.**
    Use at least one of: shuffled-target baseline, ID/memorization check,
    duplicate-row audit, or train/validation adversarial validation. Suspicious
    jumps need proof before they become research conclusions.

## Statistical Rigor

13. **Before claiming a winner, report uncertainty.**
   Use paired bootstrap (>= 1000 resamples), repeated CV, or permutation
   testing. Report the confidence interval alongside the point estimate. On
   small val sets (< 5K rows), a 0.01 AUC difference is noise.

14. **Track validation-set erosion from repeated selection.**
   After every 5 candidates on the same val split, report the bootstrap 95% CI
   of the best candidate's score. If it overlaps with the runner-up, the ranking
   is not trustworthy. After 20+ candidates, treat the leaderboard with
   increasing skepticism.

15. **Flag train-val gaps exceeding 0.05 AUC or 10% relative.**
   Every candidate must report both train and validation metrics. A large gap
   signals overfitting — investigate before accepting a candidate as "best."

16. **Lock hyperparameter search budgets up front; tune on train-only data; persist the study.**
    Declare the trial count (e.g. `n_trials=50`) in the research journal before
    starting the search, fit on CV or a tuning slice carved from train, and
    save the study artefact to `work/` (`optuna_study.db` or a trials CSV).
    Extending the budget mid-search after peeking at scores reintroduces
    selection bias on the tuning-CV scores. The tuned config's *validation*
    score — not its best tuning-CV score — is what enters `objective_score`.

## Reporting

17. **Report both discrimination and task-appropriate operating metrics.**
    Every candidate needs a ranking metric (AUC) and a metric that reflects the
    deployment decision (precision@k, F-beta, Brier, etc.). Which operating
    metric depends on the task — `experiment.md` should specify it.

18. **Keep the metric contract explicit.**
    `experiment.md` should name the target, primary validation metric,
    operating metric, directionality, and cost asymmetry. If any of these
    change, update the experiment spec before comparing new candidates to old
    ones.

19. **Report important slices, not just aggregate scores.**
    When the domain has meaningful cohorts, time periods, classes, or data
    regimes, report slice metrics for top candidates. A model that wins only in
    aggregate may be the wrong model.

## Data Handling

20. **Class rebalancing on train only.**
    Prefer `class_weight='balanced'` or sample weights over resampling. Never
    resample across split boundaries.

21. **Document dataset provenance before modeling decisions depend on it.**
    Record source files, row counts, target definition, split policy, known
    exclusions, and obvious collection bias in the journal or an output data
    note. Silent data assumptions become silent model failures.

## Candidate Design

22. **One variable per candidate; lock the pipeline within a comparison group.**
    Each candidate differs on exactly one axis: features, model family, or
    hyperparameters. The preprocessing pipeline must be identical across compared
    candidates. If you change the pipeline, re-run the baseline and start a new
    comparison group.

23. **Tune decision thresholds on validation only, after model selection.**
    Never bake threshold-dependent metrics into `objective_score`. Threshold
    tuning is a separate step that happens after the best model is identified.

## Reproducibility

24. **Seed every stochastic component; record seed, features, hyperparameters, and train metric in candidate metadata.**
    Results that can't be reproduced can't be trusted. Missing metadata makes
    debugging impossible.

25. **Record enough run context to replay the result.**
    Candidate metadata should include the command, code version or dirty-state
    note, data fingerprint when available, split seed or split file, dependency
    environment, and output paths. A leaderboard without provenance is a rumor.

26. **`results.json` must remain valid and atomic.**
    Either write a complete candidate entry or fail cleanly. Don't catch
    exceptions just to write partial entries — they break artifact trust.
