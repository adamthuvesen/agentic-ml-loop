# Research Journal: demo_classification

Synthetic binary classification. Objective metric: `val_auc`. Splits are
deterministic; numbers below come from the bundled candidate runners and are
reproducible with `runners/demo_classification_runner.py run-candidate`.

## Cycle 0001: Baseline and linear sweep

**Hypothesis:** A simple trained linear model should clear the rule-based
baseline; if it does, the signal is real and mostly linear.
**Expected direction:** logistic regression > rule baseline on `val_auc`.
**Falsifier:** logistic regression lands within noise of the rule baseline.

**What I ran:** the rule baseline, `logreg-basic`, and `logreg-engineered`
(adds hand-crafted interaction/ratio features).

**Result:**

| candidate | val_auc | train_auc |
| --- | --- | --- |
| `logreg-basic` | 0.730 | 0.736 |
| `logreg-engineered` | 0.729 | 0.738 |
| `rule-baseline` | 0.658 | 0.655 |

**Finding:** The learned linear model beats the rule baseline by ~0.07 AUC — the
signal is real. The engineered feature set is a dead heat with the basic one
(0.729 vs 0.730), so those hand-crafted features add nothing for this objective.
Train and validation AUC are within ~0.01 of each other: the linear model is not
overfitting and is not capacity-starved on its own features.

**Implication:** Linear capacity already captures most of the signal. The next
question is whether a higher-capacity model finds non-linear structure the linear
model misses — or whether it just overfits.

## Cycle 0002: Does added capacity help, or just overfit?

**Hypothesis:** Gradient boosting will find non-linear structure and beat the
linear leader on `val_auc`.
**Expected direction:** best XGBoost variant > `logreg-basic` (0.730).
**Falsifier:** XGBoost variants do not beat the linear leader on validation,
especially if train AUC pulls far ahead of validation.

**What I ran:** `xgb-base`, `xgb-basic`, and `xgb-tuned-light`.

**Result:**

| candidate | val_auc | train_auc | train−val gap |
| --- | --- | --- | --- |
| `xgb-basic` | 0.709 | 0.901 | 0.192 |
| `xgb-tuned-light` | 0.706 | 0.842 | 0.136 |
| `xgb-base` | 0.705 | 0.839 | 0.134 |

**Finding:** Hypothesis falsified. Every XGBoost variant lands *below* the linear
leader on validation (0.705–0.709 vs 0.730) while its training AUC runs to
0.84–0.90 — a 0.13–0.19 train/validation gap. The extra capacity is spent
memorizing training noise, not finding generalizable non-linear structure. Light
tuning narrows the gap (0.19 → 0.14) but does not recover the linear model's
validation performance.

**Implication:** This is a signal/feature problem, not a capacity problem. The
leaderboard leader is `logreg-basic` at `val_auc` 0.730. Adding model complexity
is the wrong lever here; the next move would be feature work or a fresh look at
what the model gets wrong — not a bigger model.

## Status

Leaderboard leader: `logreg-basic` (`val_auc` 0.730). Conclusion: the signal is
real and predominantly linear; gradient boosting overfits without a validation
gain on this synthetic dataset.
