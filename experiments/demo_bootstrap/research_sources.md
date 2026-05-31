# Research Sources: demo_bootstrap

## Reusable Takeaways

- For **binary classification with class imbalance**, ROC-AUC ranks models without
  fixing a threshold; pair with calibration or cost-sensitive metrics if decisions
  use a cutoff.
- **Temporal splits** are appropriate when the deployment distribution drifts over
  time; compare validation vs test gap to detect instability.
- Start with a **constant or linear baseline** before tree ensembles to anchor
  expected lift.
- **Polynomial features (degree=2)** on 3-4 input features are safe from overfitting
  on ~500 train rows (~10-15 terms). Pair with regularization (elastic net or L1) to
  prune irrelevant interactions. Try `interaction_only=True` first.
- **LightGBM on tiny data** (<500 rows) needs aggressive regularization: `num_leaves`
  4-8, `min_child_samples` 20-50, `n_estimators` 50-200 with early stopping. Default
  params will memorize training data.
- **Feature redundancy hurts**: when two features have r>0.8, dropping the weaker
  predictor often helps regularized models and reduces split dilution in trees.
- **Nonlinear feature effects** (e.g., conversion rate jumps above a session threshold)
  are invisible to linear models without explicit encoding — polynomial features or
  tree models can capture these.
- **Regularization sweep as signal diagnostic**: if val AUC barely changes across 6
  orders of C magnitude, the model has already extracted the available linear signal.
  The bottleneck is the feature space, not the model.
- **Cohen's d → AUC mapping**: a feature with r≈0.28 to the target corresponds to
  d≈0.58, giving a single-feature AUC of ~0.60. If the strongest features are
  collinear, combining them adds little beyond this ceiling.
- **L1 as feature selection diagnostic**: ElasticNet with l1_ratio=0.5 on small
  datasets reliably zeroes out noise features. When L1 prunes a feature to exactly 0,
  it's strong evidence the feature adds no signal after accounting for other predictors.
  Useful for confirming EDA-based feature selection decisions.

- **Model family diversity can't compensate for feature poverty.** When logistic
  regression, Naive Bayes, SVM-linear, and SVM-RBF all converge to the same AUC
  within noise, the ceiling is information-theoretic (limited feature-target mutual
  information), not algorithmic. Ensembling won't help either — the models make the
  same structural errors because they share the same inputs.
- **Target encoding only helps with diverse subgroup rates.** When the largest
  categorical has segments with nearly identical target rates (e.g., 0.41 vs 0.42),
  continuous target encoding collapses to an approximately binary signal and adds
  nothing over one-hot or binary indicators.
- **Error analysis reveals the signal boundary.** When false negatives cluster in
  low-feature-value subgroups and false positives cluster in high-feature-value
  subgroups, the model is correctly using the available signal — errors mark where
  the features can't discriminate. This pattern means the bottleneck is features,
  not model capacity.
- **Temporal drift causes systematic calibration bias.** When the base rate shifts
  between train and val/test, model probabilities are anchored to the training
  distribution. This produces asymmetric errors (mostly FN if val rate > train rate)
  and poor Brier scores. Threshold tuning helps classification but not AUC.

## Source Cards

### Source 001: sklearn — ROC-AUC

- **Type:** docs
- **URL:** https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- **Why relevant here:** Defines our primary ranking metric for binary classification.
- **Key takeaways:** AUC equals probability a random positive is ranked above a random negative.
- **Applicability / caveats:** Insensitive to class imbalance in ranking; does not encode business costs.
- **Ideas this suggests:** Use `roc_auc_score` on validation for `objective_score`.
- **Status:** used

### Source 002: Probst, Bischl, Boulesteix — tunability vs performance (2019)

- **Type:** paper
- **URL:** https://arxiv.org/abs/1802.09596
- **Why relevant here:** Motivates simple baselines before heavy tuning.
- **Key takeaways:** Many datasets are modeled well by strong defaults on random forests / boosting.
- **Applicability / caveats:** Our demo uses tiny data and linear models; still useful to avoid over-tuning.
- **Ideas this suggests:** Compare logreg to majority before adding complexity.
- **Status:** deferred

### Source 003: Small-data binary classification techniques (synthesis)

- **Type:** synthesis (Kaggle discussions, ML blogs, sklearn docs)
- **Why relevant here:** Our train set is 480 rows with 3 numeric + 1 categorical feature.
  Need techniques that extract more signal without overfitting.
- **Key takeaways:**
  - Polynomial degree-2 interactions are safe with <5 input features (~10-15 expanded terms)
  - LightGBM can beat logistic regression even below 500 rows IF regularized aggressively
  - ElasticNet (L1+L2) with CV auto-selects regularization strength and prunes weak features
  - Target encoding with K-fold smoothing extracts more from categoricals than one-hot
- **Applicability / caveats:** With 160-row val set, AUC differences <0.02 are noise.
  Must bootstrap CI before declaring a winner.
- **Ideas this suggests:** Test polynomial interactions on informative features;
  test conservative LightGBM as nonlinear comparison.
- **Status:** used
