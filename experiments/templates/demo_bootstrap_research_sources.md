# Research Sources: demo_bootstrap

## Reusable Takeaways

- For **binary classification with class imbalance**, ROC-AUC ranks models without
  fixing a threshold; pair with calibration or cost-sensitive metrics if decisions
  use a cutoff.
- **Temporal splits** are appropriate when the deployment distribution drifts over
  time; compare validation vs test gap to detect instability.
- Start with a **constant or linear baseline** before tree ensembles to anchor
  expected lift.

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
