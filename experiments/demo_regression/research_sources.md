# Research Sources: demo_regression

Durable, source-backed research memory for this experiment.

Use this file for reusable outside research: Kaggle write-ups, papers, blog posts,
docs, and similar-problem references that future cycles should build on.

Keep cycle-by-cycle reasoning, hypotheses, and results in `research_journal.md`.
Treat this as a living document: source cards preserve provenance, while
`Reusable Takeaways` should be rewritten when later findings nrevenueow, qualify, or
contradict earlier conclusions.
Phrase takeaways as scoped heuristics with caveats, not universal laws.

## Reusable Takeaways

- **Two-stage hurdle models are the practical standard for zero-inflated regression** — separate binary classification (zero vs. positive) from amount regression on positives only. Each stage can use different model families. Expect better R² than single-model approaches when zero fraction ≥ 20%. **Caveat (empirical, demo_regression Cycle 2):** The hurdle fails to improve when the zero/positive boundary is determined by an unmeasured factor. If the classifier uses the same features as the base model and those features are scale-dominated, the classifier learns the same biased rule and the hurdle gains nothing. Feature quality is the binding constraint, not the model architecture.
- **Tweedie loss does not fix temporal overfitting** — XGB-Tweedie (p=1.5) worsened the train-val gap vs. XGB-squarederror (0.461 vs. 0.438) on demo_regression. Tweedie may increase sensitivity to small-value patterns in the training period, compounding temporal generalization failures. Try Tweedie only after confirming the generalization gap isn't structural.
- **Scale confounders (revenue, seat count) inflate raw correlations** — test whether the model is capturing true expansion propensity or just account size by comparing raw-target vs. revenue-normalized target R². If normalization drops R² significantly (demo_regression: 0.435 → −0.140), the raw signal is mostly scale and behavioral features carry negligible independent signal.
- **Log(1+y) transform is CONTRAINDICATED when (a) the target is zero-inflated AND (b) a few extreme outliers dominate SS_tot** — tested on demo_regression: ridge-log dropped from 0.435 to 0.126. Root cause: top 1% of values contributed 42.8% of SS_tot; log compression during training causes systematic under-prediction of those cases; back-transform errors destroy R². Tweedie loss or two-stage hurdle are the correct treatments for this regime.
- **Bootstrap CI width reveals R² reliability** — when the top 10% of accounts own >80% of SS_tot, bootstrap R² has 95% CIs spanning ±0.13. Any claimed improvement < ~0.05 is within noise. Always compute bootstrap CIs before declaring a winner on concentrated-outcome targets.
- **When three model families cluster at the same R², the feature space is the ceiling** — additional model variants won't escape the plateau. The signal is in the features, not the architecture. The correct next move is feature engineering or problem reframing.

## Source Cards

### Source 001: Two-Stage Hurdle Models: Predicting Zero-Inflated Outcomes

- **Type:** blog
- **URL:** https://towardsdatascience.com/two-stage-hurdle-models-predicting-zero-inflated-outcomes/
- **Why relevant here:** Target has 25.3% exact zeros; single-model regression must simultaneously fit zero and positive regimes
- **Key takeaways:** Two-stage approach separates the problem into binary classification (zero vs. non-zero) followed by regression on the positive subset. Each stage can use different model families and hyperparameters. Outperforms single-model approaches on zero-inflated data.
- **Applicability / caveats:** Directly applicable. Adds implementation complexity (two models, combined prediction pipeline). Validation requires accounting for both stages in the final R² computation.
- **Ideas this suggests:** H3 — implement a logistic classifier + ridge/XGB regressor as a two-stage candidate
- **Status:** used

### Source 002: Dealing with Zero-Inflated Data: Achieving State-of-the-Art with a Two-Fold ML Approach

- **Type:** paper
- **URL:** https://www.sciencedirect.com/science/article/pii/S0952197625003392
- **Why relevant here:** Academic validation of two-stage decomposition on zero-inflated, right-skewed targets similar to this one
- **Key takeaways:** Two-fold decomposition (event classifier + amount regressor) achieves state-of-the-art on zero-inflated problems. Gradient boosting for event stage; specialized regressors for the positive tail. Right-skew requires attention in the amount model.
- **Applicability / caveats:** Validates two-stage approach rigorously. Recommends treating the two stages as independent optimization problems. May overfit on smaller datasets if the positive subset is limited.
- **Ideas this suggests:** Use LightGBM or XGBoost for the event-stage classifier; test ridge vs. XGB for the amount stage
- **Status:** used

### Source 005: Log Transform Failure on Zero-Inflated Right-Skewed Targets (empirical, this experiment)

- **Type:** empirical
- **URL:** internal — cycle 0001 demo_regression
- **Why relevant here:** Directly observed log(1+y) failure on this target
- **Key takeaways:** When top-1% values contribute >40% of SS_tot and target has >20% zeros, log transform on raw target destroys R². The model learns a compressed-scale relationship, under-predicts outliers, and back-transform errors dominate the R² metric.
- **Applicability / caveats:** Specific to R² as objective metric. Log transform may still improve calibration or MAPE. The interplay between zero-inflation and right-tail outlier dominance is the key trigger.
- **Ideas this suggests:** Use Tweedie or two-stage hurdle instead. If log transform is desired, apply only to the positive subset within a two-stage framework.
- **Status:** used

### Source 003: Tweedie Loss Function for Right-Skewed Data

- **Type:** blog
- **URL:** https://medium.com/data-science/tweedie-loss-function-for-right-skewed-data-2c5ca470678f
- **Why relevant here:** Alternative to two-stage that handles zero-inflation + right skew in a single model via XGBoost/LightGBM objective
- **Key takeaways:** Tweedie regression models non-negative, zero-inflated, right-skewed distributions by tuning a power parameter (p). Works natively in gradient boosting. Variance power can be cross-validated; p between 1 and 2 covers the compound Poisson-Gamma family common in revenue data.
- **Applicability / caveats:** Simpler than two-stage; less interpretable. Requires tuning p. May underperform if the zero-inflation mechanism is truly separable from the amount mechanism.
- **Ideas this suggests:** Try `objective="reg:tweedie"` in XGBoost with p cross-validated over [1.1, 1.5, 1.9]
- **Status:** deferred

### Source 006: Scale Confound Confirmed — H2 Scale Normalization Failure (empirical, this experiment)

- **Type:** empirical
- **URL:** internal — cycle 0002 demo_regression
- **Why relevant here:** H2 test: predict expansion_rate = target / contract_value_t0, back-multiply. Ridge-scale-norm val R²=−0.140 (vs ridge-basic 0.435). Behavioral features (usage_growth r=0.089, champion_engagement r=0.010 on |residual|) carry negligible signal independent of scale.
- **Key takeaways:** When behavioral feature correlations with residuals are in the 0.01–0.09 range and |residual| correlates r=0.80 with the scale feature, behavioral features are noise after controlling for scale. Rate normalization amplifies errors catastrophically for large-revenue accounts. The practical signal is "big accounts expand more."
- **Applicability / caveats:** Specific to this feature set and temporal split. Behavioral features might become predictive with richer signals (trajectories, change signals, historical patterns).
- **Ideas this suggests:** Feature engineering: YoY usage growth, historical expansion patterns, account health trajectories. Or reframe as ranking (who is most likely to expand) rather than absolute amount.
- **Status:** used

### Source 007: Hurdle Model Bootstrap CI — Feature Ceiling (empirical, this experiment)

- **Type:** empirical
- **URL:** internal — cycle 0002 demo_regression
- **Why relevant here:** Hurdle-logistic-ridge vs ridge-basic: paired bootstrap CI [-0.006, +0.010], P(hurdle > ridge)=71.1%. Not significant.
- **Key takeaways:** (1) When the classifier in a hurdle model uses scale-dominated features, it learns the same biased rule as the base model — the architecture improvement is neutralized by feature quality. (2) Bootstrap CI width ±0.13 on R² for top-10%-concentrated targets means differences <0.05 are noise. (3) Three linear variants clustering at 0.435–0.437 is strong evidence of a feature space ceiling.
- **Applicability / caveats:** Applies broadly to any zero-inflated regression where zeros/positives are separated by unmeasured factors. The hurdle is not a magic fix — it needs a classifier with genuine discriminating signal.
- **Status:** used

### Source 004: How to Remove or Control Confounds in Predictive Models

- **Type:** paper
- **URL:** https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giac014/6547681
- **Why relevant here:** contract_value_t0 (r=0.70) and seat_count (r=0.60) are scale confounders — models may be fitting account size rather than expansion propensity
- **Key takeaways:** Partial regression / residualization against confounders before fitting; or use confounders as explicit controls; or model the rate (expansion/revenue) instead of absolute amount. Naive feature selection that captures confounding rather than causal signal inflates apparent performance.
- **Applicability / caveats:** Directly relevant to H2. Residualization may remove too much signal if the scale variables carry genuine predictive value. Testing both approaches (raw and normalized) is the cleanest approach here.
- **Ideas this suggests:** H2 — compute expansion_rate = net_revenue_change_180d / contract_value_t0, train a model, back-multiply, compare R² to raw-target model
- **Status:** used
