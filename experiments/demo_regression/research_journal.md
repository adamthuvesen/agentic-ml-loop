# Research Journal: demo_regression

Write one entry per cycle. Include what you set out to learn, what you found,
and what it means for next steps.

## Cycle 0001: ML Baselines — Ridge, XGBoost, and Log Transform

### Objective

Establish ML baselines (ridge-basic, ridge-engineered, xgb-basic) to replace the rule-based heuristic (val R²=0.026). Simultaneously test log(1+y) target transform (H1 — high confidence) given skew=4.28 and extreme right tail (median=117, max=13,339). This cycle answers the foundational question: how much of the signal is just account scale?

### Hypothesis (written before running code)

The rule-baseline at val R²=0.026 is essentially noise — the heuristic deliberately excludes scale features. A ridge regression over base features (which includes contract_value_t0 r=0.70 and seat_count r=0.60) should jump substantially, likely to R²=0.45–0.60. XGBoost should match or exceed ridge given the likely nonlinear interactions between scale and behavioral signals.

Log(1+y) transform is expected to help because MSE loss is dominated by the extreme right tail when fitting on raw targets. By compressing the tail, the model can learn the bulk-distribution shape better. Back-transform predictions before evaluating on original scale.

**Key question:** Is the jump in R² from rule → ridge explained entirely by the inclusion of scale features (contract_value_t0, seat_count)? If so, the next move is H2 (revenue-normalization test) to see whether any propensity signal exists independent of size.

### Results

| Candidate | Train R² | Val R² | Val RMSE | Val MAE | Train-Val Gap |
|---|---|---|---|---|---|
| rule-baseline | 0.044 | 0.026 | 1018 | 623 | 0.018 |
| ridge-basic | 0.515 | 0.435 | 776 | 452 | **0.080** |
| ridge-engineered | 0.517 | 0.436 | 775 | 450 | 0.081 |
| xgb-basic | **0.793** | 0.355 | 829 | 461 | **0.438** |
| ridge-log | 0.067 | 0.126 | 965 | 463 | — |
| xgb-log | 0.402 | 0.117 | 970 | 465 | — |

### Findings

**Finding 1 — Scale features dominate.** Ridge-basic jumps from rule-baseline's 0.026 to 0.435 by including `contract_value_t0` (r=0.70) and `seat_count` (r=0.60). The 0.41 R² gain is almost entirely attributable to scale. The heuristic's behavior signals (growth, engagement, adoption) carry limited independent signal.

**Finding 2 — XGBoost is massively overfitting.** Train R²=0.793 vs Val R²=0.355 = gap of 0.44, way above the 0.05 alarm threshold. Despite heavy regularization (min_child_weight=12, reg_lambda=4.0), XGB is memorizing the training set. Ridge beats XGB on this dataset — likely because the underlying relationships are substantially linear (scale proxies are linear predictors of expansion amount) and there isn't enough signal volume for trees to generalize.

**Finding 3 — Engineered features add almost nothing.** Ridge-basic (val R²=0.435) vs ridge-engineered (0.436) — essentially identical. H4 is weakly supported but the gain is within noise. The engineered features (seat_utilization, support_burden, champion_x_growth, expansion_capacity) don't carry independent signal beyond what the base features already capture.

**Finding 4 — H1 (log transform) is FALSIFIED.** Ridge-log: 0.435 → 0.126. XGB-log: 0.355 → 0.117. Log transform catastrophically hurts R² on original scale. Root cause: the top 1% of validation values (≥4,840 revenue) contribute **42.8% of SS_tot**. Log transform compresses those high values during training, causing systematic under-prediction of extreme cases. When back-transformed, those ~10 cases are wildly off, destroying R². This is a zero-inflated + extreme-tail combination problem — log transform is the wrong tool when the objective is original-scale R².

**Finding 5 — Val R² < Test R² for ridge.** Ridge-basic: val=0.435, test=0.615. The validation set (time-ordered middle 20%) appears harder to predict than the test set (final 20%). This is an unusual pattern — typically temporal drift makes the test set harder. Could indicate that the validation period had more variance in account behavior, or that some temporal signal improved into the test period.

### Implications for Next Cycles

1. **H1 is dead** — log transform is contraindicated for this target. The correct zero-inflation + right-skew treatment is either Tweedie loss (compound Poisson-Gamma, native in XGBoost) or a two-stage hurdle.
2. **H3 (two-stage hurdle) is now the top priority.** Separating zero/non-zero prediction from amount prediction should let us model each regime cleanly. The positive subset (74.7%) can use a model trained only on positive examples, avoiding the zero-pollution problem entirely.
3. **XGB needs fundamental rethinking.** Either: (a) try Tweedie objective which changes the loss surface, or (b) the model family is genuinely worse than ridge on this data. The heavy regularization isn't fixing the generalization gap — the problem may be temporal structure that makes train-period patterns not transfer.
4. **H2 (scale normalization)** should be tested in cycle 2 — predicting `expansion_rate = expansion / contract_value_t0` and back-multiplying. This would tell us if behavioral features carry propensity signal independent of scale.

### Surprise

The finding that ridge consistently beats XGB (0.435 vs 0.355) is counterintuitive given XGBoost's typical superiority on tabular data. One explanation: the signal is almost entirely scale-based (a near-linear relationship), and tree models pay a cost for the extra complexity while ridge exploits the linearity directly. If this holds after Tweedie and hurdle tests, this dataset may have a hard linear ceiling.

## Cycle 0002: Two-Stage Hurdle and Tweedie — Testing Zero-Inflation Treatments

### Objective

Test H3 (two-stage hurdle) and Tweedie XGBoost as zero-inflation treatments against the current best (ridge-basic, val R²=0.435). Run error analysis on ridge-basic first to understand where it fails and whether zero misclassification is contributing materially to RMSE.

### Hypothesis (written before running code)

**H3 (two-stage hurdle):** Ridge-basic must simultaneously fit the zero-expansion regime (25.3% of accounts) and the positive regime (74.7%). These are likely generated by different processes — a zero expansion may reflect accounts that were never expansion candidates, while the positive regime is scale + behavior. Separating the two lets Stage 1 learn the zero/positive boundary and Stage 2 learn the amount model on a cleaner signal. Expected gain: R² improves to 0.48–0.55 if zeros are currently dragging the amount model. If the hurdle doesn't beat ridge, this is evidence that (a) ridge is already predicting near-zero for zero accounts, and (b) the ceiling is in the feature space, not the model architecture.

**Tweedie XGB:** Changes the loss surface to the compound Poisson-Gamma family (p=1.5), which natively handles zero-inflation without log transform. Unlike square error, Tweedie penalizes relative errors rather than absolute errors, so large outliers don't dominate optimization. Risk: XGB has consistently overfit on this dataset (gap=0.44 in Cycle 1); Tweedie loss changes the surface but the temporal structure problem may remain. Expected: could close the overfitting gap partially; unclear if it beats ridge.

**Key falsification test:** If neither hurdle nor Tweedie beats ridge by more than ~0.02 R² (one bootstrap CI), the model family is not the bottleneck — the feature space ceiling is real and the next move should be feature engineering or data investigation, not more model variants.

### Pre-run error analysis (planned)

Before running candidates, run error analysis on ridge-basic to answer:
1. How much do zero accounts contribute to total error? Is ridge predicting positive values for them?
2. Where is variance concentrated — top decile, tail outliers?
3. Are there systematic under/over-prediction patterns by segment or prediction quartile?

### Error Analysis: Ridge-Basic on Validation Set

Key findings before running new candidates:

**Zero accounts — ridge predicts positive for 89% of them.** Mean residual on zero accounts: -401.3, MAE=410.8. These 254 accounts are dragging RMSE significantly. Scale features (contract_value_t0) lead ridge to predict positive expansion for large-revenue accounts that didn't expand.

**Top 10% owns 81.8% of SS_tot.** Extreme concentration: top 10% (y ≥ 1570, n=96) accounts for 81.8% of SS_tot; top 5% for 71.5%. R² is almost entirely determined by how well the model handles ~100 high-value accounts.

**Q4 predictions have catastrophic errors.** MAE=1098.8 for Q4 (highest-predicted) accounts. Ridge is systematically under-predicting the largest expansion cases while over-predicting large-revenue zero cases.

**|Residual| correlates r=0.801 with contract_value_t0.** Errors are pure scale. Behavioral features (usage_growth: r=0.089, champion_engagement: r=0.010) have near-zero correlation with residuals.

**Segment R² breakdown:**
- Enterprise: R²=0.357, SMB: R²=-0.039, Mid-market: R²=0.089
- The aggregate 0.435 is lifted by enterprise accounts having large SS_tot contributions; we have essentially no predictive power for SMB/mid-market

**Critical insight:** The top 15 highest-error cases include 6 zero accounts (y=0) with contract_value_t0 of 61K–245K being predicted 2.8K–5.7K. These are the "large-account non-expanders" — ridge sees a large account and predicts expansion because scale dominates, even when the account doesn't expand.

### Results

| Candidate | Train R² | Val R² | Val RMSE | Train-Val Gap |
|---|---|---|---|---|
| rule-baseline | 0.044 | 0.026 | 1018 | 0.018 |
| ridge-basic | 0.515 | 0.435 | 776 | 0.080 |
| ridge-engineered | 0.517 | 0.436 | 775 | 0.081 |
| hurdle-logistic-ridge | 0.517 | **0.437** | 774 | 0.080 |
| xgb-basic | 0.793 | 0.355 | 829 | 0.438 |
| xgb-tweedie | **0.811** | 0.350 | 832 | **0.461** |
| ridge-scale-norm (H2) | — | **-0.140** | — | — |

Bootstrap CIs (n=2000):
- ridge-basic: mean=0.427, 95% CI [0.271, 0.536]
- hurdle-logistic-ridge: mean=0.428, 95% CI [0.273, 0.538]
- Paired diff (hurdle − ridge): mean=+0.002, 95% CI [−0.006, +0.010]; P(hurdle > ridge)=71.1%

### Findings

**Finding 1 — H3 (hurdle) does not improve materially.** Hurdle val R²=0.437 vs ridge-basic=0.435 — difference of 0.002. Bootstrap CI for the difference: [−0.006, +0.010]. Zero is inside the CI; not significant. Root cause: the logistic classifier uses the same features as ridge. Large-revenue zero accounts fool the classifier too — `contract_value_t0` is such a strong predictor of expansion in the training data that the classifier also predicts "positive" for large-revenue non-expanders. The hurdle's soft-probability approach can't fix the fundamental scale confound.

**Finding 2 — XGB Tweedie is WORSE, not better.** Train R²=0.811 vs val R²=0.350 = gap of 0.461 — even more overfit than xgb-basic (0.438). Tweedie loss with p=1.5 up-weights small positive values in the compound Poisson-Gamma family. On this dataset, this may increase sensitivity to the noisy small-expansion signal in the training period (temporal structure), leading to more overfitting. Tweedie loss did not fix the generalization gap.

**Finding 3 — H2 (scale normalization) CONFIRMED: raw R² is almost entirely scale.** Ridge-scale-norm val R²=−0.140. Predicting expansion_rate = expansion/revenue, then back-multiplying, produces worse-than-mean predictions in absolute scale. This doesn't mean behavioral features are completely noise — it means (a) behavioral features can't predict the *rate* of expansion independent of scale, and (b) the back-multiplication amplifies errors catastrophically for large-revenue accounts. Combined with error analysis (|residual| r=0.801 with contract_value_t0, behavioral features r≈0.01–0.09), the conclusion is clear: **the signal in this feature set is almost entirely scale. Behavioral features (growth, engagement, adoption) carry negligible independent signal once scale is controlled.**

**Finding 4 — Feature space ceiling confirmed.** Three meaningfully different linear approaches (ridge-basic, ridge-engineered, hurdle) all cluster at 0.435–0.437. This is the classic "feature ceiling" pattern from cross-experiment learnings. Additional model variants will not escape this plateau.

**Finding 5 — XGB cannot generalize on this data.** Every XGB variant has train-val gaps of 0.44+ despite heavy regularization. The temporal structure (train=2023–2024, val=mid-period, test=2025) creates a regime that tree models can't bridge, while ridge's linear structure extrapolates across temporal shifts better.

### Implications / Experiment Conclusion

1. **The 0.435 R² ceiling is real and is a feature space limit.** No model variant using this feature set will significantly exceed it.
2. **The behavioral features in the heuristic (growth, engagement, adoption) have essentially no signal beyond what scale already captures.** This is the key business insight from this experiment: "big accounts expand more" is the entire story at this feature level.
3. **Two-stage hurdle and Tweedie are the right treatments for zero-inflation in principle, but can't work when the zero/positive boundary is determined by an unmeasured factor** — here, the large-revenue zero accounts differ from large-revenue positive accounts in ways not captured by the 26 features.
4. **Next move if this were a real experiment:** Feature engineering focused on explaining the *non-scale* variance — e.g., change signals (YoY growth in usage), historical expansion behavior, account health trajectories. Or framing the problem as a ranking problem rather than absolute prediction (reducing sensitivity to the top-tail concentration).

### Surprise

The hurdle failing to improve despite 89% false-positive rate on zeros is initially counterintuitive, but makes sense once you see the classifier using the same scale-dominated features. The classifier learned the same rule as ridge: "large revenue -> positive." The structural barrier is the *features*, not the model. More sophisticated architectures can't escape this.

Bootstrap CI also revealed how wide the R² uncertainty is: 95% CI [0.27, 0.54] for a model with a point estimate of 0.435. The extreme top-decile concentration (81.8% of SS_tot) means R² bounces sharply depending on which high-value accounts land in the bootstrap sample. Any claimed R² improvement < ~0.05 is noise on this dataset.
