# Model Search Experiment

## Title

Synthetic regression demo

## Goal

Predict 180-day net revenue expansion amount for SaaS accounts using current health, product, and commercial signals.

## Baseline

A train-calibrated heuristic over growth, adoption, champion strength, and seat utilization.

## Dataset

Bundled synthetic tabular dataset generated in-repo by `lib/demo_regression/generate_data.py`.

## Data Profile

- **Row count**: 4800
- **Feature count**: 26 (13 base numeric, 5 engineered numeric, 4 categorical, 3 boolean, 1 datetime)
- **Feature types**: numeric, categorical, boolean, datetime (snapshot_date)
- **Missing values**: none
- **Target distribution**: 25.3% zeros, 74.7% positive; mean=574, median=117, max=13,339; skew=4.28
- **Top correlations with target**: contract_value_t0 (0.70), seat_count (0.60), avg_weekly_active_users (0.55) — all scale proxies

## Target

`net_revenue_change_180d`

## Problem Type

Zero-inflated regression (25.3% exact zeros, 74.7% positive, heavily right-skewed)

## Split Strategy

Time split with 60% train, 20% validation, and 20% holdout test, sorted by `snapshot_date`.

## Objective Metric

Validation R² (`val_r2`) for candidate ranking (`objective_score`). Secondary: validation RMSE and MAE; train-val R² gap as overfitting signal.

## Candidate Families

- rule-based heuristic baseline
- ridge regression (base features)
- ridge regression (engineered features)
- xgboost regressor

## Constraints

- deterministic demo
- no external data dependencies
- no holdout leakage into candidate ranking or model selection
- one bounded slice per loop cycle

## Known Risks

- **Scale confounders**: contract_value_t0 (r=0.70) and seat_count (r=0.60) dominate; models may be fitting account size rather than true expansion propensity
- **Right-tail outliers**: 48 rows above 99th percentile (~5,500+) can swing R² on validation; track RMSE and MAE alongside R²
- **Temporal drift**: 3-year span (2023–2025) may produce structural train-val gaps that look like overfitting but aren't fixable by regularization

## Evaluation Strategy

- **CV scheme**: single temporal split (no cross-validation; time-based ordering must be preserved)
- **Secondary metrics**: validation RMSE, validation MAE, train R² for gap tracking
- **Significance requirements**: bootstrap confidence intervals if candidates are within ~0.03 R² of each other

## Research Pointers

- Two-stage hurdle models for zero-inflated targets: classify zero vs. positive, then regress on positives only
- Tweedie loss (XGBoost/LightGBM objective) as a single-model alternative for zero-inflated right-skewed regression
- revenue-normalization test: predict expansion_rate = net_revenue_change_180d / contract_value_t0 to check if raw R² is mostly scale

## Research Hypotheses

**H1: Log/sqrt target transform improves all model R² scores** (high confidence)
- *Why:* Skew=4.28 and heavy right tail (median 117, max 13,339) penalize MSE-based fitting; log(1+y) compresses the tail and lets models learn from the bulk of the distribution. Supported by Tweedie reference and profiling.
- *Signal:* Val R² increases across ridge and XGB after transform vs. raw target

**H2: Scale normalization distinguishes true propensity signal from account-size correlation** (medium confidence)
- *Why:* contract_value_t0 (r=0.70) and seat_count (r=0.60) dominate; models may be implicitly fitting "big accounts expand more." Supported by learnings.md scale-confounder pattern.
- *Signal:* If predicting expansion_rate (expansion / revenue) and back-multiplying drops R² significantly, raw R² is mostly scale; if it holds, there is real propensity signal

**H3: Two-stage hurdle model outperforms single-objective regression** (medium confidence)
- *Why:* 25.3% zeros force a single regressor to fit both zero and positive regimes simultaneously; separating the stages should reduce bias. Strongly supported by web sources.
- *Signal:* Two-stage val R² > best single-model candidate without a proportional increase in train-val gap

**H4: Engineered features help ridge more than XGBoost** (medium confidence)
- *Why:* learnings.md: "ratio and interaction features often help linear models more than trees; tree models pay for them through split dilution and redundant noise."
- *Signal:* ridge-engineered > ridge-basic in R²; xgb-basic on engineered features doesn't improve meaningfully over base

## Success Definition

- Establish a trustworthy regression baseline
- Compare at least one linear and one tree-based candidate on the same split
- Determine whether the feature ceiling is scale-driven or contains real propensity signal
- Leave the loop with clear next directions (two-stage, log-transform, or revenue-normalization)
