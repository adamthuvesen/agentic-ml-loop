# Model Search Experiment

## Title

Synthetic regression demo

## Goal

Predict 180-day net revenue expansion amount for SaaS accounts using current health, product, and commercial signals.

## Baseline

A train-calibrated heuristic over growth, adoption, champion strength, and seat utilization.

## Dataset

Bundled synthetic tabular dataset generated in-repo by `lib/demo_regression/generate_data.py`.

## Target

`net_revenue_change_180d`

## Problem Type

Regression

## Split Strategy

Time split with 60% train, 20% validation, and 20% holdout test.

## Objective Metric

Validation R^2 for candidate ranking (`objective_score`), with holdout R^2 and validation/test RMSE reported separately.

## Candidate Families

- rule-based heuristic baseline
- ridge regression
- xgboost regressor

## Constraints

- deterministic demo
- no external data dependencies
- no holdout leakage into ranking or model selection
- one bounded slice per loop cycle

## Success Definition

- establish a trustworthy regression baseline
- compare at least one linear and one tree-based candidate
- keep a valid leaderboard and summary
