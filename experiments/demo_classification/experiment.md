# Model Search Experiment

## Title

Synthetic classification demo

## Goal

Demonstrate how a rule-based score can be compared against learned classifiers in a bounded autonomous loop.

## Baseline

A simple rule-based score using usage growth, adoption, and sponsor signals.

## Dataset

Bundled synthetic tabular dataset generated in-repo by `lib/demo_classification/generate_data.py`.

## Target

`will_expand_90d`

## Problem Type

Binary classification

## Split Strategy

Time split with 60% train, 20% validation, and 20% holdout test.

## Objective Metric

Validation AUC for candidate ranking (`val_auc`), with holdout AUC reported separately for transparency.

## Candidate Families

- rule-based baseline
- logistic regression
- xgboost

## Constraints

- deterministic demo
- no external data dependencies
- one bounded slice per loop cycle

## Success Definition

- establish a trustworthy baseline
- compare at least one linear and one tree-based candidate
- keep a valid leaderboard and summary
