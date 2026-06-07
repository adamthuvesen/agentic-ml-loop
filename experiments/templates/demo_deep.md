# Model Search Experiment

## Title

Synthetic PyTorch tabular demo

## Goal

Show how a small feed-forward network on tabular features compares to a linear baseline when the label boundary is nonlinear.

## Baseline

Logistic regression on scaled numeric signals and one-hot channel.

## Dataset

Bundled synthetic tabular dataset generated in-repo by `lib/demo_deep/generate_data.py`.
The target mixes a ring boundary, feature interactions, and sinusoidal terms.

## Target

`will_convert`

## Problem Type

Binary classification

## Split Strategy

Time split with 60% train, 20% validation, and 20% holdout test.

## Objective Metric

Validation AUC for candidate ranking (`objective_score`), with holdout AUC reported separately for transparency.

## Candidate Families

- logistic regression
- pytorch MLP (shallow, deep, wide)

## Constraints

- deterministic demo
- CPU-friendly training (seconds, not minutes)
- requires `uv sync --extra deep`
- one bounded slice per loop cycle

## Success Definition

- linear baseline established
- at least one MLP beats logreg on validation AUC
- valid leaderboard and reproducible training seeds
