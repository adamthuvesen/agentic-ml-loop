# Demo: bootstrap smoke test

Synthetic marketing funnel data for **binary classification**: whether a trial user
**converted** to paid within 14 days of `event_date`.

## Target

- `converted` (0/1)

## Split

Prefer a **temporal** split: earlier events train, later events validate and test.
No user-level leakage — each row is one trial cohort snapshot.

## Metric

**Validation ROC-AUC** for ranking and comparing candidates.

## Baseline expectation

A **majority-class** probability baseline (train prevalence) should score near 0.5 AUC.
A simple **logistic regression** on numeric + one-hot segment should beat that on this
synthetic signal.
