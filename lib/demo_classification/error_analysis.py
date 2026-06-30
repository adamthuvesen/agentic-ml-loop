"""Error analysis for logreg-basic on the demo classification dataset."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from lib.demo_classification.data import (
    TARGET_COLUMN,
    base_feature_columns,
    load_demo_dataset,
    split_demo_dataset,
)
from lib.demo_classification.modeling import RANDOM_STATE, _preprocessor

PROFILE_COLUMNS = [
    "usage_growth_90d",
    "feature_adoption_count",
    "account_age_days",
    "nps_score",
    "has_exec_sponsor",
]

HARD_CASE_COLUMNS = [
    "prob",
    "usage_growth_90d",
    "feature_adoption_count",
    "account_age_days",
    "has_exec_sponsor",
]


def _logreg_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor("base")),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )


def _fit_logreg(splits, features):
    pipeline = _logreg_pipeline()
    pipeline.fit(splits.train[features], splits.train[TARGET_COLUMN])
    val_probs = pipeline.predict_proba(splits.validation[features])[:, 1]
    val_y = splits.validation[TARGET_COLUMN].values
    return val_probs, val_y


def _print_metrics(val_y, val_probs) -> None:
    brier = brier_score_loss(val_y, val_probs)
    val_auc = roc_auc_score(val_y, val_probs)
    print(f"Val AUC: {val_auc:.4f}")
    print(f"Val Brier: {brier:.4f}")


def _print_prediction_distribution(val_probs) -> None:
    print("\nPrediction distribution:")
    print(f"  Mean: {val_probs.mean():.4f}")
    print(f"  Std:  {val_probs.std():.4f}")
    print(f"  Min:  {val_probs.min():.4f}")
    print(f"  Max:  {val_probs.max():.4f}")


def _confusion_masks(val_y, val_probs) -> dict[str, np.ndarray]:
    preds = (val_probs >= 0.5).astype(int)
    return {
        "fn": (val_y == 1) & (preds == 0),
        "fp": (val_y == 0) & (preds == 1),
        "tp": (val_y == 1) & (preds == 1),
        "tn": (val_y == 0) & (preds == 0),
    }


def _print_confusion(val_y, masks: dict[str, np.ndarray]) -> None:
    print("\nConfusion at threshold=0.5:")
    print(
        f"  TP={masks['tp'].sum()}, FP={masks['fp'].sum()}, "
        f"FN={masks['fn'].sum()}, TN={masks['tn'].sum()}"
    )
    print(f"  FN rate: {masks['fn'].sum() / (val_y == 1).sum():.3f}")
    print(f"  FP rate: {masks['fp'].sum() / (val_y == 0).sum():.3f}")


def _validation_frame(splits, features, val_y, val_probs):
    val_df = splits.validation[features].copy()
    val_df["prob"] = val_probs
    val_df["actual"] = val_y
    return val_df


def _print_error_profile(title: str, subset, val_df, val_y) -> None:
    print(f"\n--- {title} ---")
    print(f"Count: {len(subset)}")
    if len(subset) == 0:
        return
    for col in PROFILE_COLUMNS:
        print(
            f"  {col}: mean={subset[col].mean():.3f} "
            f"vs pos mean={val_df[val_y == 1][col].mean():.3f} "
            f"vs neg mean={val_df[val_y == 0][col].mean():.3f}"
        )


def _print_hard_cases(title: str, cases) -> None:
    print(f"\n--- {title} ---")
    print(f"Count: {len(cases)}")
    if len(cases) > 0:
        print(cases[HARD_CASE_COLUMNS].head(10).to_string())


def _xgb_pipeline() -> Pipeline:
    from xgboost import XGBClassifier

    return Pipeline(
        steps=[
            ("preprocessor", _preprocessor("base")),
            (
                "estimator",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=5,
                    reg_lambda=2.0,
                    reg_alpha=0.1,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def _fit_xgb_probs(splits, features):
    xgb_pipeline = _xgb_pipeline()
    xgb_pipeline.fit(splits.train[features], splits.train[TARGET_COLUMN])
    return xgb_pipeline.predict_proba(splits.validation[features])[:, 1]


def _paired_bootstrap_aucs(val_y, val_probs, xgb_probs) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(val_y)
    boot_logreg_aucs = []
    boot_xgb_aucs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(val_y[idx])) < 2:
            continue
        boot_logreg_aucs.append(roc_auc_score(val_y[idx], val_probs[idx]))
        boot_xgb_aucs.append(roc_auc_score(val_y[idx], xgb_probs[idx]))
    return np.array(boot_logreg_aucs), np.array(boot_xgb_aucs)


def _print_bootstrap_comparison(val_y, val_probs, xgb_probs) -> None:
    boot_logreg_aucs, boot_xgb_aucs = _paired_bootstrap_aucs(val_y, val_probs, xgb_probs)
    print("\n--- Paired Bootstrap CI: logreg-basic vs xgb-base ---")
    print(
        f"  logreg-basic: mean={boot_logreg_aucs.mean():.4f}  "
        f"95% CI [{np.percentile(boot_logreg_aucs, 2.5):.4f}, "
        f"{np.percentile(boot_logreg_aucs, 97.5):.4f}]"
    )
    print(
        f"  xgb-base:     mean={boot_xgb_aucs.mean():.4f}  "
        f"95% CI [{np.percentile(boot_xgb_aucs, 2.5):.4f}, "
        f"{np.percentile(boot_xgb_aucs, 97.5):.4f}]"
    )
    diffs = boot_logreg_aucs - boot_xgb_aucs
    print("\n--- Paired Bootstrap: logreg-basic - xgb-base ---")
    print(f"  Mean diff: {diffs.mean():.4f}")
    print(f"  95% CI: [{np.percentile(diffs, 2.5):.4f}, {np.percentile(diffs, 97.5):.4f}]")
    print(f"  P(logreg > xgb): {(diffs > 0).mean():.3f}")


def run_error_analysis() -> None:
    splits = split_demo_dataset(load_demo_dataset())
    features = base_feature_columns()
    val_probs, val_y = _fit_logreg(splits, features)

    _print_metrics(val_y, val_probs)
    _print_prediction_distribution(val_probs)

    masks = _confusion_masks(val_y, val_probs)
    _print_confusion(val_y, masks)

    val_df = _validation_frame(splits, features, val_y, val_probs)
    fn_df = val_df[masks["fn"]]
    _print_error_profile("False Negatives (missed positives)", fn_df, val_df, val_y)

    fp_df = val_df[masks["fp"]]
    _print_error_profile("False Positives (wrong expansions)", fp_df, val_df, val_y)

    hard_fn = fn_df[fn_df["prob"] < 0.35].sort_values("prob")
    _print_hard_cases("Highest-confidence false negatives (prob < 0.35, actual=1)", hard_fn)

    hard_fp = fp_df[fp_df["prob"] > 0.65].sort_values("prob", ascending=False)
    _print_hard_cases("Highest-confidence false positives (prob > 0.65, actual=0)", hard_fp)

    xgb_probs = _fit_xgb_probs(splits, features)
    _print_bootstrap_comparison(val_y, val_probs, xgb_probs)


if __name__ == "__main__":
    run_error_analysis()
