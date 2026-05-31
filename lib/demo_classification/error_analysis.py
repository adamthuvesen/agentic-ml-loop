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


def run_error_analysis() -> None:
    splits = split_demo_dataset(load_demo_dataset())
    features = base_feature_columns()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", _preprocessor("base")),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )
    pipeline.fit(splits.train[features], splits.train[TARGET_COLUMN])
    val_probs = pipeline.predict_proba(splits.validation[features])[:, 1]
    val_y = splits.validation[TARGET_COLUMN].values

    # Calibration / Brier score
    brier = brier_score_loss(val_y, val_probs)
    val_auc = roc_auc_score(val_y, val_probs)
    print(f"Val AUC: {val_auc:.4f}")
    print(f"Val Brier: {brier:.4f}")

    # Prediction distribution
    print("\nPrediction distribution:")
    print(f"  Mean: {val_probs.mean():.4f}")
    print(f"  Std:  {val_probs.std():.4f}")
    print(f"  Min:  {val_probs.min():.4f}")
    print(f"  Max:  {val_probs.max():.4f}")

    # False negatives and false positives at 0.5 threshold
    preds = (val_probs >= 0.5).astype(int)
    fn_mask = (val_y == 1) & (preds == 0)
    fp_mask = (val_y == 0) & (preds == 1)
    tp_mask = (val_y == 1) & (preds == 1)
    tn_mask = (val_y == 0) & (preds == 0)

    print("\nConfusion at threshold=0.5:")
    print(f"  TP={tp_mask.sum()}, FP={fp_mask.sum()}, FN={fn_mask.sum()}, TN={tn_mask.sum()}")
    print(f"  FN rate: {fn_mask.sum() / (val_y == 1).sum():.3f}")
    print(f"  FP rate: {fp_mask.sum() / (val_y == 0).sum():.3f}")

    # Analyze false negatives — what do missed positives look like?
    val_df = splits.validation[features].copy()
    val_df["prob"] = val_probs
    val_df["actual"] = val_y

    print("\n--- False Negatives (missed positives) ---")
    fn_df = val_df[fn_mask]
    print(f"Count: {len(fn_df)}")
    if len(fn_df) > 0:
        for col in [
            "usage_growth_90d",
            "feature_adoption_count",
            "account_age_days",
            "nps_score",
            "has_exec_sponsor",
        ]:
            print(
                f"  {col}: mean={fn_df[col].mean():.3f} vs pos mean={val_df[val_y == 1][col].mean():.3f} vs neg mean={val_df[val_y == 0][col].mean():.3f}"
            )

    print("\n--- False Positives (wrong expansions) ---")
    fp_df = val_df[fp_mask]
    print(f"Count: {len(fp_df)}")
    if len(fp_df) > 0:
        for col in [
            "usage_growth_90d",
            "feature_adoption_count",
            "account_age_days",
            "nps_score",
            "has_exec_sponsor",
        ]:
            print(
                f"  {col}: mean={fp_df[col].mean():.3f} vs pos mean={val_df[val_y == 1][col].mean():.3f} vs neg mean={val_df[val_y == 0][col].mean():.3f}"
            )

    # Hardest cases — highest confidence errors
    print("\n--- Highest-confidence false negatives (prob < 0.35, actual=1) ---")
    hard_fn = fn_df[fn_df["prob"] < 0.35].sort_values("prob")
    print(f"Count: {len(hard_fn)}")
    if len(hard_fn) > 0:
        print(
            hard_fn[
                [
                    "prob",
                    "usage_growth_90d",
                    "feature_adoption_count",
                    "account_age_days",
                    "has_exec_sponsor",
                ]
            ]
            .head(10)
            .to_string()
        )

    print("\n--- Highest-confidence false positives (prob > 0.65, actual=0) ---")
    hard_fp = fp_df[fp_df["prob"] > 0.65].sort_values("prob", ascending=False)
    print(f"Count: {len(hard_fp)}")
    if len(hard_fp) > 0:
        print(
            hard_fp[
                [
                    "prob",
                    "usage_growth_90d",
                    "feature_adoption_count",
                    "account_age_days",
                    "has_exec_sponsor",
                ]
            ]
            .head(10)
            .to_string()
        )

    # XGBoost for comparison
    from xgboost import XGBClassifier

    xgb_pipeline = Pipeline(
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
    xgb_pipeline.fit(splits.train[features], splits.train[TARGET_COLUMN])
    xgb_probs = xgb_pipeline.predict_proba(splits.validation[features])[:, 1]

    # Paired bootstrap: one idx per iteration scores BOTH models on the same
    # resample so the comparison is statistically valid.
    print("\n--- Paired Bootstrap CI: logreg-basic vs xgb-base ---")
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
    boot_logreg_aucs = np.array(boot_logreg_aucs)
    boot_xgb_aucs = np.array(boot_xgb_aucs)
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


if __name__ == "__main__":
    run_error_analysis()
