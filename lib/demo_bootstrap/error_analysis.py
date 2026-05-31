"""Error analysis for the best candidate (logreg-poly) on demo_bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from lib.demo_bootstrap.data import (
    TARGET_COLUMN,
    load_dataset,
    split_dataset,
)


def main() -> None:
    df = load_dataset()
    splits = split_dataset(df)

    # Reproduce logreg-poly exactly
    eng_cols = ["sessions_7d", "is_enterprise"]
    train = splits.train.copy()
    val = splits.validation.copy()

    for frame in [train, val]:
        frame["is_enterprise"] = (frame["segment"] == "enterprise").astype(int)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                    (
                                        "poly",
                                        PolynomialFeatures(degree=2, include_bias=False),
                                    ),
                                ]
                            ),
                            eng_cols,
                        ),
                    ]
                ),
            ),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=42, solver="lbfgs", C=1.0),
            ),
        ]
    )
    pipeline.fit(train[eng_cols], train[TARGET_COLUMN])
    val_probs = pipeline.predict_proba(val[eng_cols])[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    val_y = val[TARGET_COLUMN].values

    # Sanity check
    auc = roc_auc_score(val_y, val_probs)
    brier = brier_score_loss(val_y, val_probs)
    print("=== logreg-poly validation metrics ===")
    print(f"AUC: {auc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print()

    # Build analysis dataframe
    analysis = val[["sessions_7d", "days_active", "pages_viewed", "segment"]].copy()
    analysis["is_enterprise"] = (analysis["segment"] == "enterprise").astype(int)
    analysis["y_true"] = val_y
    analysis["y_pred"] = val_preds
    analysis["prob"] = val_probs
    analysis["error_type"] = "correct"
    analysis.loc[(val_y == 1) & (val_preds == 0), "error_type"] = "false_negative"
    analysis.loc[(val_y == 0) & (val_preds == 1), "error_type"] = "false_positive"

    # Error counts
    print("=== Error breakdown (threshold=0.5) ===")
    error_counts = analysis["error_type"].value_counts()
    print(error_counts)
    print()

    fn = analysis[analysis["error_type"] == "false_negative"]
    fp = analysis[analysis["error_type"] == "false_positive"]
    tp = analysis[(val_y == 1) & (val_preds == 1)]
    tn = analysis[(val_y == 0) & (val_preds == 0)]

    print(f"True positives: {len(tp)}, False negatives: {len(fn)}")
    print(f"True negatives: {len(tn)}, False positives: {len(fp)}")
    print()

    # Error patterns by segment
    print("=== Error rates by segment ===")
    for seg in ["smb", "mid", "enterprise"]:
        seg_mask = analysis["segment"] == seg
        seg_data = analysis[seg_mask]
        if len(seg_data) == 0:
            continue
        n_errors = (seg_data["error_type"] != "correct").sum()
        n_fn = (seg_data["error_type"] == "false_negative").sum()
        n_fp = (seg_data["error_type"] == "false_positive").sum()
        print(
            f"  {seg:12s}: {len(seg_data):3d} rows, "
            f"{n_errors:2d} errors ({n_errors / len(seg_data):.1%}), "
            f"FN={n_fn}, FP={n_fp}"
        )
    print()

    # Error patterns by sessions_7d buckets
    print("=== Error rates by sessions_7d bucket ===")
    bins = [0, 2, 4, 6, 100]
    labels = ["0-2", "3-4", "5-6", "7+"]
    analysis["sess_bucket"] = pd.cut(analysis["sessions_7d"], bins=bins, labels=labels, right=True)
    for bucket in labels:
        bucket_mask = analysis["sess_bucket"] == bucket
        bucket_data = analysis[bucket_mask]
        if len(bucket_data) == 0:
            continue
        n_errors = (bucket_data["error_type"] != "correct").sum()
        n_fn = (bucket_data["error_type"] == "false_negative").sum()
        n_fp = (bucket_data["error_type"] == "false_positive").sum()
        pos_rate = bucket_data["y_true"].mean()
        avg_prob = bucket_data["prob"].mean()
        print(
            f"  sessions {bucket:4s}: {len(bucket_data):3d} rows, "
            f"actual_rate={pos_rate:.2f}, avg_pred={avg_prob:.2f}, "
            f"errors={n_errors} (FN={n_fn}, FP={n_fp})"
        )
    print()

    # Error patterns by days_active (threshold at >=4)
    print("=== Error rates by days_active (NOT in model) ===")
    for threshold_label, mask in [
        ("days_active < 4", analysis["days_active"] < 4),
        ("days_active >= 4", analysis["days_active"] >= 4),
    ]:
        subset = analysis[mask]
        if len(subset) == 0:
            continue
        n_errors = (subset["error_type"] != "correct").sum()
        n_fn = (subset["error_type"] == "false_negative").sum()
        n_fp = (subset["error_type"] == "false_positive").sum()
        pos_rate = subset["y_true"].mean()
        avg_prob = subset["prob"].mean()
        print(
            f"  {threshold_label:20s}: {len(subset):3d} rows, "
            f"actual_rate={pos_rate:.2f}, avg_pred={avg_prob:.2f}, "
            f"errors={n_errors} (FN={n_fn}, FP={n_fp})"
        )
    print()

    # Calibration analysis
    print("=== Calibration analysis ===")
    prob_true, prob_pred = calibration_curve(val_y, val_probs, n_bins=5, strategy="quantile")
    print("Bin  | Predicted | Actual  | Count")
    print("-----|-----------|---------|------")
    # Get bin counts manually
    bin_edges = np.quantile(val_probs, np.linspace(0, 1, 6))
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred, strict=True)):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        count = ((val_probs >= low) & (val_probs <= high)).sum()
        print(f"  {i + 1}  |  {pp:.3f}   | {pt:.3f}  |  {count}")
    print()

    # Prediction distribution
    print("=== Prediction probability distribution ===")
    print(f"  Min: {val_probs.min():.4f}")
    print(f"  25%: {np.percentile(val_probs, 25):.4f}")
    print(f"  50%: {np.percentile(val_probs, 50):.4f}")
    print(f"  75%: {np.percentile(val_probs, 75):.4f}")
    print(f"  Max: {val_probs.max():.4f}")
    print()

    # False negatives profile: what do missed positives look like?
    print("=== False negative profile (missed positives) ===")
    if len(fn) > 0:
        print(f"  Count: {len(fn)}")
        print(
            f"  Avg sessions_7d: {fn['sessions_7d'].mean():.1f} (overall: {analysis['sessions_7d'].mean():.1f})"
        )
        print(
            f"  Avg pages_viewed: {fn['pages_viewed'].mean():.1f} (overall: {analysis['pages_viewed'].mean():.1f})"
        )
        print(
            f"  Avg days_active: {fn['days_active'].mean():.1f} (overall: {analysis['days_active'].mean():.1f})"
        )
        print(
            f"  Enterprise %: {fn['is_enterprise'].mean():.1%} (overall: {analysis['is_enterprise'].mean():.1%})"
        )
        print(f"  Avg predicted prob: {fn['prob'].mean():.3f}")
        print(
            f"  days_active >= 4: {(fn['days_active'] >= 4).mean():.1%} (overall: {(analysis['days_active'] >= 4).mean():.1%})"
        )
    print()

    # False positives profile
    print("=== False positive profile (wrongly flagged) ===")
    if len(fp) > 0:
        print(f"  Count: {len(fp)}")
        print(
            f"  Avg sessions_7d: {fp['sessions_7d'].mean():.1f} (overall: {analysis['sessions_7d'].mean():.1f})"
        )
        print(
            f"  Avg pages_viewed: {fp['pages_viewed'].mean():.1f} (overall: {analysis['pages_viewed'].mean():.1f})"
        )
        print(
            f"  Avg days_active: {fp['days_active'].mean():.1f} (overall: {analysis['days_active'].mean():.1f})"
        )
        print(
            f"  Enterprise %: {fp['is_enterprise'].mean():.1%} (overall: {analysis['is_enterprise'].mean():.1%})"
        )
        print(f"  Avg predicted prob: {fp['prob'].mean():.3f}")
        print(
            f"  days_active >= 4: {(fp['days_active'] >= 4).mean():.1%} (overall: {(analysis['days_active'] >= 4).mean():.1%})"
        )
    print()

    # Check if days_active threshold adds signal AFTER conditioning on model features
    print("=== Residual signal check: days_active >= 4 ===")
    for bucket_label in [
        "low_prob (<0.45)",
        "mid_prob (0.45-0.55)",
        "high_prob (>0.55)",
    ]:
        if bucket_label.startswith("low"):
            mask = val_probs < 0.45
        elif bucket_label.startswith("mid"):
            mask = (val_probs >= 0.45) & (val_probs <= 0.55)
        else:
            mask = val_probs > 0.55
        subset = analysis[mask]
        if len(subset) < 5:
            continue
        active_high = subset[subset["days_active"] >= 4]
        active_low = subset[subset["days_active"] < 4]
        if len(active_high) >= 3 and len(active_low) >= 3:
            print(
                f"  {bucket_label}: "
                f"days>=4 actual_rate={active_high['y_true'].mean():.2f} (n={len(active_high)}), "
                f"days<4 actual_rate={active_low['y_true'].mean():.2f} (n={len(active_low)})"
            )
    print()

    # Temporal analysis (using row index as proxy for time ordering)
    print("=== Error rate by temporal position (val set) ===")
    n_val = len(analysis)
    half = n_val // 2
    early = analysis.iloc[:half]
    late = analysis.iloc[half:]
    for label, subset in [("Early val", early), ("Late val", late)]:
        n_err = (subset["error_type"] != "correct").sum()
        pos_rate = subset["y_true"].mean()
        avg_prob = subset["prob"].mean()
        print(
            f"  {label}: {len(subset)} rows, actual_rate={pos_rate:.2f}, "
            f"avg_pred={avg_prob:.2f}, errors={n_err} ({n_err / len(subset):.1%})"
        )


if __name__ == "__main__":
    main()
