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

ENGINEERED_COLUMNS = ["sessions_7d", "is_enterprise"]
PROFILE_COLUMNS = ("sessions_7d", "pages_viewed", "days_active", "is_enterprise")


def _add_enterprise_indicator(frame: pd.DataFrame) -> None:
    frame["is_enterprise"] = (frame["segment"] == "enterprise").astype(int)


def _train_validation_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_dataset()
    splits = split_dataset(df)
    train = splits.train.copy()
    val = splits.validation.copy()
    for frame in [train, val]:
        _add_enterprise_indicator(frame)
    return train, val


def _logreg_poly_pipeline() -> Pipeline:
    return Pipeline(
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
                            ENGINEERED_COLUMNS,
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


def _fit_validation_predictions() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    train, val = _train_validation_frames()
    pipeline = _logreg_poly_pipeline()
    pipeline.fit(train[ENGINEERED_COLUMNS], train[TARGET_COLUMN])
    val_probs = pipeline.predict_proba(val[ENGINEERED_COLUMNS])[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    return val, val[TARGET_COLUMN].values, val_probs, val_preds


def _print_validation_metrics(val_y: np.ndarray, val_probs: np.ndarray) -> None:
    auc = roc_auc_score(val_y, val_probs)
    brier = brier_score_loss(val_y, val_probs)
    print("=== logreg-poly validation metrics ===")
    print(f"AUC: {auc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print()


def _analysis_frame(
    val: pd.DataFrame,
    val_y: np.ndarray,
    val_preds: np.ndarray,
    val_probs: np.ndarray,
) -> pd.DataFrame:
    analysis = val[["sessions_7d", "days_active", "pages_viewed", "segment"]].copy()
    analysis["is_enterprise"] = (analysis["segment"] == "enterprise").astype(int)
    analysis["y_true"] = val_y
    analysis["y_pred"] = val_preds
    analysis["prob"] = val_probs
    analysis["error_type"] = "correct"
    analysis.loc[(val_y == 1) & (val_preds == 0), "error_type"] = "false_negative"
    analysis.loc[(val_y == 0) & (val_preds == 1), "error_type"] = "false_positive"
    return analysis


def _error_totals(frame: pd.DataFrame) -> tuple[int, int, int]:
    return (
        int((frame["error_type"] != "correct").sum()),
        int((frame["error_type"] == "false_negative").sum()),
        int((frame["error_type"] == "false_positive").sum()),
    )


def _print_error_breakdown(
    analysis: pd.DataFrame, val_y: np.ndarray, val_preds: np.ndarray
) -> None:
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


def _print_segment_error_rates(analysis: pd.DataFrame) -> None:
    print("=== Error rates by segment ===")
    for seg in ["smb", "mid", "enterprise"]:
        seg_mask = analysis["segment"] == seg
        seg_data = analysis[seg_mask]
        if len(seg_data) == 0:
            continue
        n_errors, n_fn, n_fp = _error_totals(seg_data)
        print(
            f"  {seg:12s}: {len(seg_data):3d} rows, "
            f"{n_errors:2d} errors ({n_errors / len(seg_data):.1%}), "
            f"FN={n_fn}, FP={n_fp}"
        )
    print()


def _print_sessions_bucket_error_rates(analysis: pd.DataFrame) -> None:
    print("=== Error rates by sessions_7d bucket ===")
    bins = [0, 2, 4, 6, 100]
    labels = ["0-2", "3-4", "5-6", "7+"]
    analysis["sess_bucket"] = pd.cut(analysis["sessions_7d"], bins=bins, labels=labels, right=True)
    for bucket in labels:
        bucket_mask = analysis["sess_bucket"] == bucket
        bucket_data = analysis[bucket_mask]
        if len(bucket_data) == 0:
            continue
        n_errors, n_fn, n_fp = _error_totals(bucket_data)
        pos_rate = bucket_data["y_true"].mean()
        avg_prob = bucket_data["prob"].mean()
        print(
            f"  sessions {bucket:4s}: {len(bucket_data):3d} rows, "
            f"actual_rate={pos_rate:.2f}, avg_pred={avg_prob:.2f}, "
            f"errors={n_errors} (FN={n_fn}, FP={n_fp})"
        )
    print()


def _print_days_active_error_rates(analysis: pd.DataFrame) -> None:
    print("=== Error rates by days_active (NOT in model) ===")
    for threshold_label, mask in [
        ("days_active < 4", analysis["days_active"] < 4),
        ("days_active >= 4", analysis["days_active"] >= 4),
    ]:
        subset = analysis[mask]
        if len(subset) == 0:
            continue
        n_errors, n_fn, n_fp = _error_totals(subset)
        pos_rate = subset["y_true"].mean()
        avg_prob = subset["prob"].mean()
        print(
            f"  {threshold_label:20s}: {len(subset):3d} rows, "
            f"actual_rate={pos_rate:.2f}, avg_pred={avg_prob:.2f}, "
            f"errors={n_errors} (FN={n_fn}, FP={n_fp})"
        )
    print()


def _print_calibration(val_y: np.ndarray, val_probs: np.ndarray) -> None:
    print("=== Calibration analysis ===")
    prob_true, prob_pred = calibration_curve(val_y, val_probs, n_bins=5, strategy="quantile")
    print("Bin  | Predicted | Actual  | Count")
    print("-----|-----------|---------|------")
    bin_edges = np.quantile(val_probs, np.linspace(0, 1, 6))
    for i, (pt, pp) in enumerate(zip(prob_true, prob_pred, strict=True)):
        low = bin_edges[i]
        high = bin_edges[i + 1]
        count = ((val_probs >= low) & (val_probs <= high)).sum()
        print(f"  {i + 1}  |  {pp:.3f}   | {pt:.3f}  |  {count}")
    print()


def _print_probability_distribution(val_probs: np.ndarray) -> None:
    print("=== Prediction probability distribution ===")
    print(f"  Min: {val_probs.min():.4f}")
    print(f"  25%: {np.percentile(val_probs, 25):.4f}")
    print(f"  50%: {np.percentile(val_probs, 50):.4f}")
    print(f"  75%: {np.percentile(val_probs, 75):.4f}")
    print(f"  Max: {val_probs.max():.4f}")
    print()


def _print_error_profile(title: str, subset: pd.DataFrame, analysis: pd.DataFrame) -> None:
    print(f"=== {title} ===")
    if len(subset) > 0:
        print(f"  Count: {len(subset)}")
        for column in PROFILE_COLUMNS[:3]:
            print(
                f"  Avg {column}: {subset[column].mean():.1f} "
                f"(overall: {analysis[column].mean():.1f})"
            )
        print(
            f"  Enterprise %: {subset['is_enterprise'].mean():.1%} "
            f"(overall: {analysis['is_enterprise'].mean():.1%})"
        )
        print(f"  Avg predicted prob: {subset['prob'].mean():.3f}")
        print(
            f"  days_active >= 4: {(subset['days_active'] >= 4).mean():.1%} "
            f"(overall: {(analysis['days_active'] >= 4).mean():.1%})"
        )
    print()


def _probability_bucket_mask(label: str, val_probs: np.ndarray) -> np.ndarray:
    if label.startswith("low"):
        return val_probs < 0.45
    if label.startswith("mid"):
        return (val_probs >= 0.45) & (val_probs <= 0.55)
    return val_probs > 0.55


def _print_residual_days_active_signal(analysis: pd.DataFrame, val_probs: np.ndarray) -> None:
    print("=== Residual signal check: days_active >= 4 ===")
    for bucket_label in [
        "low_prob (<0.45)",
        "mid_prob (0.45-0.55)",
        "high_prob (>0.55)",
    ]:
        subset = analysis[_probability_bucket_mask(bucket_label, val_probs)]
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


def _print_temporal_error_rates(analysis: pd.DataFrame) -> None:
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


def main() -> None:
    val, val_y, val_probs, val_preds = _fit_validation_predictions()
    _print_validation_metrics(val_y, val_probs)

    analysis = _analysis_frame(val, val_y, val_preds, val_probs)
    _print_error_breakdown(analysis, val_y, val_preds)
    _print_segment_error_rates(analysis)
    _print_sessions_bucket_error_rates(analysis)
    _print_days_active_error_rates(analysis)
    _print_calibration(val_y, val_probs)
    _print_probability_distribution(val_probs)

    fn = analysis[analysis["error_type"] == "false_negative"]
    fp = analysis[analysis["error_type"] == "false_positive"]
    _print_error_profile("False negative profile (missed positives)", fn, analysis)
    _print_error_profile("False positive profile (wrongly flagged)", fp, analysis)
    _print_residual_days_active_signal(analysis, val_probs)
    _print_temporal_error_rates(analysis)


if __name__ == "__main__":
    main()
