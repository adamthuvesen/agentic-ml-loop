"""Error analysis for demo_regression — run against ridge-basic on the validation split.

Usage:
    uv run python lib/demo_regression/error_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from lib.demo_regression.data import (
    TARGET_COLUMN,
    base_feature_columns,
    load_demo_regression_dataset,
    split_demo_regression_dataset,
)
from lib.demo_regression.modeling import _regression_preprocessor

NUMERIC_CORRELATION_COLUMNS = [
    "contract_value_t0",
    "seat_count",
    "avg_weekly_active_users",
    "usage_growth_90d",
    "feature_adoption_count",
    "champion_engagement_score",
    TARGET_COLUMN,
    "_pred",
]


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", _regression_preprocessor("base")),
            ("estimator", Ridge(alpha=2.0)),
        ]
    )


def _fit_ridge_predictions(splits, selected_features):
    pipeline = _ridge_pipeline()
    pipeline.fit(splits.train[selected_features], splits.train[TARGET_COLUMN])
    return pipeline.predict(splits.validation[selected_features])


def _analysis_frame(splits, val_preds):
    val_true = splits.validation[TARGET_COLUMN].values
    residuals = val_true - val_preds  # positive = under-prediction, negative = over-prediction
    abs_residuals = np.abs(residuals)
    val = splits.validation.copy()
    val["_pred"] = val_preds
    val["_residual"] = residuals
    val["_abs_residual"] = abs_residuals
    val["_is_zero"] = val[TARGET_COLUMN] == 0
    return val, val_true, residuals, abs_residuals


def _print_header_metrics(val, val_true, residuals, abs_residuals) -> None:
    print("=" * 70)
    print("RIDGE-BASIC ERROR ANALYSIS — VALIDATION SET")
    print("=" * 70)
    print(f"  N validation rows:   {len(val)}")
    print(
        f"  Val R²:              {1 - np.sum(residuals**2) / np.sum((val_true - val_true.mean()) ** 2):.4f}"
    )
    print(f"  Val RMSE:            {np.sqrt(np.mean(residuals**2)):.1f}")
    print(f"  Val MAE:             {np.mean(abs_residuals):.1f}")
    print()


def _print_zero_positive_breakdown(val) -> None:
    print("─── Zero vs Positive Accounts ─────────────────────────────────────")
    zero_mask = val["_is_zero"]
    for label, mask in [
        ("Zero accounts (y=0)", zero_mask),
        ("Positive accounts (y>0)", ~zero_mask),
    ]:
        sub = val[mask]
        if len(sub) == 0:
            continue
        mean_err = sub["_residual"].mean()
        mae = sub["_abs_residual"].mean()
        pct_pos_pred = (sub["_pred"] > 0).mean()
        print(
            f"  {label}: n={len(sub)}, mean_residual={mean_err:.1f}, MAE={mae:.1f}, pct_pred_positive={pct_pos_pred:.1%}"
        )
    print()


def _print_tail_contribution(val_true) -> None:
    print("─── SS_tot Contribution by Target Percentile ──────────────────────")
    ss_tot = np.sum((val_true - val_true.mean()) ** 2)
    for pct in [90, 95, 99]:
        threshold = np.percentile(val_true, pct)
        tail_mask = val_true >= threshold
        ss_tail = np.sum((val_true[tail_mask] - val_true.mean()) ** 2)
        print(
            f"  Top {100 - pct}% (y >= {threshold:.0f}): n={tail_mask.sum()}, SS_tot share={ss_tail / ss_tot:.1%}"
        )
    print()


def _print_prediction_quartiles(val) -> None:
    print("─── Mean Residual by Prediction Quartile ──────────────────────────")
    val["_pred_q"] = pd.qcut(val["_pred"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    for q in ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]:
        sub = val[val["_pred_q"] == q]
        print(
            f"  {q}: n={len(sub)}, mean_residual={sub['_residual'].mean():.1f}, MAE={sub['_abs_residual'].mean():.1f}"
        )
    print()


def _print_top_errors(val) -> None:
    print("─── Top 15 Highest Absolute Residual Cases ─────────────────────────")
    top_errors = val.nlargest(15, "_abs_residual")[
        [
            "_abs_residual",
            "_residual",
            TARGET_COLUMN,
            "_pred",
            "contract_value_t0",
            "seat_count",
            "_is_zero",
        ]
    ]
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(top_errors.to_string(index=False))
    print()


def _print_residual_correlations(val) -> None:
    print("─── Correlation of |Residual| with Key Features ────────────────────")
    corrs = (
        val[NUMERIC_CORRELATION_COLUMNS + ["_abs_residual"]]
        .corr()["_abs_residual"]
        .drop("_abs_residual")
        .sort_values(ascending=False)
    )
    for feat, corr in corrs.items():
        print(f"  {feat:<35s}: {corr:+.3f}")
    print()


def _print_segment_r2(val) -> None:
    print("─── Val R² by Segment ──────────────────────────────────────────────")
    for seg_col in ["plan_tier", "segment"]:
        if seg_col not in val.columns:
            continue
        print(f"  {seg_col}:")
        for seg_val in sorted(val[seg_col].unique()):
            sub = val[val[seg_col] == seg_val]
            y_t = sub[TARGET_COLUMN].values
            if len(sub) < 5 or y_t.std() < 1:
                continue
            y_p = sub["_pred"].values
            r2 = 1 - np.sum((y_t - y_p) ** 2) / np.sum((y_t - y_t.mean()) ** 2)
            print(
                f"    {seg_val}: n={len(sub)}, R²={r2:.3f}, mean_y={y_t.mean():.0f}, mean_pred={y_p.mean():.0f}"
            )
    print()


def run_error_analysis() -> None:
    df = load_demo_regression_dataset()
    splits = split_demo_regression_dataset(df)
    selected_features = base_feature_columns()

    val_preds = _fit_ridge_predictions(splits, selected_features)
    val, val_true, residuals, abs_residuals = _analysis_frame(splits, val_preds)

    _print_header_metrics(val, val_true, residuals, abs_residuals)
    _print_zero_positive_breakdown(val)
    _print_tail_contribution(val_true)
    _print_prediction_quartiles(val)
    _print_top_errors(val)
    _print_residual_correlations(val)
    _print_segment_r2(val)

    print("=" * 70)
    print("END OF ERROR ANALYSIS")
    print("=" * 70)


if __name__ == "__main__":
    run_error_analysis()
