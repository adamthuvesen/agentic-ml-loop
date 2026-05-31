"""Bootstrap confidence intervals for top regression candidates.

Usage:
    uv run python lib/demo_regression/bootstrap_ci.py
"""

from __future__ import annotations


import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from lib.demo_regression.data import (
    TARGET_COLUMN,
    load_demo_regression_dataset,
    split_demo_regression_dataset,
    base_feature_columns,
)
from lib.demo_regression.modeling import _regression_preprocessor

RANDOM_STATE = 42
N_BOOTSTRAP = 2000


def _fit_ridge_basic(splits):
    selected = base_feature_columns()
    pipeline = Pipeline(
        [
            ("preprocessor", _regression_preprocessor("base")),
            ("estimator", Ridge(alpha=2.0)),
        ]
    )
    pipeline.fit(splits.train[selected], splits.train[TARGET_COLUMN])
    return pipeline.predict(splits.validation[selected])


def _fit_hurdle(splits):
    selected = base_feature_columns()
    preprocessor = _regression_preprocessor("base")
    train_x = preprocessor.fit_transform(splits.train[selected])

    train_y_binary = (splits.train[TARGET_COLUMN] > 0).astype(int)
    stage1 = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    stage1.fit(train_x, train_y_binary)

    positive_mask = splits.train[TARGET_COLUMN] > 0
    stage2 = Ridge(alpha=2.0)
    stage2.fit(
        train_x[positive_mask.values], splits.train.loc[positive_mask, TARGET_COLUMN]
    )

    val_x = preprocessor.transform(splits.validation[selected])
    prob_pos = stage1.predict_proba(val_x)[:, 1]
    amount = np.maximum(stage2.predict(val_x), 0.0)
    return prob_pos * amount


def bootstrap_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    r2_samples = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        if yt.std() < 1e-8:
            continue
        r2_samples.append(r2_score(yt, yp))
    r2_samples = np.array(r2_samples)
    return (
        float(np.mean(r2_samples)),
        float(np.percentile(r2_samples, 2.5)),
        float(np.percentile(r2_samples, 97.5)),
    )


def run() -> None:
    df = load_demo_regression_dataset()
    splits = split_demo_regression_dataset(df)
    val_y = splits.validation[TARGET_COLUMN].values

    ridge_preds = _fit_ridge_basic(splits)
    hurdle_preds = _fit_hurdle(splits)

    print("=" * 65)
    print(f"BOOTSTRAP R² CONFIDENCE INTERVALS (n_boot={N_BOOTSTRAP})")
    print("=" * 65)

    for name, preds in [
        ("ridge-basic", ridge_preds),
        ("hurdle-logistic-ridge", hurdle_preds),
    ]:
        mean, lo, hi = bootstrap_r2_ci(val_y, preds)
        print(f"  {name:<30}: mean={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print()
    # Paired bootstrap: is hurdle significantly better than ridge?
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(val_y)
    diff_samples = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        yt = val_y[idx]
        if yt.std() < 1e-8:
            continue
        r2_ridge = r2_score(yt, ridge_preds[idx])
        r2_hurdle = r2_score(yt, hurdle_preds[idx])
        diff_samples.append(r2_hurdle - r2_ridge)
    diff_samples = np.array(diff_samples)
    p_hurdle_better = float(np.mean(diff_samples > 0))
    print(
        f"  Hurdle vs Ridge: mean diff={diff_samples.mean():+.4f}, 95% CI [{np.percentile(diff_samples, 2.5):+.4f}, {np.percentile(diff_samples, 97.5):+.4f}]"
    )
    print(f"  P(hurdle > ridge): {p_hurdle_better:.1%}")
    print()
    print("=" * 65)


if __name__ == "__main__":
    run()
