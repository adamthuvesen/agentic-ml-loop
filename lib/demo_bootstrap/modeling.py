from __future__ import annotations

from collections.abc import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from lib.candidate_result import CandidateResult
from lib.demo_bootstrap.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    BootstrapSplits,
    feature_columns,
)

OBJECTIVE_METRIC = "val_auc"
SPLIT_STRATEGY = "time_split_60_20_20"
RANDOM_STATE = 42


def _evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    if y_true.nunique() < 2:
        auc = float("nan")
        ll = float("nan")
    else:
        auc = float(roc_auc_score(y_true, probabilities))
        ll = float(log_loss(y_true, probabilities, labels=[0, 1]))
    return {
        "auc": auc,
        "log_loss": ll,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def _evaluate_all_splits(
    probabilities_by_split: dict[str, np.ndarray], splits: BootstrapSplits
) -> dict[str, dict[str, float]]:
    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    return {
        split_name: _evaluate_predictions(frames[split_name][TARGET_COLUMN], probs)
        for split_name, probs in probabilities_by_split.items()
    }


def run_majority_baseline(splits: BootstrapSplits) -> CandidateResult:
    """Predict train-set positive rate for every row (sanity baseline)."""
    p = float(splits.train[TARGET_COLUMN].mean())
    probabilities = {
        "train": np.full(len(splits.train), p),
        "validation": np.full(len(splits.validation), p),
        "test": np.full(len(splits.test), p),
    }
    metrics = _evaluate_all_splits(probabilities, splits)
    feats = feature_columns()
    return CandidateResult(
        candidate_id="majority-baseline",
        model_family="constant",
        feature_set="none",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=f"Predict P(y=1)={p:.4f} for all rows (train prevalence).",
        metrics=metrics,
        hyperparameters={"train_positive_rate": p},
        selected_features=feats,
    )


def _logreg_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(NUMERIC_COLUMNS),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(CATEGORICAL_COLUMNS),
            ),
        ]
    )


def run_logreg_engineered(splits: BootstrapSplits) -> CandidateResult:
    """Logreg on hand-picked features: sessions_7d, is_enterprise, sessions_high.

    Drops pages_viewed (r=0.837 with sessions, hurts val AUC) and days_active
    (r=-0.018 with target, pure noise). Adds binary is_enterprise and
    sessions_high (>=5) to capture nonlinear effects found in EDA.
    """
    eng_cols = ["sessions_7d", "is_enterprise", "sessions_high"]
    train = splits.train.copy()
    val = splits.validation.copy()
    test = splits.test.copy()

    for df in [train, val, test]:
        df["is_enterprise"] = (df["segment"] == "enterprise").astype(int)
        df["sessions_high"] = (df["sessions_7d"] >= 5).astype(int)

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
                                ]
                            ),
                            eng_cols,
                        ),
                    ]
                ),
            ),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )
    train_y = train[TARGET_COLUMN]
    pipeline.fit(train[eng_cols], train_y)
    probabilities_by_split = {
        "train": pipeline.predict_proba(train[eng_cols])[:, 1],
        "validation": pipeline.predict_proba(val[eng_cols])[:, 1],
        "test": pipeline.predict_proba(test[eng_cols])[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id="logreg-engineered",
        model_family="logistic_regression",
        feature_set="engineered",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=(
            "Logreg on sessions_7d + is_enterprise + sessions_high. "
            "Drops pages_viewed (collinear) and days_active (noise)."
        ),
        metrics=metrics,
        hyperparameters={
            "max_iter": 500,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        },
        selected_features=eng_cols,
    )


def run_logreg_tiny(splits: BootstrapSplits) -> CandidateResult:
    cols = feature_columns()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _logreg_preprocessor()),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )
    train_x = splits.train[cols]
    train_y = splits.train[TARGET_COLUMN]
    val_x = splits.validation[cols]
    test_x = splits.test[cols]

    pipeline.fit(train_x, train_y)
    probabilities_by_split = {
        "train": pipeline.predict_proba(train_x)[:, 1],
        "validation": pipeline.predict_proba(val_x)[:, 1],
        "test": pipeline.predict_proba(test_x)[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    hyperparameters: dict[str, float | int | str] = {
        "max_iter": 500,
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
    }
    return CandidateResult(
        candidate_id="logreg-tiny",
        model_family="logistic_regression",
        feature_set="base",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes="Logistic regression on numeric + one-hot segment.",
        metrics=metrics,
        hyperparameters=hyperparameters,
        selected_features=cols,
    )


def run_logreg_poly(splits: BootstrapSplits) -> CandidateResult:
    """Logreg with degree-2 polynomial interactions on (sessions_7d, is_enterprise).

    Tests whether explicit interaction and quadratic terms capture the nonlinear
    session effect (conversion jumps at 5+ sessions) better than raw linear features.
    """
    eng_cols = ["sessions_7d", "is_enterprise"]
    train = splits.train.copy()
    val = splits.validation.copy()
    test = splits.test.copy()

    for df in [train, val, test]:
        df["is_enterprise"] = (df["segment"] == "enterprise").astype(int)

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
                LogisticRegression(
                    max_iter=500,
                    random_state=RANDOM_STATE,
                    solver="lbfgs",
                    C=1.0,
                ),
            ),
        ]
    )
    train_y = train[TARGET_COLUMN]
    pipeline.fit(train[eng_cols], train_y)
    probabilities_by_split = {
        "train": pipeline.predict_proba(train[eng_cols])[:, 1],
        "validation": pipeline.predict_proba(val[eng_cols])[:, 1],
        "test": pipeline.predict_proba(test[eng_cols])[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id="logreg-poly",
        model_family="logistic_regression",
        feature_set="poly-interactions",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=(
            "Logreg with degree-2 polynomial features on sessions_7d + is_enterprise. "
            "Tests nonlinear session effect and enterprise interaction."
        ),
        metrics=metrics,
        hyperparameters={
            "max_iter": 500,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
            "C": 1.0,
            "poly_degree": 2,
        },
        selected_features=eng_cols,
    )


def run_lgbm_conservative(splits: BootstrapSplits) -> CandidateResult:
    """LightGBM with aggressive regularization on base features.

    Tests whether a tree model captures nonlinear effects (session thresholds,
    segment interactions) natively without explicit feature engineering.
    Uses the same base feature set as logreg-tiny for fair comparison.
    """
    cols = feature_columns()
    train = splits.train.copy()
    val = splits.validation.copy()
    test = splits.test.copy()

    # Encode segment as category for LightGBM native handling
    for df in [train, val, test]:
        df["segment"] = df["segment"].astype("category")

    params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": 150,
        "num_leaves": 6,
        "max_depth": 3,
        "min_child_samples": 30,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 5.0,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        train[cols],
        train[TARGET_COLUMN],
        eval_set=[(val[cols], val[TARGET_COLUMN])],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    probabilities_by_split = {
        "train": model.predict_proba(train[cols])[:, 1],
        "validation": model.predict_proba(val[cols])[:, 1],
        "test": model.predict_proba(test[cols])[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id="lgbm-conservative",
        model_family="lightgbm",
        feature_set="base",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=(
            "LightGBM with aggressive regularization (num_leaves=6, max_depth=3, "
            "min_child_samples=30). Same base features as logreg-tiny. "
            f"Early stopped at {model.best_iteration_} iterations."
        ),
        metrics=metrics,
        hyperparameters={
            **{k: v for k, v in params.items() if k != "verbose"},
            "best_iteration": model.best_iteration_,
        },
        selected_features=cols,
    )


def run_logreg_minimal(splits: BootstrapSplits) -> CandidateResult:
    """Logreg on only the two cleanest signals: sessions_7d and is_enterprise.

    Drops pages_viewed (r=0.84 with sessions_7d, redundant), days_active
    (r=-0.002 with target, noise), and sessions_high threshold feature.
    Tests whether fewer features reduce the train-val gap without losing signal.
    """
    min_cols = ["sessions_7d", "is_enterprise"]
    train = splits.train.copy()
    val = splits.validation.copy()
    test = splits.test.copy()

    for df in [train, val, test]:
        df["is_enterprise"] = (df["segment"] == "enterprise").astype(int)

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )
    train_y = train[TARGET_COLUMN]
    pipeline.fit(train[min_cols], train_y)
    probabilities_by_split = {
        "train": pipeline.predict_proba(train[min_cols])[:, 1],
        "validation": pipeline.predict_proba(val[min_cols])[:, 1],
        "test": pipeline.predict_proba(test[min_cols])[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id="logreg-minimal",
        model_family="logistic_regression",
        feature_set="minimal",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=(
            "Logreg on sessions_7d + is_enterprise only. Drops collinear "
            "pages_viewed and noise feature days_active for tighter generalization."
        ),
        metrics=metrics,
        hyperparameters={
            "max_iter": 500,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        },
        selected_features=min_cols,
    )


CANDIDATE_RUNNERS: dict[str, Callable[[BootstrapSplits], CandidateResult]] = {
    "majority-baseline": run_majority_baseline,
    "logreg-tiny": run_logreg_tiny,
    "logreg-engineered": run_logreg_engineered,
    "logreg-minimal": run_logreg_minimal,
    "logreg-poly": run_logreg_poly,
    "lgbm-conservative": run_lgbm_conservative,
}
