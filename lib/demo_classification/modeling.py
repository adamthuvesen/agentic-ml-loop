from __future__ import annotations

from typing import Any

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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from lib.candidate_result import CandidateResult
from lib.demo_classification.data import (
    BASE_NUMERIC_COLUMNS,
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    DemoSplits,
    base_feature_columns,
    engineered_feature_columns,
)

OBJECTIVE_METRIC = "val_auc"
SPLIT_STRATEGY = "time_split_60_20_20"
RANDOM_STATE = 42


def _preprocessor(feature_set: str) -> ColumnTransformer:
    if feature_set not in ("base", "engineered"):
        raise ValueError(f"Unknown feature_set: {feature_set!r}. Must be 'base' or 'engineered'.")
    numeric_features = list(BASE_NUMERIC_COLUMNS)
    if feature_set == "engineered":
        numeric_features += ENGINEERED_NUMERIC_COLUMNS
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
                numeric_features,
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
                CATEGORICAL_COLUMNS,
            ),
            (
                "boolean",
                "passthrough",
                BOOLEAN_COLUMNS,
            ),
        ]
    )


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
    probabilities_by_split: dict[str, np.ndarray], splits: DemoSplits
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


def _fit_pipeline_model(
    candidate_id: str,
    model_family: str,
    feature_set: str,
    estimator: Any,
    splits: DemoSplits,
    notes: str,
    hyperparameters: dict[str, Any],
) -> CandidateResult:
    selected_features = (
        engineered_feature_columns() if feature_set == "engineered" else base_feature_columns()
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _preprocessor(feature_set)),
            ("estimator", estimator),
        ]
    )
    train_x = splits.train[selected_features]
    train_y = splits.train[TARGET_COLUMN]
    validation_x = splits.validation[selected_features]
    test_x = splits.test[selected_features]

    pipeline.fit(train_x, train_y)
    probabilities_by_split = {
        "train": pipeline.predict_proba(train_x)[:, 1],
        "validation": pipeline.predict_proba(validation_x)[:, 1],
        "test": pipeline.predict_proba(test_x)[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id=candidate_id,
        model_family=model_family,
        feature_set=feature_set,
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=notes,
        metrics=metrics,
        hyperparameters=hyperparameters,
        selected_features=selected_features,
    )


def run_rule_baseline(splits: DemoSplits) -> CandidateResult:
    def score(frame: pd.DataFrame) -> np.ndarray:
        raw = (
            0.35 * (frame["usage_growth_90d"] > 0.12).astype(float)
            + 0.20 * (frame["feature_adoption_count"] >= 4).astype(float)
            + 0.15 * (frame["seat_utilization"] > 0.22).astype(float)
            + 0.15 * frame["has_exec_sponsor"].astype(float)
            + 0.10 * frame["previous_expansion"].astype(float)
            + 0.05 * (frame["plan_tier"].isin(["pro", "business"])).astype(float)
        )
        return np.clip(0.05 + 0.90 * raw, 0.05, 0.95)

    probabilities = {
        "train": score(splits.train),
        "validation": score(splits.validation),
        "test": score(splits.test),
    }
    metrics = _evaluate_all_splits(probabilities, splits)
    return CandidateResult(
        candidate_id="rule-baseline",
        model_family="rule_based",
        feature_set="heuristic",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes="Thresholded heuristic over usage growth, feature adoption, and sponsor signals.",
        metrics=metrics,
        hyperparameters={},
        selected_features=[
            "usage_growth_90d",
            "feature_adoption_count",
            "seat_utilization",
            "has_exec_sponsor",
            "previous_expansion",
            "plan_tier",
        ],
    )


def run_logreg_basic(splits: DemoSplits) -> CandidateResult:
    return _fit_pipeline_model(
        candidate_id="logreg-basic",
        model_family="logistic_regression",
        feature_set="base",
        estimator=LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
        splits=splits,
        notes="Logistic regression over the base feature set.",
        hyperparameters={
            "max_iter": 500,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        },
    )


def run_logreg_engineered(splits: DemoSplits) -> CandidateResult:
    return _fit_pipeline_model(
        candidate_id="logreg-engineered",
        model_family="logistic_regression",
        feature_set="engineered",
        estimator=LogisticRegression(max_iter=800, random_state=RANDOM_STATE, solver="lbfgs"),
        splits=splits,
        notes="Logistic regression with engineered utilization, support intensity, and momentum features.",
        hyperparameters={
            "max_iter": 800,
            "random_state": RANDOM_STATE,
            "solver": "lbfgs",
        },
    )


def run_xgb_base(splits: DemoSplits) -> CandidateResult:
    estimator = XGBClassifier(
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
    )
    return _fit_pipeline_model(
        candidate_id="xgb-base",
        model_family="xgboost",
        feature_set="base",
        estimator=estimator,
        splits=splits,
        notes="XGBoost on base features with conservative regularization for small dataset.",
        hyperparameters={
            "n_estimators": 150,
            "max_depth": 3,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "reg_lambda": 2.0,
            "reg_alpha": 0.1,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
        },
    )


def run_xgb_basic(splits: DemoSplits) -> CandidateResult:
    estimator = XGBClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    return _fit_pipeline_model(
        candidate_id="xgb-basic",
        model_family="xgboost",
        feature_set="engineered",
        estimator=estimator,
        splits=splits,
        notes="XGBoost baseline over engineered features.",
        hyperparameters={
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
        },
    )


def run_xgb_tuned_light(splits: DemoSplits) -> CandidateResult:
    search_space = [
        {
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 180,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 220,
            "max_depth": 5,
            "learning_rate": 0.04,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
        },
        {
            "n_estimators": 140,
            "max_depth": 4,
            "learning_rate": 0.07,
            "subsample": 1.0,
            "colsample_bytree": 0.9,
        },
    ]

    selected_features = engineered_feature_columns()
    train_x = splits.train[selected_features]
    train_y = splits.train[TARGET_COLUMN]
    validation_x = splits.validation[selected_features]
    validation_y = splits.validation[TARGET_COLUMN]
    test_x = splits.test[selected_features]

    best_auc = float("-inf")
    best_pipeline: Pipeline | None = None
    best_params: dict[str, Any] | None = None
    for params in search_space:
        estimator = XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", **params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", _preprocessor("engineered")),
                ("estimator", estimator),
            ]
        )
        pipeline.fit(train_x, train_y)
        validation_probabilities = pipeline.predict_proba(validation_x)[:, 1]
        validation_auc = roc_auc_score(validation_y, validation_probabilities)
        if validation_auc > best_auc:
            best_auc = float(validation_auc)
            best_pipeline = pipeline
            best_params = {
                **params,
                "random_state": RANDOM_STATE,
                "eval_metric": "logloss",
            }

    if best_pipeline is None or best_params is None:
        raise RuntimeError("No valid pipeline found during tuning search.")
    probabilities_by_split = {
        "train": best_pipeline.predict_proba(train_x)[:, 1],
        "validation": best_pipeline.predict_proba(validation_x)[:, 1],
        "test": best_pipeline.predict_proba(test_x)[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id="xgb-tuned-light",
        model_family="xgboost",
        feature_set="engineered",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes="Light manual sweep across four XGBoost configs, selecting by validation AUC.",
        metrics=metrics,
        hyperparameters=best_params,
        selected_features=selected_features,
    )


CANDIDATE_RUNNERS = {
    "rule-baseline": run_rule_baseline,
    "logreg-basic": run_logreg_basic,
    "logreg-engineered": run_logreg_engineered,
    "xgb-base": run_xgb_base,
    "xgb-basic": run_xgb_basic,
    "xgb-tuned-light": run_xgb_tuned_light,
}
