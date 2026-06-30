from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from lib.candidate_result import CandidateResult
from lib.demo_regression.data import (
    BASE_NUMERIC_COLUMNS,
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERIC_COLUMNS,
    TARGET_COLUMN,
    DemoRegressionSplits,
    base_feature_columns,
    engineered_feature_columns,
)

logger = logging.getLogger(__name__)


OBJECTIVE_METRIC = "val_r2"
SPLIT_STRATEGY = "time_split_60_20_20"
RANDOM_STATE = 42


@dataclass(frozen=True)
class RegressionModelSpec:
    candidate_id: str
    model_family: str
    feature_set: str
    estimator: Any
    notes: str
    hyperparameters: dict[str, Any]
    preprocessor: ColumnTransformer


def _regression_preprocessor(feature_set: str) -> ColumnTransformer:
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


def _xgb_preprocessor(feature_set: str) -> ColumnTransformer:
    if feature_set not in ("base", "engineered"):
        raise ValueError(f"Unknown feature_set: {feature_set!r}. Must be 'base' or 'engineered'.")
    numeric_features = list(BASE_NUMERIC_COLUMNS)
    if feature_set == "engineered":
        numeric_features += ENGINEERED_NUMERIC_COLUMNS
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            ("boolean", "passthrough", BOOLEAN_COLUMNS),
        ]
    )


def _evaluate_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    r2 = float("nan") if y_true.nunique() < 2 else float(r2_score(y_true, predictions))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, predictions)),
    }


def _evaluate_all_splits(
    predictions_by_split: dict[str, np.ndarray], splits: DemoRegressionSplits
) -> dict[str, dict[str, float]]:
    frames = {
        "train": splits.train,
        "validation": splits.validation,
        "test": splits.test,
    }
    return {
        split_name: _evaluate_predictions(frames[split_name][TARGET_COLUMN], preds)
        for split_name, preds in predictions_by_split.items()
    }


def _fit_log_transform_model(
    spec: RegressionModelSpec, splits: DemoRegressionSplits
) -> CandidateResult:
    """Fit a pipeline with log(1+y) target transform; evaluate predictions on original scale."""
    selected_features = (
        engineered_feature_columns() if spec.feature_set == "engineered" else base_feature_columns()
    )
    inner_pipeline = Pipeline(
        steps=[
            ("preprocessor", spec.preprocessor),
            ("estimator", spec.estimator),
        ]
    )
    # TransformedTargetRegressor applies log1p to y before fitting, expm1 after predict
    wrapped = TransformedTargetRegressor(
        regressor=inner_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    train_x = splits.train[selected_features]
    train_y = splits.train[TARGET_COLUMN]
    validation_x = splits.validation[selected_features]
    test_x = splits.test[selected_features]

    wrapped.fit(train_x, train_y)
    raw_predictions = {
        "train": wrapped.predict(train_x),
        "validation": wrapped.predict(validation_x),
        "test": wrapped.predict(test_x),
    }
    metrics = _evaluate_all_splits(raw_predictions, splits)
    return CandidateResult(
        candidate_id=spec.candidate_id,
        model_family=spec.model_family,
        feature_set=spec.feature_set,
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["r2"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=spec.notes,
        metrics=metrics,
        hyperparameters=spec.hyperparameters,
        selected_features=selected_features,
    )


def _fit_pipeline_model(spec: RegressionModelSpec, splits: DemoRegressionSplits) -> CandidateResult:
    selected_features = (
        engineered_feature_columns() if spec.feature_set == "engineered" else base_feature_columns()
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", spec.preprocessor),
            ("estimator", spec.estimator),
        ]
    )
    train_x = splits.train[selected_features]
    train_y = splits.train[TARGET_COLUMN]
    validation_x = splits.validation[selected_features]
    test_x = splits.test[selected_features]

    pipeline.fit(train_x, train_y)
    predictions_by_split = {
        "train": pipeline.predict(train_x),
        "validation": pipeline.predict(validation_x),
        "test": pipeline.predict(test_x),
    }
    metrics = _evaluate_all_splits(predictions_by_split, splits)
    return CandidateResult(
        candidate_id=spec.candidate_id,
        model_family=spec.model_family,
        feature_set=spec.feature_set,
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["r2"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=spec.notes,
        metrics=metrics,
        hyperparameters=spec.hyperparameters,
        selected_features=selected_features,
    )


def run_rule_baseline(splits: DemoRegressionSplits) -> CandidateResult:
    rule_features = [
        "usage_growth_90d",
        "feature_adoption_count",
        "champion_engagement_score",
        "seat_utilization",
        "has_exec_sponsor",
        "multi_product",
    ]
    weights = np.array([0.24, 0.16, 0.18, 0.20, 0.12, 0.10], dtype=float)

    def raw_score(frame: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
        result = np.zeros(len(frame), dtype=float)
        for feature, weight in zip(rule_features, weights, strict=True):
            values = frame[feature].astype(float).values
            ref_values = np.sort(ref[feature].astype(float).values)
            ranks = np.searchsorted(ref_values, values) / max(len(ref_values), 1)
            result += weight * ranks
        return result

    train_raw = raw_score(splits.train, splits.train)
    val_raw = raw_score(splits.validation, splits.train)
    test_raw = raw_score(splits.test, splits.train)

    design = np.column_stack([np.ones_like(train_raw), train_raw])
    intercept, slope = np.linalg.lstsq(design, splits.train[TARGET_COLUMN].values, rcond=None)[0]

    predictions = {
        "train": intercept + slope * train_raw,
        "validation": intercept + slope * val_raw,
        "test": intercept + slope * test_raw,
    }
    metrics = _evaluate_all_splits(predictions, splits)
    return CandidateResult(
        candidate_id="rule-baseline",
        model_family="rule_based",
        feature_set="heuristic",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["r2"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes="Train-calibrated heuristic over growth, adoption, champion strength, and seat utilization.",
        metrics=metrics,
        hyperparameters={
            "weights": dict(zip(rule_features, weights.tolist(), strict=True)),
            "calibration": {
                "intercept": float(intercept),
                "slope": float(slope),
            },
        },
        selected_features=rule_features,
    )


def run_ridge_basic(splits: DemoRegressionSplits) -> CandidateResult:
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="ridge-basic",
            model_family="ridge_regression",
            feature_set="base",
            estimator=Ridge(alpha=2.0),
            notes="Ridge regression over base features with train-fitted scaling and one-hot categoricals.",
            hyperparameters={
                "alpha": 2.0,
            },
            preprocessor=_regression_preprocessor("base"),
        ),
        splits,
    )


def run_ridge_engineered(splits: DemoRegressionSplits) -> CandidateResult:
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="ridge-engineered",
            model_family="ridge_regression",
            feature_set="engineered",
            estimator=Ridge(alpha=3.0),
            notes="Ridge regression with engineered utilization, burden, and capacity features.",
            hyperparameters={
                "alpha": 3.0,
            },
            preprocessor=_regression_preprocessor("engineered"),
        ),
        splits,
    )


def run_xgb_basic(splits: DemoRegressionSplits) -> CandidateResult:
    estimator = XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="xgb-basic",
            model_family="xgboost_regressor",
            feature_set="engineered",
            estimator=estimator,
            notes="Regularized XGBoost regressor over base plus engineered regression features.",
            hyperparameters={
                "n_estimators": 260,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 12,
                "reg_alpha": 0.4,
                "reg_lambda": 4.0,
                "random_state": RANDOM_STATE,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
            },
            preprocessor=_xgb_preprocessor("engineered"),
        ),
        splits,
    )


def run_ridge_log(splits: DemoRegressionSplits) -> CandidateResult:
    """Ridge on base features with log(1+y) target transform; evaluate on original scale."""
    return _fit_log_transform_model(
        RegressionModelSpec(
            candidate_id="ridge-log",
            model_family="ridge_regression",
            feature_set="base",
            estimator=Ridge(alpha=2.0),
            notes="Ridge on base features with log(1+y) target transform; back-transform to evaluate on original scale.",
            hyperparameters={
                "alpha": 2.0,
                "target_transform": "log1p",
            },
            preprocessor=_regression_preprocessor("base"),
        ),
        splits,
    )


def run_xgb_log(splits: DemoRegressionSplits) -> CandidateResult:
    """XGBoost with log(1+y) target transform; tests H1 — log transform reduces right-tail MSE dominance."""
    estimator = XGBRegressor(
        n_estimators=260,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    return _fit_log_transform_model(
        RegressionModelSpec(
            candidate_id="xgb-log",
            model_family="xgboost_regressor",
            feature_set="engineered",
            estimator=estimator,
            notes="XGBoost with log(1+y) target transform; same architecture as xgb-basic, tests H1 for tree models.",
            hyperparameters={
                "n_estimators": 260,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 12,
                "reg_alpha": 0.4,
                "reg_lambda": 4.0,
                "random_state": RANDOM_STATE,
                "target_transform": "log1p",
            },
            preprocessor=_xgb_preprocessor("engineered"),
        ),
        splits,
    )


def run_hurdle_logistic_ridge(splits: DemoRegressionSplits) -> CandidateResult:
    """Two-stage hurdle: logistic classifier for P(zero vs. positive) on full train;
    ridge regressor on positive-only training subset for E[amount | positive].
    Final prediction: P(positive) * E[amount | positive] (soft combination).
    """
    selected_features = base_feature_columns()

    # Preprocessor fitted on full training set (scale stats from all accounts)
    preprocessor = _regression_preprocessor("base")
    train_x = splits.train[selected_features]
    train_x_transformed = preprocessor.fit_transform(train_x)

    # Stage 1: binary classifier (y == 0 vs. y > 0)
    train_y_binary = (splits.train[TARGET_COLUMN] > 0).astype(int)
    stage1 = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    stage1.fit(train_x_transformed, train_y_binary)

    # Stage 2: ridge regressor on positive subset only
    positive_mask = splits.train[TARGET_COLUMN] > 0
    train_x_pos = train_x_transformed[positive_mask.values]
    train_y_pos = splits.train.loc[positive_mask, TARGET_COLUMN]
    stage2 = Ridge(alpha=2.0)
    stage2.fit(train_x_pos, train_y_pos)

    def predict(df: pd.DataFrame) -> np.ndarray:
        x_t = preprocessor.transform(df[selected_features])
        prob_positive = stage1.predict_proba(x_t)[:, 1]
        amount = np.maximum(stage2.predict(x_t), 0.0)
        return prob_positive * amount

    predictions_by_split = {
        "train": predict(splits.train),
        "validation": predict(splits.validation),
        "test": predict(splits.test),
    }
    metrics = _evaluate_all_splits(predictions_by_split, splits)
    return CandidateResult(
        candidate_id="hurdle-logistic-ridge",
        model_family="two_stage_hurdle",
        feature_set="base",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["r2"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=(
            "Two-stage hurdle: logistic(base) for P(positive), "
            "ridge(base) on positive-only train subset for E[amount|positive]. "
            "Soft prediction: P(positive) * E[amount|positive]."
        ),
        metrics=metrics,
        hyperparameters={
            "stage1": {"model": "LogisticRegression", "C": 1.0},
            "stage2": {"model": "Ridge", "alpha": 2.0},
            "prediction_mode": "soft_probability",
        },
        selected_features=selected_features,
    )


def run_xgb_tweedie(splits: DemoRegressionSplits) -> CandidateResult:
    """XGBoost with Tweedie loss (power=1.5) — native single-model treatment for
    zero-inflated, right-skewed targets. Avoids log transform while changing the
    loss surface to handle zeros and positives as a compound Poisson-Gamma process.
    """
    estimator = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=12,
        reg_alpha=0.4,
        reg_lambda=4.0,
        random_state=RANDOM_STATE,
        objective="reg:tweedie",
        tweedie_variance_power=1.5,
    )
    return _fit_pipeline_model(
        RegressionModelSpec(
            candidate_id="xgb-tweedie",
            model_family="xgboost_tweedie",
            feature_set="base",
            estimator=estimator,
            notes=(
                "XGBoost with Tweedie loss (power=1.5) for zero-inflated right-skewed target. "
                "Single-model alternative to two-stage hurdle; same regularization as xgb-basic."
            ),
            hyperparameters={
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 12,
                "reg_alpha": 0.4,
                "reg_lambda": 4.0,
                "random_state": RANDOM_STATE,
                "objective": "reg:tweedie",
                "tweedie_variance_power": 1.5,
            },
            preprocessor=_xgb_preprocessor("base"),
        ),
        splits,
    )


def run_ridge_scale_normalized(splits: DemoRegressionSplits) -> CandidateResult:
    """H2 — scale normalization diagnostic: predict expansion_rate = target / contract_value_t0,
    then back-multiply by contract_value_t0. Tests whether behavioral features carry propensity
    signal independent of account size.
    """
    selected_features = base_feature_columns()
    revenue_col = "contract_value_t0"

    train_revenue = splits.train[revenue_col].values
    val_revenue = splits.validation[revenue_col].values
    test_revenue = splits.test[revenue_col].values

    floored = int((train_revenue < 1.0).sum())
    if floored > 0:
        logger.warning(
            "run_ridge_scale_normalized: %d train row(s) have %s < 1.0; flooring to 1.0 for rate computation",
            floored,
            revenue_col,
        )
    train_rate = splits.train[TARGET_COLUMN].values / np.maximum(train_revenue, 1.0)

    preprocessor = _regression_preprocessor("base")
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", Ridge(alpha=2.0)),
        ]
    )
    pipeline.fit(splits.train[selected_features], train_rate)

    def predict_amount(df: pd.DataFrame, revenue: np.ndarray) -> np.ndarray:
        rate_pred = pipeline.predict(df[selected_features])
        return np.maximum(rate_pred * revenue, 0.0)

    predictions_by_split = {
        "train": predict_amount(splits.train, train_revenue),
        "validation": predict_amount(splits.validation, val_revenue),
        "test": predict_amount(splits.test, test_revenue),
    }
    metrics = _evaluate_all_splits(predictions_by_split, splits)
    return CandidateResult(
        candidate_id="ridge-scale-norm",
        model_family="ridge_regression",
        feature_set="base_normalized",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["r2"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=(
            "H2 — predicts expansion_rate = target / contract_value_t0, back-multiplied. "
            "Diagnostic for scale confound: if R² drops significantly, raw R² is mostly scale."
        ),
        metrics=metrics,
        hyperparameters={
            "alpha": 2.0,
            "target_transform": "rate_normalization",
            "normalization_feature": revenue_col,
        },
        selected_features=selected_features,
    )


CANDIDATE_RUNNERS = {
    "rule-baseline": run_rule_baseline,
    "ridge-basic": run_ridge_basic,
    "ridge-engineered": run_ridge_engineered,
    "xgb-basic": run_xgb_basic,
    "ridge-log": run_ridge_log,
    "xgb-log": run_xgb_log,
    "hurdle-logistic-ridge": run_hurdle_logistic_ridge,
    "xgb-tweedie": run_xgb_tweedie,
    "ridge-scale-norm": run_ridge_scale_normalized,
}
