from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

AGENTIC_ROOT = Path(__file__).resolve().parents[2]
if str(AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTIC_ROOT))

from lib.external_targets.candidate_specs import apply_derived_features, load_candidate_spec  # noqa: E402,I001

OBJECTIVE_METRIC = "val_r2_log"
SPLIT_STRATEGY = "expanding_year_folds_2015_2022_val_2023_test"
TARGET_COLUMN = "worldwide_gross"
RESULT_PREFIX = "RESULT_JSON:"
BUDGET_COLUMN = "production_budget"
YEAR_COLUMN = "release_year"
IMPUTE_RATIO = 0.4
IMPUTE_TOL = 1e-3
VALIDATION_YEARS = tuple(year for year in range(2015, 2023) if year != 2020)
TEST_YEAR = 2023
LEAKAGE_COLUMNS = {
    "social_media_buzz",
    "social_buzz_to_budget",
    "marketing_efficiency",
    "viral_potential",
    "buzz_to_votes_ratio",
}

PRODUCTION_FEATURES = [
    "votes",
    "production_budget",
    "ad_to_prod_ratio",
    "franchise_rating",
    "release_year",
    "mpaa_encoded",
    "company_freq",
    "genre_action",
    "genre_comedy",
    "super_genre_encoded",
    "is_covid_era",
    "is_july_4th_weekend",
    "is_weekend_release",
]

CANDIDATES: dict[str, dict[str, Any]] = {
    "log-budget-baseline": {
        "family": "rule_based",
        "feature_set": "budget_only",
        "mode": "log_budget",
        "features": ["production_budget"],
        "notes": "Target repo LogBudgetBaseline: log1p(revenue) ~ log1p(production_budget).",
    },
    "ridge-production-13": {
        "family": "ridge",
        "feature_set": "production_13",
        "mode": "ridge",
        "features": PRODUCTION_FEATURES,
        "notes": "Linear ridge sanity check over the target repo production feature contract.",
    },
    "xgb-production-13": {
        "family": "xgboost",
        "feature_set": "production_13",
        "mode": "xgb",
        "features": PRODUCTION_FEATURES,
        "notes": "Target repo production 13-feature XGBoost-style candidate.",
        "params": {
            "n_estimators": 700,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
        },
    },
    "xgb-compact-budget-demand": {
        "family": "xgboost",
        "feature_set": "compact_budget_demand",
        "mode": "xgb",
        "features": [
            "votes",
            "production_budget",
            "ad_to_prod_ratio",
            "franchise_rating",
            "rating",
            "runtime",
            "release_year",
        ],
        "notes": "Compact demand/budget feature subset to test whether the 13-feature contract is mostly redundant.",
        "params": {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.03,
            "reg_lambda": 0.2,
        },
    },
    "xgb-no-covid-weekend": {
        "family": "xgboost",
        "feature_set": "production_minus_covid_weekend",
        "mode": "xgb",
        "features": [
            feature
            for feature in PRODUCTION_FEATURES
            if feature not in {"is_covid_era", "is_weekend_release"}
        ],
        "notes": "Production features without COVID/weekend indicators to test release-window noise.",
        "params": {
            "n_estimators": 700,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
        },
    },
    "xgb-regularized-shallow": {
        "family": "xgboost",
        "feature_set": "production_13_regularized",
        "mode": "xgb",
        "features": PRODUCTION_FEATURES,
        "notes": "Shallower, more regularized XGBoost for small yearly validation folds.",
        "params": {
            "n_estimators": 500,
            "learning_rate": 0.04,
            "max_depth": 3,
            "min_child_weight": 8,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    },
    "random-forest-production-13": {
        "family": "random_forest",
        "feature_set": "production_13",
        "mode": "rf",
        "features": PRODUCTION_FEATURES,
        "notes": "Random forest control for non-boosted tree variance on yearly folds.",
    },
}


def _configure_target_repo(repo: Path) -> None:
    dummy_env = {
        "AWS_S3_BUCKET": "dummy",
        "SAGEMAKER_ROLE_ARN": "arn:aws:iam::123456789012:role/dummy",
        "SNOWFLAKE_USER": "dummy",
        "SNOWFLAKE_ACCOUNT": "dummy",
        "SNOWFLAKE_DATABASE": "BOX_OFFICE",
        "SNOWFLAKE_WAREHOUSE": "DUMMY",
        "SNOWFLAKE_ROLE": "DBT_RUNNER",
        "SNOWFLAKE_PRIVATE_KEY_PATH": "/tmp/dummy.pem",
        "AWS_REGION": "eu-north-1",
        "TMDB_API_TOKEN": "dummy",
        "SNOWFLAKE_SCHEMA_RAW": "RAW",
        "SNOWFLAKE_SCHEMA_STAGING": "STAGING",
        "SNOWFLAKE_SCHEMA_ML_TRAINING": "ML_TRAINING",
        "SNOWFLAKE_SCHEMA_FEATURE_STORE": "FEATURE_STORE",
    }
    for key, value in dummy_env.items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(repo))


def _load_snapshot(repo: Path, data_dir: Path | None) -> tuple[pd.DataFrame, pd.Series, int]:
    root = data_dir or repo / "analysis" / "datasets_high"
    x_path = root / "X_train.csv"
    y_path = root / "y_train.csv"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing box-office snapshot files under {root}")
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0].astype(float)
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")
    keep = (X[BUDGET_COLUMN] / y - IMPUTE_RATIO).abs() >= IMPUTE_TOL
    return (
        X.loc[keep].reset_index(drop=True),
        y.loc[keep].reset_index(drop=True),
        int((~keep).sum()),
    )


def _safe_features(X: pd.DataFrame, requested: list[str]) -> list[str]:
    missing = [feature for feature in requested if feature not in X.columns]
    if missing:
        raise ValueError(f"Snapshot missing requested features: {missing}")
    leaked = [feature for feature in requested if feature in LEAKAGE_COLUMNS]
    if leaked:
        raise ValueError(f"Candidate requested known leaked features: {leaked}")
    return requested


def _fold_metrics(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict[str, float]:
    y_true_dollars = np.expm1(y_true_log.to_numpy())
    y_pred_dollars = np.expm1(y_pred_log)
    return {
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "neg_rmse_log": -float(np.sqrt(np.mean((y_true_log.to_numpy() - y_pred_log) ** 2))),
        "mae_dollars": float(mean_absolute_error(y_true_dollars, y_pred_dollars)),
        "r2_dollars": float(r2_score(y_true_dollars, y_pred_dollars)),
        "median_ape": float(
            np.median(np.abs(y_pred_dollars - y_true_dollars) / np.maximum(y_true_dollars, 1.0))
        ),
    }


def _fit_predict(
    mode: str,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    X_val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "log_budget":
        from box_office.ml.backtest import LogBudgetBaseline

        model = LogBudgetBaseline().fit(X_train[BUDGET_COLUMN], y_train_log)
        return model.predict_log(X_train[BUDGET_COLUMN]), model.predict_log(X_val[BUDGET_COLUMN])
    if mode == "ridge":
        model = Ridge(alpha=5.0)
        model.fit(X_train, y_train_log)
        return model.predict(X_train), model.predict(X_val)
    if mode == "rf":
        model = RandomForestRegressor(
            n_estimators=260,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train_log)
        return model.predict(X_train), model.predict(X_val)
    if mode == "xgb":
        model = XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
        model.fit(X_train, y_train_log, verbose=False)
        return model.predict(X_train), model.predict(X_val)
    raise ValueError(f"Unknown candidate mode: {mode}")


def _evaluate_candidate(
    X: pd.DataFrame,
    y: pd.Series,
    candidate: dict[str, Any],
    *,
    include_test: bool,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    features = _safe_features(X, list(candidate["features"]))
    y_log = np.log1p(y)
    per_year: dict[int, dict[str, float]] = {}
    per_year_train: dict[int, dict[str, float]] = {}
    years_to_score = (*VALIDATION_YEARS, TEST_YEAR) if include_test else VALIDATION_YEARS
    for year in years_to_score:
        train_mask = X[YEAR_COLUMN] < year
        val_mask = X[YEAR_COLUMN] == year
        if train_mask.sum() < 2 or val_mask.sum() < 1:
            continue
        X_train = X.loc[train_mask, features].copy()
        X_val = X.loc[val_mask, features].copy()
        y_train_log = y_log.loc[train_mask]
        y_val_log = y_log.loc[val_mask]
        target_mode = candidate.get("target_mode", "log_revenue")
        train_offset = np.zeros(len(X_train))
        val_offset = np.zeros(len(X_val))
        y_model = y_train_log
        if target_mode == "residual_log_budget":
            from box_office.ml.backtest import LogBudgetBaseline

            baseline = LogBudgetBaseline().fit(X_train[BUDGET_COLUMN], y_train_log)
            train_offset = baseline.predict_log(X_train[BUDGET_COLUMN])
            val_offset = baseline.predict_log(X_val[BUDGET_COLUMN])
            y_model = y_train_log - train_offset
        elif target_mode == "log_roi":
            train_offset = np.log1p(X_train[BUDGET_COLUMN].to_numpy())
            val_offset = np.log1p(X_val[BUDGET_COLUMN].to_numpy())
            y_model = y_train_log - train_offset
        elif target_mode != "log_revenue":
            raise ValueError(f"Unsupported box-office target_mode {target_mode!r}")
        y_train_pred_log, y_pred_log = _fit_predict(
            candidate["mode"],
            candidate.get("params", {}),
            X_train,
            y_model,
            X_val,
        )
        y_train_pred_log = y_train_pred_log + train_offset
        y_pred_log = y_pred_log + val_offset
        per_year[int(year)] = {
            **_fold_metrics(y_val_log, y_pred_log),
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
        }
        per_year_train[int(year)] = _fold_metrics(y_train_log, y_train_pred_log)

    validation_values = [per_year[year] for year in VALIDATION_YEARS if year in per_year]
    train_values = [per_year_train[year] for year in VALIDATION_YEARS if year in per_year_train]
    test_value = per_year.get(TEST_YEAR)
    if not validation_values or not train_values:
        raise ValueError("Candidate did not produce validation folds.")
    if include_test and test_value is None:
        raise ValueError("Candidate did not produce the final test fold.")

    def avg(key: str, rows: list[dict[str, float]]) -> float:
        return float(np.mean([row[key] for row in rows]))

    metrics = {
        "train": {
            "r2_log": avg("r2_log", train_values),
            "neg_rmse_log": avg("neg_rmse_log", train_values),
            "mae_dollars": avg("mae_dollars", train_values),
            "r2_dollars": avg("r2_dollars", train_values),
            "median_ape": avg("median_ape", train_values),
        },
        "validation": {
            "r2_log": avg("r2_log", validation_values),
            "neg_rmse_log": avg("neg_rmse_log", validation_values),
            "mae_dollars": avg("mae_dollars", validation_values),
            "r2_dollars": avg("r2_dollars", validation_values),
            "median_ape": avg("median_ape", validation_values),
        },
    }
    if include_test:
        metrics["test"] = {
            "r2_log": test_value["r2_log"],
            "neg_rmse_log": test_value["neg_rmse_log"],
            "mae_dollars": test_value["mae_dollars"],
            "r2_dollars": test_value["r2_dollars"],
            "median_ape": test_value["median_ape"],
        }
    return metrics, {"per_year": per_year, "features": features}


def evaluate(
    repo: Path,
    candidate_id: str,
    data_dir: Path | None,
    *,
    include_test: bool,
    experiment_dir: Path | None,
) -> dict[str, Any]:
    _configure_target_repo(repo)
    start = time.perf_counter()
    X, y, dropped = _load_snapshot(repo, data_dir)
    candidate = load_candidate_spec(candidate_id, CANDIDATES, experiment_dir)
    X = apply_derived_features(
        X,
        candidate.get("derived_features"),
        forbidden_sources=LEAKAGE_COLUMNS | {TARGET_COLUMN},
    )
    metrics, details = _evaluate_candidate(X, y, candidate, include_test=include_test)
    return {
        "candidate_id": candidate_id,
        "model_family": candidate["family"],
        "feature_set": candidate["feature_set"],
        "objective_metric": OBJECTIVE_METRIC,
        "objective_score": metrics["validation"]["r2_log"],
        "split_strategy": SPLIT_STRATEGY,
        "status": "completed",
        "notes": candidate["notes"],
        "metrics": metrics,
        "hyperparameters": {
            **candidate.get("params", {}),
            "runtime_seconds": round(time.perf_counter() - start, 3),
            "cost_usd": 0.0,
            "dropped_imputed_budget_rows": dropped,
            "validation_years": list(VALIDATION_YEARS),
            "test_year": TEST_YEAR if include_test else None,
            "final_holdout_included": include_test,
            "target_mode": candidate.get("target_mode", "log_revenue"),
            "derived_features": candidate.get("derived_features", []),
            "local_candidate_spec": candidate.get("local_candidate_spec"),
            "leakage_guard": "known target-derived social/buzz columns excluded; train years precede eval year",
            "per_year": details["per_year"],
            "data_dir": str(data_dir or repo / "analysis" / "datasets_high"),
        },
        "selected_features": details["features"],
    }


def emit_result(payload: dict[str, Any]) -> None:
    import json

    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    emit_result(
        evaluate(
            args.repo.resolve(),
            args.candidate,
            args.data_dir,
            include_test=args.include_test,
            experiment_dir=args.experiment_dir.resolve() if args.experiment_dir else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
