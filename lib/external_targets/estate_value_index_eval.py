from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

AGENTIC_ROOT = Path(__file__).resolve().parents[2]
if str(AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTIC_ROOT))

from lib.external_targets.candidate_specs import apply_derived_features, load_candidate_spec  # noqa: E402,I001

OBJECTIVE_METRIC = "val_neg_mae"
SPLIT_STRATEGY = "temporal_60_20_20_by_sold_date"
TARGET_COLUMN = "sold_price"
RESULT_PREFIX = "RESULT_JSON:"
FORBIDDEN_FEATURES = {
    "sold_price",
    "price_change",
    "sold_date",
    "listing_id",
    "url",
    "address",
    "description",
    "images",
    "scraped_at",
    "scraped_at_date",
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "listing-price-baseline": {
        "family": "rule_based",
        "feature_set": "asking_price",
        "mode": "listing_price",
        "notes": "Predict sold_price directly from listing_price; no target fitting.",
        "features": ["listing_price"],
    },
    "median-baseline": {
        "family": "constant",
        "feature_set": "constant",
        "mode": "median",
        "notes": "Train-median sold_price baseline.",
        "features": [],
    },
    "lgbm-lean6": {
        "family": "lightgbm",
        "feature_set": "lean_6",
        "mode": "lgbm",
        "notes": "Target repo LightGBM defaults over the 6-feature compact set.",
    },
    "lgbm-lean8": {
        "family": "lightgbm",
        "feature_set": "lean_8",
        "mode": "lgbm",
        "notes": "Target repo LightGBM defaults over the 8-feature compact set.",
    },
    "lgbm-lean10": {
        "family": "lightgbm",
        "feature_set": "lean_10",
        "mode": "lgbm",
        "notes": "Target repo default 10-feature LightGBM training contract.",
    },
    "lgbm-lean15-area": {
        "family": "lightgbm",
        "feature_set": "lean_15",
        "mode": "lgbm",
        "notes": "Adds area categorical and a wider 15-feature compact set.",
    },
    "lgbm-lean10-pruned": {
        "family": "lightgbm",
        "feature_set": "lean_10_pruned",
        "mode": "lgbm",
        "source_feature_set": "lean_10",
        "importance_threshold": 0.02,
        "notes": "One bounded low-importance pruning pass at normalized importance <= 0.02.",
    },
    "lgbm-full-registry": {
        "family": "lightgbm",
        "feature_set": "full",
        "mode": "lgbm",
        "notes": "Full target feature registry, with target/time/id columns excluded.",
    },
}


@dataclass(frozen=True)
class MatrixSpec:
    frames: dict[str, pd.DataFrame]
    all_features: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


@dataclass(frozen=True)
class LGBMTrainingRequest:
    trainer: Any
    X_train: pd.DataFrame
    y_train: pd.Series
    categorical_indices: list[int]
    candidate: dict[str, Any]
    train_frame: pd.DataFrame


def _configure_target_repo(repo: Path) -> None:
    os.environ["GCS_ENABLED"] = "false"
    os.environ["EVI_CONFIG_FILE"] = str(repo / "config" / "pipeline_config.yaml")
    sys.path.insert(0, str(repo / "src"))


def _load_raw_frame(repo: Path, data_file: Path | None) -> pd.DataFrame:
    from estate_value_index.ml.training_workflow.data import load_training_dataframe

    data_path = data_file or repo / "data" / "raw" / "booli" / "booli_listings_prod.json"
    frame, skip_feature_engineering = load_training_dataframe(
        use_materialized_features=False,
        data_source="json",
        data_file=data_path,
    )
    if skip_feature_engineering:
        raise ValueError("Estate evaluator expects raw local JSON, not materialized features.")
    return frame


def _temporal_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values("sold_date").reset_index(drop=True)
    if len(ordered) < 30:
        raise ValueError(f"Need at least 30 rows for temporal 60/20/20 split; got {len(ordered)}")
    train_boundary = ordered.iloc[int(len(ordered) * 0.60)]["sold_date"]
    test_boundary = ordered.iloc[int(len(ordered) * 0.80)]["sold_date"]
    train = ordered[ordered["sold_date"] < train_boundary].copy()
    validation = ordered[
        (ordered["sold_date"] >= train_boundary) & (ordered["sold_date"] < test_boundary)
    ].copy()
    test = ordered[ordered["sold_date"] >= test_boundary].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Temporal boundary produced an empty train/validation/test split.")
    if not (train["sold_date"].max() < validation["sold_date"].min() < test["sold_date"].min()):
        raise ValueError("Temporal split failed strict chronological ordering.")
    return {"train": train, "validation": validation, "test": test}


def _engineer_splits(raw_frame: pd.DataFrame, *, include_test: bool) -> dict[str, pd.DataFrame]:
    from estate_value_index.ml import (
        build_feature_context,
        create_optimized_features,
        filter_valid_listings,
    )

    filtered = filter_valid_listings(raw_frame, min_price=3_000_000, drop_na_features=False)
    filtered = filtered.copy()
    filtered["sold_date"] = pd.to_datetime(filtered["sold_date"], errors="coerce")
    filtered = filtered.loc[filtered["sold_date"].notna()].reset_index(drop=True)
    split_raw = _temporal_split(filtered)
    train = create_optimized_features(split_raw["train"])
    context = build_feature_context(train)
    frames = {
        "train": train,
        "validation": create_optimized_features(split_raw["validation"], context=context),
    }
    if include_test:
        frames["test"] = create_optimized_features(split_raw["test"], context=context)
    return frames


def _load_feature_subset(
    repo: Path, feature_set: str | None
) -> tuple[list[str] | None, list[str] | None]:
    if feature_set in (None, "full"):
        return None, None
    config = yaml.safe_load((repo / "config" / "feature_subsets.yaml").read_text()) or {}
    subset = config[feature_set]
    return list(subset.get("numeric") or []), list(subset.get("categorical") or [])


def _resolve_features(
    repo: Path,
    frames: dict[str, pd.DataFrame],
    candidate: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    from estate_value_index.ml import get_feature_lists

    if any(key in candidate for key in ("numeric_features", "categorical_features")):
        base_numeric = list(candidate.get("numeric_features") or candidate.get("features") or [])
        base_categorical = list(candidate.get("categorical_features") or [])
    else:
        target_feature_set = candidate.get("source_feature_set", candidate["feature_set"])
        base_numeric, base_categorical, _ = get_feature_lists()
        subset_numeric, subset_categorical = _load_feature_subset(repo, target_feature_set)
        if subset_numeric is not None:
            base_numeric = subset_numeric
            base_categorical = subset_categorical or []

    available = set(frames["train"].columns)
    numeric = [
        col
        for col in base_numeric
        if col in available and col not in FORBIDDEN_FEATURES and col != TARGET_COLUMN
    ]
    categorical = [
        col
        for col in base_categorical
        if col in available and col not in FORBIDDEN_FEATURES and col != TARGET_COLUMN
    ]
    all_features = numeric + categorical
    if not all_features:
        raise ValueError(f"No usable features for candidate {candidate!r}")
    return numeric, categorical, all_features


def _target_values(frame: pd.DataFrame, candidate: dict[str, Any]) -> pd.Series:
    mode = candidate.get("target_mode", "log_price")
    if mode == "log_price":
        return np.log1p(frame[TARGET_COLUMN])
    if mode == "log_ratio_to_listing":
        return np.log(
            (frame[TARGET_COLUMN].astype(float) + 1.0)
            / (frame["listing_price"].astype(float) + 1.0)
        )
    if mode == "residual_to_listing":
        return frame[TARGET_COLUMN].astype(float) - frame["listing_price"].astype(float)
    raise ValueError(f"Unsupported estate target_mode {mode!r}")


def _inverse_target_values(
    frame: pd.DataFrame, raw_predictions: np.ndarray, candidate: dict[str, Any]
) -> np.ndarray:
    mode = candidate.get("target_mode", "log_price")
    if mode == "log_price":
        return np.expm1(raw_predictions)
    if mode == "log_ratio_to_listing":
        return (frame["listing_price"].astype(float).to_numpy() + 1.0) * np.exp(
            raw_predictions
        ) - 1.0
    if mode == "residual_to_listing":
        return frame["listing_price"].astype(float).to_numpy() + raw_predictions
    raise ValueError(f"Unsupported estate target_mode {mode!r}")


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs(y_pred - y_true.to_numpy()) / np.maximum(y_true.to_numpy(), 1.0)))
    return {
        "neg_mae": -mae,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _evaluate_predictions(
    frames: dict[str, pd.DataFrame],
    predictions: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {split: _metrics(frames[split][TARGET_COLUMN], predictions[split]) for split in frames}


def _run_baseline(
    frames: dict[str, pd.DataFrame], candidate_id: str, candidate: dict[str, Any]
) -> dict[str, Any]:
    if candidate["mode"] == "listing_price":
        predictions = {
            split: frames[split]["listing_price"].astype(float).to_numpy() for split in frames
        }
    else:
        model = DummyRegressor(strategy="median")
        model.fit(np.zeros((len(frames["train"]), 1)), frames["train"][TARGET_COLUMN])
        predictions = {split: model.predict(np.zeros((len(frames[split]), 1))) for split in frames}
    metrics = _evaluate_predictions(frames, predictions)
    return {
        "candidate_id": candidate_id,
        "model_family": candidate["family"],
        "feature_set": candidate["feature_set"],
        "objective_metric": OBJECTIVE_METRIC,
        "objective_score": metrics["validation"]["neg_mae"],
        "split_strategy": SPLIT_STRATEGY,
        "status": "completed",
        "notes": candidate["notes"],
        "metrics": metrics,
        "hyperparameters": {"leakage_guard": "no fit on validation/test labels"},
        "selected_features": candidate["features"],
    }


def _prepare_matrix(
    spec: MatrixSpec,
    split_name: str,
    train_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from estate_value_index.ml import handle_missing_values

    if split_name == "train":
        matrix, _, _, _ = handle_missing_values(
            spec.frames["train"][spec.all_features].copy(),
            spec.frames["train"][spec.all_features].copy(),
            spec.numeric_features,
            spec.categorical_features,
        )
        for col in spec.categorical_features:
            if col in matrix.columns and matrix[col].dtype == "object":
                matrix[col] = matrix[col].astype("category")
        return matrix
    if train_matrix is None:
        raise ValueError("train_matrix is required for non-train splits")
    _, matrix, _, _ = handle_missing_values(
        spec.frames["train"][spec.all_features].copy(),
        spec.frames[split_name][spec.all_features].copy(),
        spec.numeric_features,
        spec.categorical_features,
    )
    for col in spec.categorical_features:
        if col in matrix.columns and matrix[col].dtype == "object":
            matrix[col] = matrix[col].astype("category")
    return matrix


def _recency_weights(frame: pd.DataFrame, half_life_days: float | None) -> np.ndarray | None:
    if not half_life_days:
        return None
    dates = pd.to_datetime(frame["sold_date"], errors="coerce")
    age_days = (dates.max() - dates).dt.days.clip(lower=0).fillna(0).to_numpy()
    weights = np.power(0.5, age_days / float(half_life_days))
    mean_weight = float(np.mean(weights))
    return weights / mean_weight if mean_weight else weights


def _train_lgbm(request: LGBMTrainingRequest) -> Any:
    custom_params = request.candidate.get("params")
    sample_weight = _recency_weights(
        request.train_frame,
        request.candidate.get("sample_weight_half_life_days"),
    )
    if custom_params or sample_weight is not None:
        params = request.trainer.DEFAULT_PARAMS.copy()
        params.update(custom_params or {})
        request.trainer.best_params = params
        request.trainer.model = request.trainer._create_model(params)
        fit_kwargs: dict[str, Any] = {}
        if request.categorical_indices:
            fit_kwargs["categorical_feature"] = request.categorical_indices
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        request.trainer.model.fit(request.X_train, request.y_train, **fit_kwargs)
        return request.trainer.model
    return request.trainer.train(
        request.X_train,
        request.y_train,
        hyperparameter_tuning=bool(request.candidate.get("hyperparameter_tuning", False)),
        categorical_indices=request.categorical_indices,
    )


def _run_lgbm(
    repo: Path, frames: dict[str, pd.DataFrame], candidate_id: str, candidate: dict[str, Any]
) -> dict[str, Any]:
    from estate_value_index.ml import get_categorical_indices
    from estate_value_index.ml.training import LGBMTrainer

    numeric_features, categorical_features, all_features = _resolve_features(
        repo, frames, candidate
    )
    matrix_spec = MatrixSpec(
        frames=frames,
        all_features=all_features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    X_train = _prepare_matrix(matrix_spec, "train")
    X_validation = _prepare_matrix(matrix_spec, "validation", X_train)
    y_train_model = _target_values(frames["train"], candidate)

    trainer = LGBMTrainer(random_state=42)
    model = _train_lgbm(
        LGBMTrainingRequest(
            trainer=trainer,
            X_train=X_train,
            y_train=y_train_model,
            categorical_indices=get_categorical_indices(X_train, categorical_features),
            candidate=candidate,
            train_frame=frames["train"],
        )
    )

    dropped: list[str] = []
    if threshold := candidate.get("importance_threshold"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns).astype(float)
        total = float(importances.sum())
        normalized = importances / total if total > 0 else importances
        dropped = normalized[normalized <= float(threshold)].index.tolist()
        if dropped and len(all_features) - len(dropped) >= 5:
            all_features = [feature for feature in all_features if feature not in dropped]
            numeric_features = [feature for feature in numeric_features if feature not in dropped]
            categorical_features = [
                feature for feature in categorical_features if feature not in dropped
            ]
            matrix_spec = MatrixSpec(
                frames=frames,
                all_features=all_features,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            X_train = _prepare_matrix(matrix_spec, "train")
            X_validation = _prepare_matrix(matrix_spec, "validation", X_train)
            y_train_model = _target_values(frames["train"], candidate)
            trainer = LGBMTrainer(random_state=42)
            model = _train_lgbm(
                LGBMTrainingRequest(
                    trainer=trainer,
                    X_train=X_train,
                    y_train=y_train_model,
                    categorical_indices=get_categorical_indices(X_train, categorical_features),
                    candidate=candidate,
                    train_frame=frames["train"],
                )
            )

    predictions = {
        "train": _inverse_target_values(frames["train"], model.predict(X_train), candidate),
        "validation": _inverse_target_values(
            frames["validation"], model.predict(X_validation), candidate
        ),
    }
    if "test" in frames:
        X_test = _prepare_matrix(matrix_spec, "test", X_train)
        predictions["test"] = _inverse_target_values(
            frames["test"], model.predict(X_test), candidate
        )
    metrics = _evaluate_predictions(frames, predictions)
    return {
        "candidate_id": candidate_id,
        "model_family": candidate["family"],
        "feature_set": candidate["feature_set"],
        "objective_metric": OBJECTIVE_METRIC,
        "objective_score": metrics["validation"]["neg_mae"],
        "split_strategy": SPLIT_STRATEGY,
        "status": "completed",
        "notes": candidate["notes"],
        "metrics": metrics,
        "hyperparameters": {
            **trainer.get_best_params(),
            "pruned_features": dropped,
            "target_mode": candidate.get("target_mode", "log_price"),
            "sample_weight_half_life_days": candidate.get("sample_weight_half_life_days"),
            "derived_features": candidate.get("derived_features", []),
            "local_candidate_spec": candidate.get("local_candidate_spec"),
            "leakage_guard": "split before feature engineering; feature context fit on train only",
        },
        "selected_features": all_features,
    }


def evaluate(
    repo: Path,
    candidate_id: str,
    data_file: Path | None,
    *,
    include_test: bool,
    experiment_dir: Path | None,
) -> dict[str, Any]:
    _configure_target_repo(repo)
    start = time.perf_counter()
    candidate = load_candidate_spec(candidate_id, CANDIDATES, experiment_dir)
    frames = _engineer_splits(_load_raw_frame(repo, data_file), include_test=include_test)
    frames = {
        split: apply_derived_features(
            frame,
            candidate.get("derived_features"),
            forbidden_sources=FORBIDDEN_FEATURES | {TARGET_COLUMN},
        )
        for split, frame in frames.items()
    }
    if candidate["mode"] in {"listing_price", "median"}:
        payload = _run_baseline(frames, candidate_id, candidate)
    else:
        payload = _run_lgbm(repo, frames, candidate_id, candidate)
    payload["hyperparameters"]["runtime_seconds"] = round(time.perf_counter() - start, 3)
    payload["hyperparameters"]["cost_usd"] = 0.0
    payload["hyperparameters"]["data_file"] = str(
        data_file or repo / "data" / "raw" / "booli" / "booli_listings_prod.json"
    )
    payload["hyperparameters"]["split_sizes"] = {
        split: len(frame) for split, frame in frames.items()
    }
    payload["hyperparameters"]["split_date_ranges"] = {
        split: [
            str(frame["sold_date"].min().date()),
            str(frame["sold_date"].max().date()),
        ]
        for split, frame in frames.items()
    }
    payload["hyperparameters"]["final_holdout_included"] = include_test
    if experiment_dir is not None:
        payload["hyperparameters"]["experiment_dir"] = str(experiment_dir)
    return payload


def emit_result(payload: dict[str, Any]) -> None:
    import json

    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--experiment-dir", type=Path)
    args = parser.parse_args()
    emit_result(
        evaluate(
            args.repo.resolve(),
            args.candidate,
            args.data_file,
            include_test=args.include_test,
            experiment_dir=args.experiment_dir.resolve() if args.experiment_dir else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
