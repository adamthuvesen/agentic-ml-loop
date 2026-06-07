from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

from lib.candidate_result import CandidateResult
from lib.demo_deep.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    DemoSplits,
    feature_columns,
)

OBJECTIVE_METRIC = "val_auc"
SPLIT_STRATEGY = "time_split_60_20_20"
RANDOM_STATE = 42
MAX_EPOCHS = 120
PATIENCE = 12
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def _preprocessor() -> ColumnTransformer:
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


def _fit_arrays(
    preprocessor: ColumnTransformer,
    splits: DemoSplits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = feature_columns()
    train_x = preprocessor.fit_transform(splits.train[cols])
    train_y = splits.train[TARGET_COLUMN].to_numpy(dtype=np.float32)
    val_x = preprocessor.transform(splits.validation[cols])
    val_y = splits.validation[TARGET_COLUMN].to_numpy(dtype=np.float32)
    test_x = preprocessor.transform(splits.test[cols])
    return train_x, train_y, val_x, val_y, test_x, splits.test[TARGET_COLUMN].to_numpy()


class TabularMLP(nn.Module):
    def __init__(self, in_features: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for index, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            if dropout > 0 and index < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _predict_proba(model: TabularMLP, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(features.astype(np.float32))
        logits = model(tensor)
        return torch.sigmoid(logits).numpy()


def _train_mlp(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    hidden_dims: Sequence[int],
    dropout: float,
) -> tuple[TabularMLP, int]:
    torch.manual_seed(RANDOM_STATE)
    model = TabularMLP(train_x.shape[1], hidden_dims, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(train_x.astype(np.float32))
    y_tensor = torch.from_numpy(train_y.astype(np.float32))
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    best_val_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

        val_probs = _predict_proba(model, val_x)
        if len(np.unique(val_y)) < 2:
            val_auc = float("-inf")
        else:
            val_auc = float(roc_auc_score(val_y, val_probs))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def _run_logreg_baseline(splits: DemoSplits) -> CandidateResult:
    cols = feature_columns()
    pipeline = Pipeline(
        steps=[
            ("preprocessor", _preprocessor()),
            (
                "estimator",
                LogisticRegression(max_iter=500, random_state=RANDOM_STATE, solver="lbfgs"),
            ),
        ]
    )
    pipeline.fit(splits.train[cols], splits.train[TARGET_COLUMN])
    probabilities_by_split = {
        "train": pipeline.predict_proba(splits.train[cols])[:, 1],
        "validation": pipeline.predict_proba(splits.validation[cols])[:, 1],
        "test": pipeline.predict_proba(splits.test[cols])[:, 1],
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    return CandidateResult(
        candidate_id="logreg-baseline",
        model_family="logistic_regression",
        feature_set="base",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes="Linear baseline on scaled numeric features and one-hot channel.",
        metrics=metrics,
        hyperparameters={"max_iter": 500, "random_state": RANDOM_STATE, "solver": "lbfgs"},
        selected_features=cols,
    )


def _run_mlp(
    splits: DemoSplits,
    *,
    candidate_id: str,
    hidden_dims: Sequence[int],
    dropout: float,
    notes: str,
) -> CandidateResult:
    cols = feature_columns()
    preprocessor = _preprocessor()
    train_x, train_y, val_x, val_y, test_x, _test_y = _fit_arrays(preprocessor, splits)
    model, best_epoch = _train_mlp(
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    probabilities_by_split = {
        "train": _predict_proba(model, train_x),
        "validation": _predict_proba(model, val_x),
        "test": _predict_proba(model, test_x),
    }
    metrics = _evaluate_all_splits(probabilities_by_split, splits)
    hyperparameters: dict[str, Any] = {
        "hidden_dims": list(hidden_dims),
        "dropout": dropout,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_epoch": best_epoch,
        "random_state": RANDOM_STATE,
    }
    return CandidateResult(
        candidate_id=candidate_id,
        model_family="pytorch_mlp",
        feature_set="base",
        objective_metric=OBJECTIVE_METRIC,
        objective_score=metrics["validation"]["auc"],
        split_strategy=SPLIT_STRATEGY,
        status="completed",
        notes=notes,
        metrics=metrics,
        hyperparameters=hyperparameters,
        selected_features=cols,
    )


def run_mlp_shallow(splits: DemoSplits) -> CandidateResult:
    return _run_mlp(
        splits,
        candidate_id="mlp-shallow",
        hidden_dims=(32,),
        dropout=0.0,
        notes="Single hidden layer (32 units). Early stopping on validation AUC.",
    )


def run_mlp_deep(splits: DemoSplits) -> CandidateResult:
    return _run_mlp(
        splits,
        candidate_id="mlp-deep",
        hidden_dims=(64, 32),
        dropout=0.2,
        notes="Two hidden layers with dropout. Targets the nonlinear ring + wave boundary.",
    )


def run_mlp_wide(splits: DemoSplits) -> CandidateResult:
    return _run_mlp(
        splits,
        candidate_id="mlp-wide",
        hidden_dims=(128, 64),
        dropout=0.1,
        notes="Wider MLP for extra capacity on the interaction terms.",
    )


CANDIDATE_RUNNERS: dict[str, Callable[[DemoSplits], CandidateResult]] = {
    "logreg-baseline": _run_logreg_baseline,
    "mlp-shallow": run_mlp_shallow,
    "mlp-deep": run_mlp_deep,
    "mlp-wide": run_mlp_wide,
}
