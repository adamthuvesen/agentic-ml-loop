from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ranked_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return result entries ranked descending by numeric objective score."""
    return sorted(
        [
            result
            for result in results
            if isinstance(result, dict) and isinstance(result.get("objective_score"), (int, float))
        ],
        key=lambda result: result.get("objective_score", float("-inf")),
        reverse=True,
    )


def series_is_binary(series: pd.Series) -> bool:
    """Return True when a series has non-null values limited to binary labels."""
    if pd.api.types.is_float_dtype(series):
        return False
    if not (pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series)):
        values = pd.Series(series).dropna().unique().tolist()
        normalized: set[int] = set()
        for value in values:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return False
            if not numeric.is_integer():
                return False
            normalized.add(int(numeric))
        return bool(values) and normalized.issubset({0, 1})
    unique_vals = set(series.dropna().unique().tolist())
    return bool(unique_vals) and unique_vals <= {0, 1}


def series_is_numeric(series: pd.Series) -> bool:
    """Return True when pandas treats a series as numeric."""
    return pd.api.types.is_numeric_dtype(series)


def top_feature_drift(
    train: pd.DataFrame,
    other: pd.DataFrame,
    *,
    excluded_columns: set[str | None],
) -> list[dict[str, Any]]:
    """Return the largest normalized mean shifts between train and another split."""
    feature_drift: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in train.columns.intersection(other.columns)
        if column not in excluded_columns and series_is_numeric(train[column])
    ]
    for column in numeric_columns:
        train_values = pd.to_numeric(train[column], errors="coerce").dropna()
        other_values = pd.to_numeric(other[column], errors="coerce").dropna()
        if len(train_values) < 5 or len(other_values) < 5:
            continue
        delta = float(other_values.mean() - train_values.mean())
        scale = float(train_values.std(ddof=0))
        if not np.isfinite(scale) or scale == 0:
            scale = max(
                abs(float(train_values.mean())),
                abs(float(other_values.mean())),
                1e-6,
            )
        feature_drift.append(
            {
                "column": column,
                "delta": delta,
                "scaled_delta": float(abs(delta) / scale),
            }
        )
    feature_drift.sort(key=lambda item: item["scaled_delta"], reverse=True)
    return feature_drift[:3]
