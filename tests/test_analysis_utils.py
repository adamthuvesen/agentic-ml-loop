from __future__ import annotations

import pandas as pd

from lib.analysis_utils import (
    ranked_results,
    series_is_binary,
    series_is_numeric,
    top_feature_drift,
)


class TestRankedResults:
    def test_sorts_numeric_objective_scores_descending(self) -> None:
        results = [
            {"candidate_id": "low", "objective_score": 0.1},
            {"candidate_id": "skip"},
            {"candidate_id": "high", "objective_score": 0.9},
        ]

        ranked = ranked_results(results)

        assert [item["candidate_id"] for item in ranked] == ["high", "low"]


class TestSeriesHelpers:
    def test_series_is_binary_accepts_integer_and_string_labels(self) -> None:
        assert series_is_binary(pd.Series([0, 1, 1, None], dtype="Int64"))
        assert series_is_binary(pd.Series(["0", "1", "1"]))

    def test_series_is_binary_rejects_float_dtype_and_multiclass(self) -> None:
        assert not series_is_binary(pd.Series([0.0, 1.0]))
        assert not series_is_binary(pd.Series([0, 1, 2]))

    def test_series_is_numeric_uses_pandas_dtype(self) -> None:
        assert series_is_numeric(pd.Series([1, 2, 3]))
        assert not series_is_numeric(pd.Series(["1", "2", "3"]))


class TestTopFeatureDrift:
    def test_returns_largest_scaled_mean_shift(self) -> None:
        train = pd.DataFrame(
            {
                "stable": [1, 1, 1, 1, 1],
                "shifted": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )
        validation = pd.DataFrame(
            {
                "stable": [1, 1, 1, 1, 1],
                "shifted": [10, 11, 12, 13, 14],
                "target": [1, 0, 1, 0, 1],
            }
        )

        drift = top_feature_drift(
            train,
            validation,
            excluded_columns={"target"},
        )

        assert drift[0]["column"] == "shifted"
        assert drift[0]["scaled_delta"] > 1
