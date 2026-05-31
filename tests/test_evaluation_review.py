from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lib.demo_classification.data import (
    TARGET_COLUMN,
    load_demo_dataset,
    split_demo_dataset,
)
from lib.evaluation_review import _row_overlap_count, generate_evaluation_review
from runners.demo_classification_runner import init_demo


class TestGenerateEvaluationReview:
    def test_writes_review_without_touching_results_or_experiment_md(self, tmp_path: Path) -> None:
        with patch("lib.runner.ROOT", tmp_path):
            exp_dir = init_demo()

        results_before = (exp_dir / "results.json").read_text()
        experiment_before = (exp_dir / "experiment.md").read_text()
        splits = split_demo_dataset(load_demo_dataset())

        report = generate_evaluation_review(
            exp_dir,
            {
                "train": splits.train,
                "validation": splits.validation,
                "test": splits.test,
            },
            target_column=TARGET_COLUMN,
            results=[
                {
                    "candidate_id": "best",
                    "objective_metric": "val_auc",
                    "objective_score": 0.68,
                    "hyperparameters": {"val_auc_ci_95": [0.62, 0.70]},
                    "metrics": {"validation": {"precision": 0.0, "recall": 0.0}},
                },
                {
                    "candidate_id": "runner-up",
                    "objective_metric": "val_auc",
                    "objective_score": 0.67,
                    "hyperparameters": {"val_auc_ci_95": [0.63, 0.69]},
                },
            ],
        )

        assert (exp_dir / "evaluation_review.md").exists()
        assert (exp_dir / "evaluation_review.json").exists()
        assert report["concerns"]
        assert (exp_dir / "results.json").read_text() == results_before
        assert (exp_dir / "experiment.md").read_text() == experiment_before
        assert json.loads((exp_dir / "evaluation_review.json").read_text())["concerns"]

    def test_detects_known_concern_and_cleaner_case(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "bad-exp"
        bad_dir.mkdir()
        (bad_dir / "results.json").write_text("[]\n")

        train = pd.DataFrame(
            {
                "feature_a": list(range(20)),
                "feature_b": [0] * 20,
                TARGET_COLUMN: [0] * 20,
            }
        )
        validation = pd.concat([train.iloc[:5], train.iloc[:5]], ignore_index=True)
        validation[TARGET_COLUMN] = 1

        bad_report = generate_evaluation_review(
            bad_dir,
            {"train": train, "validation": validation},
            target_column=TARGET_COLUMN,
            results=[
                {
                    "candidate_id": "best",
                    "objective_metric": "val_auc",
                    "objective_score": 0.66,
                    "hyperparameters": {"val_auc_ci_95": [0.58, 0.68]},
                    "metrics": {"validation": {"precision": 0.0, "recall": 0.0}},
                },
                {
                    "candidate_id": "runner-up",
                    "objective_metric": "val_auc",
                    "objective_score": 0.655,
                    "hyperparameters": {"val_auc_ci_95": [0.60, 0.67]},
                },
            ],
        )
        assert bad_report["concerns"]

        clean_dir = tmp_path / "clean-exp"
        clean_dir.mkdir()
        (clean_dir / "results.json").write_text("[]\n")
        clean_train = pd.DataFrame(
            {
                "feature_a": list(range(20)),
                "feature_b": [0, 1] * 10,
                TARGET_COLUMN: [0, 1] * 10,
            }
        )
        clean_validation = pd.DataFrame(
            {
                "feature_a": list(range(5, 15)),
                "feature_b": [0, 1] * 5,
                TARGET_COLUMN: [0, 1] * 5,
            }
        )

        clean_report = generate_evaluation_review(
            clean_dir,
            {"train": clean_train, "validation": clean_validation},
            target_column=TARGET_COLUMN,
            results=[
                {
                    "candidate_id": "best",
                    "objective_metric": "val_auc",
                    "objective_score": 0.72,
                    "hyperparameters": {"val_auc_ci_95": [0.71, 0.73]},
                    "metrics": {"validation": {"precision": 0.4, "recall": 0.5}},
                },
                {
                    "candidate_id": "runner-up",
                    "objective_metric": "val_auc",
                    "objective_score": 0.64,
                    "hyperparameters": {"val_auc_ci_95": [0.62, 0.66]},
                },
            ],
        )
        assert clean_report["concerns"] == []


class TestRowOverlapCount:
    def test_auxiliary_split_column_does_not_hide_overlap(self) -> None:
        """Rows that differ only in a 'split' column should be detected as overlapping."""
        train = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4, 5],
                "feature_b": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
                "split": ["train"] * 5,
            }
        )
        val = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4, 5],
                "feature_b": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
                "split": ["validation"] * 5,  # differs from train
            }
        )
        # Excluding target + auxiliary columns should still find all 5 rows overlapping
        overlap = _row_overlap_count(train, val, excluded_columns={"target", "split"})
        assert overlap == 5

    def test_no_target_returns_zero(self) -> None:
        """When excluded_columns contains no real target, _row_overlap_count returns 0."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        # Only None in exclusion set — no real target column provided
        assert _row_overlap_count(df, df, excluded_columns={None}) == 0
