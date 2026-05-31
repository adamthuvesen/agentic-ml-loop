from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from lib.demo_classification.data import (
    TARGET_COLUMN,
    load_demo_dataset,
    split_demo_dataset,
)
from lib.diagnostics import generate_experiment_diagnostics
from runners.demo_classification_runner import init_demo


class TestGenerateExperimentDiagnostics:
    def test_writes_diagnostics_without_touching_results(self, tmp_path: Path) -> None:
        with patch("lib.runner.ROOT", tmp_path):
            exp_dir = init_demo()

        results_before = (exp_dir / "results.json").read_text()
        splits = split_demo_dataset(load_demo_dataset())

        report = generate_experiment_diagnostics(
            exp_dir,
            {
                "train": splits.train,
                "validation": splits.validation,
                "test": splits.test,
            },
            target_column=TARGET_COLUMN,
        )

        summary_path = exp_dir / "diagnostics" / "summary.md"
        report_path = exp_dir / "diagnostics" / "report.json"
        assert summary_path.exists()
        assert report_path.exists()
        assert "### Data Profile" in summary_path.read_text()
        assert report["splits"]
        assert json.loads(report_path.read_text())["splits"]
        assert (exp_dir / "results.json").read_text() == results_before

    def test_includes_error_patterns_when_predictions_are_available(self, tmp_path: Path) -> None:
        exp_dir = tmp_path / "experiments" / "demo_classification"
        exp_dir.mkdir(parents=True)
        (exp_dir / "results.json").write_text("[]\n")

        splits = split_demo_dataset(load_demo_dataset())
        validation = splits.validation.copy()
        validation["pred_prob"] = validation[TARGET_COLUMN].map({0: 0.1, 1: 0.9}).astype(float)
        flipped_rows = validation.index[:8]
        validation.loc[flipped_rows, "pred_prob"] = 1.0 - validation.loc[flipped_rows, "pred_prob"]

        report = generate_experiment_diagnostics(
            exp_dir,
            {
                "train": splits.train,
                "validation": validation,
                "test": splits.test,
            },
            target_column=TARGET_COLUMN,
            prediction_column="pred_prob",
        )

        assert report["error_patterns"]
        summary = (exp_dir / "diagnostics" / "summary.md").read_text()
        assert "### Error Patterns" in summary
