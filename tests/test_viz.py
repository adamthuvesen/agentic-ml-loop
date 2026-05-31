from __future__ import annotations

import json
from pathlib import Path

from viz.bundle import bundle
from viz.generate import _validation_total_for_experiment, parse_results_json


class TestValidationTotals:
    def test_non_value_metric_has_no_validation_total(self) -> None:
        experiment_dir = Path("experiments/demo_classification")
        assert _validation_total_for_experiment(experiment_dir, "val_auc") is None


class TestParseResultsJson:
    def test_parses_captured_at_k_metrics(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "value-model",
                        "model_family": "lightgbm",
                        "objective_metric": "val_captured_at_20pct",
                        "objective_score": 0.6789,
                        "metrics": {
                            "validation": {
                                "captured_at_10pct": 0.4,
                                "captured_at_20pct": 0.6789,
                                "auc": 0.71,
                            }
                        },
                    }
                ]
            )
        )

        candidate = parse_results_json(path)[0]

        assert candidate["primary_metric"]["family"] == "captured_at_k"
        assert candidate["at_20"] == 0.679
        assert candidate["auc"] == 0.71

    def test_parses_classification_metrics_without_fake_capture_fields(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "logreg",
                        "model_family": "logistic_regression",
                        "objective_metric": "val_auc",
                        "objective_score": 0.8123,
                        "metrics": {
                            "validation": {
                                "auc": 0.8123,
                                "precision": 0.5,
                                "recall": 0.4,
                            }
                        },
                    }
                ]
            )
        )

        candidate = parse_results_json(path)[0]

        assert candidate["primary_metric"]["family"] == "classification"
        assert candidate["metrics"]["auc"] == 0.812
        assert "at_20" not in candidate

    def test_parses_regression_metrics_without_fake_capture_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "ridge",
                        "model_family": "ridge",
                        "objective_metric": "val_r2",
                        "objective_score": 0.4369,
                        "metrics": {"validation": {"r2": 0.4369, "rmse": 12.345}},
                    }
                ]
            )
        )

        candidate = parse_results_json(path)[0]

        assert candidate["primary_metric"]["family"] == "regression"
        assert candidate["metrics"]["r2"] == 0.437
        assert candidate["metrics"]["rmse"] == 12.345
        assert "at_20" not in candidate

    def test_parses_unknown_metrics_generically(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "custom",
                        "model_family": "custom",
                        "objective_metric": "val_business_score",
                        "objective_score": 2.5,
                        "metrics": {"validation": {"business_score": 2.5}},
                    }
                ]
            )
        )

        candidate = parse_results_json(path)[0]

        assert candidate["primary_metric"]["family"] == "generic"
        assert candidate["metrics"]["business_score"] == 2.5
        assert "at_20" not in candidate

    def test_preserves_zero_scores(self, tmp_path: Path) -> None:
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "zero",
                        "model_family": "constant",
                        "objective_metric": "val_auc",
                        "objective_score": 0.0,
                        "metrics": {"validation": {"auc": 0.0}},
                    }
                ]
            )
        )

        candidate = parse_results_json(path)[0]

        assert candidate["score"] == 0.0
        assert candidate["primary_metric"]["value"] == 0.0
        assert candidate["metrics"]["auc"] == 0.0


def test_bundle_embeds_escaped_json_payload(tmp_path: Path) -> None:
    script_path = tmp_path / "script.json"
    script_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "name": "Replay",
                    "metric": "val_auc",
                    "objective": "</script><script>alert(1)</script>",
                },
                "scenes": [],
                "total_cycles": 0,
                "total_candidates": 0,
            }
        )
    )

    output_path = tmp_path / "replay.html"
    bundle(str(script_path), str(output_path))

    html = output_path.read_text()
    prefix = '<script id="script-data" type="application/json">'
    start = html.index(prefix) + len(prefix)
    end = html.index("</script>", start)
    payload = html[start:end]

    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in payload
