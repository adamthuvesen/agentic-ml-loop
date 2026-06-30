from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.diagnostics import get_diagnostics_observations
from lib.evaluation_review import get_evaluation_observations
from lib.observations import dedup_observations
from lib.signals import (
    _count_external_sources,
    advisory_signals,
    research_signals,
)


def _make_experiment(
    tmp_path: Path,
    *,
    results: list | None = None,
    journal: str = "",
) -> Path:
    d = tmp_path / "exp"
    d.mkdir(exist_ok=True)
    (d / "experiment.md").write_text("# Experiment\n")
    (d / "research_journal.md").write_text(journal or "# Journal\n")
    (d / "research_sources.md").write_text("# Research Sources\n")
    (d / "results.json").write_text(json.dumps(results if results is not None else []) + "\n")
    return d


class TestGetDiagnosticsObservations:
    def test_returns_empty_when_no_report(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        assert get_diagnostics_observations(d) == []

    def test_extracts_split_drift(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        report = {
            "split_comparison": [
                {
                    "split": "validation",
                    "top_feature_drift": [
                        {"column": "age", "delta": 2.5, "scaled_delta": 1.5},
                    ],
                }
            ],
            "missingness": [],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(report))
        obs = get_diagnostics_observations(d)
        assert len(obs) >= 1
        assert obs[0][0] == 85
        assert "split drift" in obs[0][1].lower()
        assert "age" in obs[0][1]

    def test_extracts_high_missingness(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        report = {
            "split_comparison": [],
            "missingness": [{"column": "income", "missing_rate": 0.25}],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(report))
        obs = get_diagnostics_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 70
        assert "income" in obs[0][1]

    def test_skips_low_missingness(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        report = {
            "split_comparison": [],
            "missingness": [{"column": "age", "missing_rate": 0.05}],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(report))
        obs = get_diagnostics_observations(d)
        assert len(obs) == 0

    def test_extracts_subgroup_variance(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        report = {
            "split_comparison": [],
            "missingness": [],
            "subgroup_slices": [
                {
                    "feature": "gender",
                    "low_group": "M",
                    "low_value": 0.3,
                    "high_group": "F",
                    "high_value": 0.7,
                    "gap": 0.4,
                }
            ],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(report))
        obs = get_diagnostics_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 60

    def test_extracts_interaction_candidates(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        report = {
            "split_comparison": [],
            "missingness": [],
            "subgroup_slices": [],
            "interaction_candidates": [
                {
                    "feature_a": "age",
                    "feature_b": "income",
                    "interaction_corr": 0.6,
                    "lift": 0.15,
                }
            ],
        }
        (diag_dir / "report.json").write_text(json.dumps(report))
        obs = get_diagnostics_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 50

    def test_corrupt_json_is_reported_loudly(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        (diag_dir / "report.json").write_text("{bad json")
        with pytest.warns(UserWarning, match="could not be read"):
            obs = get_diagnostics_observations(d)
        assert len(obs) == 1
        priority, message = obs[0]
        assert priority == 95
        assert "could not be read" in message
        assert "report.json" in message


class TestGetEvaluationObservations:
    def test_returns_empty_when_no_review(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        assert get_evaluation_observations(d) == []

    def test_extracts_split_concern(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        review = {
            "concerns": [
                {
                    "kind": "split",
                    "title": "Split Reliability Concern",
                    "concern": "The current split may be unstable.",
                    "priority": 91.0,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        obs = get_evaluation_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 90
        assert "split" in obs[0][1].lower()

    def test_extracts_leakage_concern(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        review = {
            "concerns": [
                {
                    "kind": "leakage",
                    "title": "Leakage Marker Concern",
                    "concern": "Row overlap detected.",
                    "priority": 96.0,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        obs = get_evaluation_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 95

    def test_extracts_instability_concern(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        review = {
            "concerns": [
                {
                    "kind": "instability",
                    "title": "Leaderboard Instability Concern",
                    "concern": "Noisy leaderboard.",
                    "priority": 80.5,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        obs = get_evaluation_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 80

    def test_extracts_metric_concern(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        review = {
            "concerns": [
                {
                    "kind": "metric",
                    "title": "Metric Fit Concern",
                    "concern": "AUC may not reflect deployment quality.",
                    "priority": 70.5,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        obs = get_evaluation_observations(d)
        assert len(obs) == 1
        assert obs[0][0] == 75

    def test_handles_multiple_concerns(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        review = {
            "concerns": [
                {
                    "kind": "split",
                    "title": "Split Concern",
                    "concern": "Bad split.",
                    "priority": 91,
                },
                {
                    "kind": "leakage",
                    "title": "Leakage Concern",
                    "concern": "Overlap.",
                    "priority": 96,
                },
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        obs = get_evaluation_observations(d)
        assert len(obs) == 2
        # Sorted by priority descending
        assert obs[0][0] == 95
        assert obs[1][0] == 90

    def test_handles_corrupt_json(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "evaluation_review.json").write_text("not json")
        assert get_evaluation_observations(d) == []


class TestBuildAdvisorySignals:
    def test_returns_empty_when_no_sources(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        assert advisory_signals(d) == []

    def test_returns_only_research_signals_when_no_other_sources(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {
                    "candidate_id": "a",
                    "objective_score": 0.701,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.68, 0.72]},
                },
                {
                    "candidate_id": "b",
                    "objective_score": 0.695,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.67, 0.71]},
                },
            ],
            journal="# Journal\n\n## Cycle 0001: compare\n\nDone.\n",
        )
        signals = advisory_signals(d)
        assert len(signals) >= 1
        assert any("plateau" in s.lower() or "clustered" in s.lower() for _, s in signals)

    def test_merges_all_sources(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {"candidate_id": "a", "objective_score": 0.7},
                {"candidate_id": "b", "objective_score": 0.72},
                {"candidate_id": "c", "objective_score": 0.75},
            ],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )

        # Add diagnostics report
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        diag_report = {
            "split_comparison": [
                {
                    "split": "validation",
                    "top_feature_drift": [
                        {"column": "age", "delta": 2.0, "scaled_delta": 1.5},
                    ],
                }
            ],
            "missingness": [{"column": "income", "missing_rate": 0.30}],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(diag_report))

        # Add evaluation review
        eval_review = {
            "concerns": [
                {
                    "kind": "instability",
                    "title": "Leaderboard Instability",
                    "concern": "Noisy leaderboard.",
                    "priority": 80.5,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(eval_review))

        signals = advisory_signals(d)
        assert len(signals) >= 3
        texts = [s for _, s in signals]
        # Should contain observations from all three sources
        assert any("age" in t for t in texts)  # diagnostics
        assert any("instability" in t.lower() or "noisy" in t.lower() for t in texts)  # eval review
        assert any(
            "error analysis" in t.lower() or "diagnostics" in t.lower() for t in texts
        )  # research signal (missing analysis)

    def test_caps_at_eight(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {"candidate_id": "a", "objective_score": 0.7},
                {"candidate_id": "b", "objective_score": 0.72},
                {"candidate_id": "c", "objective_score": 0.75},
            ],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )

        # Add diagnostics with many findings
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        diag_report = {
            "split_comparison": [
                {
                    "split": "validation",
                    "top_feature_drift": [
                        {"column": f"feat_{i}", "delta": 2.0, "scaled_delta": 1.5 + i}
                        for i in range(5)
                    ],
                }
            ],
            "missingness": [
                {"column": f"col_{i}", "missing_rate": 0.2 + i * 0.1} for i in range(5)
            ],
            "subgroup_slices": [
                {
                    "feature": f"cat_{i}",
                    "low_group": "A",
                    "low_value": 0.1,
                    "high_group": "B",
                    "high_value": 0.9,
                    "gap": 0.8,
                }
                for i in range(3)
            ],
            "interaction_candidates": [
                {
                    "feature_a": f"a_{i}",
                    "feature_b": f"b_{i}",
                    "interaction_corr": 0.5,
                    "lift": 0.1 + i * 0.05,
                }
                for i in range(3)
            ],
        }
        (diag_dir / "report.json").write_text(json.dumps(diag_report))

        signals = advisory_signals(d)
        assert len(signals) <= 8

    def test_dedup_split_drift_overlap(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)

        # Diagnostics flags split drift
        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        diag_report = {
            "split_comparison": [
                {
                    "split": "validation",
                    "top_feature_drift": [
                        {"column": "age", "delta": 2.0, "scaled_delta": 1.5},
                    ],
                }
            ],
            "missingness": [],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(diag_report))

        # Evaluation review also flags split concern
        eval_review = {
            "concerns": [
                {
                    "kind": "split",
                    "title": "Split Reliability Concern",
                    "concern": "The current split may be unstable due to drift.",
                    "priority": 91.0,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(eval_review))

        signals = advisory_signals(d)
        # Both flag split-related issues; dedup should keep only one
        split_signals = [s for _, s in signals if "split" in s.lower()]
        assert len(split_signals) == 1

    def test_sorted_by_priority_descending(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {"candidate_id": "a", "objective_score": 0.7},
                {"candidate_id": "b", "objective_score": 0.72},
                {"candidate_id": "c", "objective_score": 0.75},
            ],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )

        diag_dir = d / "diagnostics"
        diag_dir.mkdir()
        diag_report = {
            "split_comparison": [],
            "missingness": [{"column": "income", "missing_rate": 0.30}],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diag_dir / "report.json").write_text(json.dumps(diag_report))

        signals = advisory_signals(d)
        priorities = [p for p, _ in signals]
        assert priorities == sorted(priorities, reverse=True)


class TestDedupObservations:
    def test_keeps_higher_priority_on_split_drift(self) -> None:
        observations = [
            (85, "Split drift detected: feature age shifted by 1.5 std"),
            (
                90,
                "Split Reliability Concern: The current split may be unstable due to drift.",
            ),
        ]
        result = dedup_observations(observations)
        assert len(result) == 1
        assert result[0][0] == 90

    def test_keeps_all_when_no_overlap(self) -> None:
        observations = [
            (85, "Split drift detected: feature age shifted by 1.5 std"),
            (70, "High missingness: income missing in 25% of rows."),
        ]
        result = dedup_observations(observations)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        assert dedup_observations([]) == []


class TestBuildResearchSignalsBackwardCompat:
    """Ensure research_signals still returns list[str] (no priority scores)."""

    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {
                    "candidate_id": "a",
                    "objective_score": 0.701,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.68, 0.72]},
                },
                {
                    "candidate_id": "b",
                    "objective_score": 0.695,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.67, 0.71]},
                },
            ],
            journal="# Journal\n\n## Cycle 0001: compare\n\nDone.\n",
        )
        signals = research_signals(d)
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, str)

    def test_research_signal_keyword_overrides_still_work(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)

        signals = research_signals(
            d,
            results=[
                {
                    "candidate_id": "a",
                    "objective_score": 0.701,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.68, 0.72]},
                },
                {
                    "candidate_id": "b",
                    "objective_score": 0.695,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.67, 0.71]},
                },
            ],
            journal_cycles=1,
            external_sources=2,
            has_error_analysis=True,
            has_diagnostics=True,
            limit=1,
        )

        assert len(signals) == 1
        assert "clustered" in signals[0].lower() or "plateau" in signals[0].lower()

    def test_advisory_signal_keyword_overrides_still_work(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)

        signals = advisory_signals(
            d,
            results=[
                {"candidate_id": "a", "objective_score": 0.701},
                {"candidate_id": "b", "objective_score": 0.695},
            ],
            journal_cycles=1,
            external_sources=2,
            has_error_analysis=True,
            has_diagnostics=True,
        )

        assert all(
            isinstance(priority, int) and isinstance(text, str) for priority, text in signals
        )


class TestCountExternalSources:
    def test_placeholder_untouched_returns_zero(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: <title>\n\n- **Type:** kaggle\n"
        )
        assert _count_external_sources(d) == 0

    def test_overwritten_source_001_returns_one(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: Kaggle top solution\n\n- real content\n"
        )
        assert _count_external_sources(d) == 1

    def test_missing_file_returns_zero(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "research_sources.md").unlink()
        assert _count_external_sources(d) == 0

    def test_placeholder_plus_real_sources(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: <title>\n\n- placeholder\n\n"
            "### Source 002: Real research\n\n- real\n"
        )
        assert _count_external_sources(d) == 1
