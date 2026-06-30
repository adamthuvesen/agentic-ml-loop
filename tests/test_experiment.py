from __future__ import annotations

from pathlib import Path

import pytest

from experiment import (
    EXPERIMENT_ROOT_ALLOWLIST,
    LOOP_MANAGED_FILES,
    count_journal_cycles,
    diagnostics_dir,
    diagnostics_report_path,
    diagnostics_summary_path,
    evaluation_review_path,
    evaluation_review_report_path,
    get_min_cycles_before_complete,
    get_objective_metric,
    read_diagnostics_summary,
    read_evaluation_review,
    research_sources_template,
    results_file,
    stray_root_entries,
    validate_experiment,
)


@pytest.fixture()
def experiment_dir(tmp_path: Path) -> Path:
    """Create a valid experiment directory with required files."""
    d = tmp_path / "test-exp"
    d.mkdir()
    (d / "experiment.md").write_text("# Experiment\n")
    (d / "research_journal.md").write_text("# Journal\n")
    (d / "results.json").write_text(
        '[{"candidate_id": "a", "objective_score": 0.5, "objective_metric": "val_auc"}]\n'
    )
    return d


class TestResearchSourcesTemplate:
    def test_research_sources_template_describes_living_synthesis(self) -> None:
        template = research_sources_template("demo")
        normalized = " ".join(template.split())
        assert "living document" in template
        assert "Reusable Takeaways" in template
        assert "current summary for future cycles" in normalized
        assert "contradictions unresolved" in normalized
        assert "provenance, not final truth" in normalized


class TestMinCyclesContract:
    def test_get_min_cycles_before_complete(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("Minimum loop cycles before EXPERIMENT_COMPLETE: 7\n")
        assert get_min_cycles_before_complete(d) == 7

    def test_get_min_cycles_missing_returns_none(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# No contract\n")
        assert get_min_cycles_before_complete(d) is None

    def test_count_journal_cycles(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "research_journal.md").write_text("# J\n\n## Cycle 0001: A\n\n## Cycle 0002: B\n")
        assert count_journal_cycles(d) == 2


class TestValidateExperiment:
    def test_passes_on_valid(self, experiment_dir: Path) -> None:
        errors = validate_experiment(experiment_dir)
        assert errors == []

    def test_fails_missing_experiment_md(self, experiment_dir: Path) -> None:
        (experiment_dir / "experiment.md").unlink()
        errors = validate_experiment(experiment_dir)
        assert any("experiment.md" in e for e in errors)

    def test_fails_invalid_json(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text("{bad json")
        errors = validate_experiment(experiment_dir)
        assert any("invalid JSON" in e for e in errors)

    def test_fails_results_not_list(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text('{"not": "a list"}\n')
        errors = validate_experiment(experiment_dir)
        assert any("must be a JSON list" in e for e in errors)

    def test_strict_fails_empty_results(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text("[]\n")
        errors = validate_experiment(experiment_dir, strict_completion=True)
        assert any("strict completion" in e for e in errors)

    def test_strict_passes_with_results(self, experiment_dir: Path) -> None:
        errors = validate_experiment(experiment_dir, strict_completion=True)
        assert errors == []

    def test_fails_result_entry_missing_candidate_id(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text(
            '[{"objective_score": 0.5, "objective_metric": "val_auc"}]\n'
        )
        errors = validate_experiment(experiment_dir)
        assert any("candidate_id" in e for e in errors)

    def test_fails_result_entry_missing_objective_score(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text(
            '[{"candidate_id": "a", "objective_metric": "val_auc"}]\n'
        )
        errors = validate_experiment(experiment_dir)
        assert any("objective_score" in e for e in errors)

    def test_strict_fails_result_entry_missing_objective_metric(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text(
            '[{"candidate_id": "a", "objective_score": 0.5}]\n'
        )
        errors = validate_experiment(experiment_dir, strict_completion=True)
        assert any("objective_metric" in e for e in errors)

    def test_fails_non_finite_objective_score(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text(
            '[{"candidate_id": "a", "objective_score": NaN, "objective_metric": "val_auc"}]\n'
        )
        errors = validate_experiment(experiment_dir)
        assert any("finite numeric objective_score" in e for e in errors)

    def test_strict_requires_at_least_one_valid_entry(self, experiment_dir: Path) -> None:
        (experiment_dir / "results.json").write_text('[{"foo": 1}]\n')
        errors = validate_experiment(experiment_dir, strict_completion=True)
        assert any("at least one valid result entry" in e for e in errors)


class TestResultsFile:
    def test_returns_results_json(self, experiment_dir: Path) -> None:
        assert results_file(experiment_dir) == experiment_dir / "results.json"


class TestDiagnosticsPaths:
    def test_returns_diagnostics_paths(self, experiment_dir: Path) -> None:
        assert diagnostics_dir(experiment_dir) == experiment_dir / "diagnostics"
        assert diagnostics_summary_path(experiment_dir) == (
            experiment_dir / "diagnostics" / "summary.md"
        )
        assert diagnostics_report_path(experiment_dir) == (
            experiment_dir / "diagnostics" / "report.json"
        )

    def test_read_diagnostics_summary_returns_none_when_missing(self, experiment_dir: Path) -> None:
        assert read_diagnostics_summary(experiment_dir) is None


class TestEvaluationReviewPaths:
    def test_returns_evaluation_review_paths(self, experiment_dir: Path) -> None:
        assert evaluation_review_path(experiment_dir) == (experiment_dir / "evaluation_review.md")
        assert evaluation_review_report_path(experiment_dir) == (
            experiment_dir / "evaluation_review.json"
        )

    def test_read_evaluation_review_returns_none_when_missing(self, experiment_dir: Path) -> None:
        assert read_evaluation_review(experiment_dir) is None


class TestGetObjectiveMetric:
    def test_extracts_parenthesized_identifier(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text(
            "# Experiment\n\n## Objective Metric\n\nValidation AUC (`val_auc`). Secondary: log loss.\n"
        )
        assert get_objective_metric(d) == "val_auc"

    def test_extracts_from_parens_format(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text(
            "# Experiment\n\n## Objective Metric\n\nValidation R² (val_r2).\n"
        )
        assert get_objective_metric(d) == "val_r2"

    def test_returns_none_when_no_section(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Experiment\n\n## Goal\n\nSome goal.\n")
        assert get_objective_metric(d) is None

    def test_returns_none_when_no_parenthesized_token(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Experiment\n\n## Objective Metric\n\nValidation AUC.\n")
        assert get_objective_metric(d) is None

    def test_returns_none_when_no_experiment_md(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        assert get_objective_metric(d) is None


class TestMetricConsistencyValidation:
    @pytest.fixture()
    def valid_exp(self, tmp_path: Path) -> Path:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text(
            "# Experiment\n\n## Objective Metric\n\nValidation AUC (val_auc).\n"
        )
        (d / "research_journal.md").write_text("# Journal\n")
        return d

    def test_no_warnings_when_all_match(self, valid_exp: Path) -> None:
        import json

        (valid_exp / "results.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "a",
                        "objective_score": 0.8,
                        "objective_metric": "val_auc",
                    }
                ]
            )
        )
        msgs = validate_experiment(valid_exp)
        assert not any("warning" in m for m in msgs)

    def test_warns_on_metric_mismatch(self, valid_exp: Path) -> None:
        import json

        (valid_exp / "results.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "bad",
                        "objective_score": 0.7,
                        "objective_metric": "test_auc",
                    }
                ]
            )
        )
        msgs = validate_experiment(valid_exp)
        warnings = [m for m in msgs if m.startswith("warning:")]
        assert len(warnings) == 1
        assert "bad" in warnings[0]
        assert "test_auc" in warnings[0]

    def test_strict_mode_upgrades_to_error(self, valid_exp: Path) -> None:
        import json

        (valid_exp / "results.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "bad",
                        "objective_score": 0.7,
                        "objective_metric": "test_auc",
                    }
                ]
            )
        )
        msgs = validate_experiment(valid_exp, strict_completion=True)
        errors = [m for m in msgs if not m.startswith("warning:")]
        assert any("bad" in e and "test_auc" in e for e in errors)

    def test_skips_check_when_no_declared_metric(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Experiment\n\n## Goal\n\nSomething.\n")
        (d / "research_journal.md").write_text("# Journal\n")
        import json

        (d / "results.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "a",
                        "objective_score": 0.5,
                        "objective_metric": "test_auc",
                    }
                ]
            )
        )
        msgs = validate_experiment(d)
        assert not any("warning" in m for m in msgs)


class TestPrefixConventionWarning:
    def test_no_warning_for_val_prefix(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text(
            "# Experiment\n\n## Objective Metric\n\nValidation AUC (val_auc).\n"
        )
        (d / "research_journal.md").write_text("# Journal\n")
        import json

        (d / "results.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "a",
                        "objective_score": 0.8,
                        "objective_metric": "val_auc",
                    }
                ]
            )
        )
        msgs = validate_experiment(d)
        assert not any("prefix" in m for m in msgs)

    def test_warns_for_test_prefix(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text(
            "# Experiment\n\n## Objective Metric\n\nTest AUC (test_auc).\n"
        )
        (d / "research_journal.md").write_text("# Journal\n")
        import json

        (d / "results.json").write_text(
            json.dumps(
                [
                    {
                        "candidate_id": "a",
                        "objective_score": 0.8,
                        "objective_metric": "test_auc",
                    }
                ]
            )
        )
        msgs = validate_experiment(d)
        warnings = [m for m in msgs if "prefix" in m.lower() or "convention" in m.lower()]
        assert len(warnings) >= 1


class TestStrayRootFilesWarning:
    """validate_experiment() warns when files outside the allowlist sit at root."""

    @pytest.fixture()
    def clean_exp(self, tmp_path: Path) -> Path:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Experiment\n")
        (d / "research_journal.md").write_text("# Journal\n")
        (d / "results.json").write_text("[]")
        return d

    def test_clean_root_passes_silently(self, clean_exp: Path) -> None:
        msgs = validate_experiment(clean_exp)
        stray_warnings = [m for m in msgs if "stray" in m.lower()]
        assert stray_warnings == []

    def test_allowlisted_loop_files_pass_silently(self, clean_exp: Path) -> None:
        # All loop-managed files should not trigger the warning
        (clean_exp / "research_sources.md").write_text("# Sources\n")
        (clean_exp / "status.md").write_text("status")
        (clean_exp / "loop_state.json").write_text("{}")
        (clean_exp / ".loop.lock").write_text("")
        (clean_exp / "results.json.lock").write_text("")
        (clean_exp / "evaluation_review.md").write_text("")
        (clean_exp / "feedback.json").write_text("{}")
        (clean_exp / "notebook.yaml").write_text("title: Portable notebook\n")
        (clean_exp / "cycles").mkdir()
        (clean_exp / "outputs").mkdir()
        (clean_exp / "work").mkdir()
        (clean_exp / "scripts").mkdir()
        msgs = validate_experiment(clean_exp)
        stray_warnings = [m for m in msgs if "stray" in m.lower()]
        assert stray_warnings == []

    def test_stray_csv_at_root_emits_warning(self, clean_exp: Path) -> None:
        (clean_exp / "cluster_map.csv").write_text("a,b\n1,2\n")
        msgs = validate_experiment(clean_exp)
        stray_warnings = [m for m in msgs if m.startswith("warning:") and "stray" in m]
        assert len(stray_warnings) == 1
        assert "cluster_map.csv" in stray_warnings[0]
        assert "outputs/" in stray_warnings[0] or "work/" in stray_warnings[0]

    def test_multiple_strays_listed(self, clean_exp: Path) -> None:
        (clean_exp / "feature_ranks.csv").write_text("")
        (clean_exp / "summary.json").write_text("{}")
        (clean_exp / "cycle03_eda.py").write_text("")
        msgs = validate_experiment(clean_exp)
        stray_warnings = [m for m in msgs if m.startswith("warning:") and "stray" in m]
        assert len(stray_warnings) == 1
        warning = stray_warnings[0]
        for name in ("feature_ranks.csv", "summary.json", "cycle03_eda.py"):
            assert name in warning


def test_loop_managed_files_are_allowlisted() -> None:
    assert LOOP_MANAGED_FILES <= EXPERIMENT_ROOT_ALLOWLIST


def test_stray_warning_is_non_blocking(tmp_path: Path) -> None:
    d = tmp_path / "exp"
    d.mkdir()
    (d / "experiment.md").write_text("# Experiment\n")
    (d / "research_journal.md").write_text("# Journal\n")
    (d / "results.json").write_text("[]")
    (d / "stray.csv").write_text("")

    msgs = validate_experiment(d, strict_completion=False)

    errors = [m for m in msgs if not m.startswith("warning:")]
    assert errors == []


def test_dotfiles_other_than_loop_lock_ignored(tmp_path: Path) -> None:
    d = tmp_path / "exp"
    d.mkdir()
    (d / "experiment.md").write_text("# Experiment\n")
    (d / "research_journal.md").write_text("# Journal\n")
    (d / "results.json").write_text("[]")
    (d / "outputs").mkdir()
    (d / ".DS_Store").write_text("")
    (d / ".cache").mkdir()

    msgs = validate_experiment(d)

    stray_warnings = [m for m in msgs if "stray" in m.lower()]
    assert stray_warnings == []


class TestStrayRootEntriesHelper:
    def test_returns_empty_for_clean_root(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("")
        (d / "outputs").mkdir()
        assert stray_root_entries(d) == []

    def test_returns_sorted_names(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("")
        (d / "z.csv").write_text("")
        (d / "a.csv").write_text("")
        assert stray_root_entries(d) == ["a.csv", "z.csv"]
