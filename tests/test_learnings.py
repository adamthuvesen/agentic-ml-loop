from __future__ import annotations

from pathlib import Path

from lib.learnings import (
    build_experiment_profile,
    replace_or_append_learnings,
    retrieve_relevant_learnings,
)


def _write_experiment(
    tmp_path: Path,
    name: str,
    *,
    problem_type: str,
    objective_metric: str,
    split_strategy: str,
    data_profile: str,
    extra: str = "",
) -> Path:
    experiment_dir = tmp_path / name
    experiment_dir.mkdir()
    (experiment_dir / "experiment.md").write_text(
        "\n".join(
            [
                "# Model Search Experiment",
                "",
                "## Title",
                "",
                name,
                "",
                "## Goal",
                "",
                "Test retrieval.",
                "",
                "## Data Profile",
                "",
                data_profile,
                "",
                "## Problem Type",
                "",
                problem_type,
                "",
                "## Split Strategy",
                "",
                split_strategy,
                "",
                "## Objective Metric",
                "",
                objective_metric,
                "",
                extra,
            ]
        ).strip()
        + "\n"
    )
    return experiment_dir


class TestLearningsRetrieval:
    def test_builds_profile_tags(self, tmp_path: Path) -> None:
        experiment_dir = _write_experiment(
            tmp_path,
            "classification-exp",
            problem_type="Binary classification",
            objective_metric="Validation AUC (`val_auc`)",
            split_strategy="Temporal split with holdout test.",
            data_profile="- Row count: 53,707\n- Feature count: 80",
            extra="## Known Risks\n\n- Class imbalance is high.\n",
        )

        profile = build_experiment_profile(experiment_dir)
        assert "classification" in profile.tags
        assert "auc" in profile.tags
        assert "temporal-split" in profile.tags
        assert "medium-data" in profile.tags
        assert "medium-feature-set" in profile.tags
        assert "imbalance" in profile.tags

    def test_retrieves_only_relevant_excerpts(self, tmp_path: Path) -> None:
        experiment_dir = _write_experiment(
            tmp_path,
            "ranking-exp",
            problem_type="Open",
            objective_metric="`val_captured_at_20pct` top-20% revenue capture",
            split_strategy="Temporal split by quarter with no holdout.",
            data_profile="- Row count: 4,433\n- Feature count: 11",
        )
        learnings_text = """
# Cross-Experiment Learnings

## Pattern: Classification baseline

Tags: classification, auc, logistic-regression

- Start with logistic regression.

## Pattern: Value ranking

Tags: ranking, value-modeling, temporal-split, top-k

- Binary outcome features often fail to predict magnitude.
- Treat temporal shift as structural.
""".strip()

        retrieved = retrieve_relevant_learnings(
            experiment_dir,
            learnings_text=learnings_text,
        )

        assert [excerpt.title for excerpt in retrieved] == ["Pattern: Value ranking"]

    def test_omits_low_relevance_memory(self, tmp_path: Path) -> None:
        experiment_dir = _write_experiment(
            tmp_path,
            "regression-exp",
            problem_type="Regression",
            objective_metric="Validation R^2",
            split_strategy="Random split.",
            data_profile="- Row count: 1,200\n- Feature count: 8",
        )
        learnings_text = """
# Cross-Experiment Learnings

## Pattern: Classification only

Tags: classification, auc, logistic-regression

- Start with logistic regression.
""".strip()

        assert (
            retrieve_relevant_learnings(
                experiment_dir,
                learnings_text=learnings_text,
            )
            == []
        )


class TestReplaceOrAppendLearnings:
    """Tests for the section-level replace-or-append logic."""

    _HEADER = (
        "# Cross-Experiment Learnings\n\n"
        "Canonical cross-experiment memory for agentic-ml-loop.\n\n"
    )

    def test_appends_when_no_existing_section(self, tmp_path: Path) -> None:
        lf = tmp_path / "learnings.md"
        lf.write_text(self._HEADER + "## Pattern: Some pattern\n\n- A bullet.\n")

        new_section = "\n## From `demo_bootstrap` (2026-04-03)\n\n- New insight.\n"
        replace_or_append_learnings(lf, "demo_bootstrap", new_section)

        content = lf.read_text()
        assert "## Pattern: Some pattern" in content
        assert "## From `demo_bootstrap` (2026-04-03)" in content
        assert "- New insight." in content

    def test_replaces_existing_section(self, tmp_path: Path) -> None:
        lf = tmp_path / "learnings.md"
        lf.write_text(
            self._HEADER
            + "## From `demo_bootstrap` (2026-04-02)\n\n- Old insight.\n\n"
            + "## Pattern: Some pattern\n\n- Unrelated.\n"
        )

        new_section = "\n## From `demo_bootstrap` (2026-04-03)\n\n- New insight.\n"
        replace_or_append_learnings(lf, "demo_bootstrap", new_section)

        content = lf.read_text()
        assert "Old insight" not in content
        assert "2026-04-02" not in content
        assert "- New insight." in content
        assert "## Pattern: Some pattern" in content
        assert "- Unrelated." in content

    def test_preserves_unrelated_sections(self, tmp_path: Path) -> None:
        lf = tmp_path / "learnings.md"
        lf.write_text(
            self._HEADER
            + "## From `demo_classification` (2026-04-01)\n\n- Classification insight.\n\n"
            + "## From `demo_regression` (2026-04-02)\n\n- Regression insight.\n"
        )

        new_section = (
            "\n## From `demo_bootstrap` (2026-04-03)\n\n- Bootstrap insight.\n"
        )
        replace_or_append_learnings(lf, "demo_bootstrap", new_section)

        content = lf.read_text()
        assert "- Classification insight." in content
        assert "- Regression insight." in content
        assert "- Bootstrap insight." in content

    def test_removes_multiple_duplicates(self, tmp_path: Path) -> None:
        lf = tmp_path / "learnings.md"
        lf.write_text(
            self._HEADER
            + "## From `demo_bootstrap` (2026-04-02)\n\n- First dup.\n\n"
            + "## Pattern: Keep me\n\n- Important.\n\n"
            + "## From `demo_bootstrap` (2026-04-03)\n\n- Second dup.\n"
        )

        new_section = "\n## From `demo_bootstrap` (2026-04-04)\n\n- Final version.\n"
        replace_or_append_learnings(lf, "demo_bootstrap", new_section)

        content = lf.read_text()
        assert "First dup" not in content
        assert "Second dup" not in content
        assert "- Final version." in content
        assert "## Pattern: Keep me" in content
        assert "- Important." in content

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        lf = tmp_path / "learnings.md"
        new_section = "\n## From `demo_bootstrap` (2026-04-03)\n\n- Fresh start.\n"
        replace_or_append_learnings(lf, "demo_bootstrap", new_section)

        content = lf.read_text()
        assert "## From `demo_bootstrap`" in content
        assert "- Fresh start." in content
