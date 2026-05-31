from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from lib.learnings import LEARNINGS_EXTRACTION_PROMPT, extract_and_append_learnings


class TestExtractAndAppendLearnings:
    def test_appends_to_canonical_learnings_with_profile_tags(
        self, tmp_path: Path
    ) -> None:
        experiment_dir = tmp_path / "experiments" / "demo"
        experiment_dir.mkdir(parents=True)
        (experiment_dir / "experiment.md").write_text(
            "\n".join(
                [
                    "# Model Search Experiment",
                    "",
                    "## Problem Type",
                    "",
                    "Binary classification",
                    "",
                    "## Split Strategy",
                    "",
                    "Temporal split with holdout test.",
                    "",
                    "## Objective Metric",
                    "",
                    "Validation AUC (`val_auc`)",
                    "",
                    "## Data Profile",
                    "",
                    "- Row count: 53,707",
                    "- Feature count: 80",
                    "",
                ]
            )
            + "\n"
        )
        (experiment_dir / "research_journal.md").write_text(
            "# Journal\n\nUseful notes.\n"
        )

        result = subprocess.CompletedProcess(
            args=["claude", "--print"],
            returncode=0,
            stdout="- Always compare simple baselines first.\n",
            stderr="",
        )

        with (
            patch("lib.learnings.subprocess.run", return_value=result),
            patch("lib.learnings.ROOT", tmp_path),
            patch("experiment.ROOT", tmp_path),
        ):
            assert extract_and_append_learnings(experiment_dir) is True

        payload = (tmp_path / "learnings.md").read_text()
        assert "## From `demo`" in payload
        assert "Tags:" in payload
        assert "classification" in payload
        assert "auc" in payload
        assert "temporal-split" in payload

    def test_prompt_construction_safe_with_braces_in_journal(self) -> None:
        """Journal text containing Python dicts / JSON must not crash prompt build."""
        brace_journal = (
            "# Journal\n\n"
            "Tried param_grid = {'max_depth': 3, 'n_estimators': [100, 200]}\n"
            'Also tested {"learning_rate": 0.05} with {nested: {keys}}.\n'
        )
        # Should not raise KeyError, ValueError, or IndexError
        result = LEARNINGS_EXTRACTION_PROMPT.replace(
            "{experiment_id}", "test-experiment"
        ).replace("{journal}", brace_journal)
        assert "test-experiment" in result
        assert "{'max_depth': 3" in result
