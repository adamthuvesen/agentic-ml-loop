from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from runners.demo_regression_runner import init_demo


class TestInitRegressionDemo:
    def test_no_extra_artifacts(self, tmp_path: Path) -> None:
        with patch("lib.runner.ROOT", tmp_path):
            d = init_demo()

        names = {f.name for f in d.iterdir()}
        for unwanted in (
            "summary.md",
            "PROGRESS.md",
            "leaderboard.json",
            "feedback.json",
        ):
            assert unwanted not in names
        assert not (d / "results").is_dir()

    def test_force_removes_old_files_and_dirs(self, tmp_path: Path) -> None:
        with patch("lib.runner.ROOT", tmp_path):
            d = init_demo()
            (d / "status.md").write_text("stale\n")
            (d / "loop_state.json").write_text("{}\n")
            (d / "evaluation_review.md").write_text("stale review\n")
            (d / "evaluation_review.json").write_text("{}\n")
            extra_dir = d / "cycles"
            extra_dir.mkdir()
            (extra_dir / "old.txt").write_text("old\n")
            diagnostics_dir = d / "diagnostics"
            diagnostics_dir.mkdir()
            (diagnostics_dir / "summary.md").write_text("old diagnostics\n")

            refreshed = init_demo(force=True)

        assert refreshed == d
        assert not (d / "status.md").exists()
        assert not (d / "loop_state.json").exists()
        assert not (d / "evaluation_review.md").exists()
        assert not (d / "evaluation_review.json").exists()
        assert not extra_dir.exists()
        assert not diagnostics_dir.exists()
