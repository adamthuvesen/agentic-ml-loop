from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.runner import save_candidate_result

pytest.importorskip("torch")

from runners.demo_deep_runner import init_demo  # noqa: E402

_HAS_TORCH = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed (needs the deep extra)")
class TestDemoDeepRunner:
    def test_init_demo_creates_expected_files(self, tmp_path: Path) -> None:
        with patch("lib.runner.ROOT", tmp_path):
            experiment_dir = init_demo()

        expected = {
            "experiment.md",
            "research_journal.md",
            "research_sources.md",
            "results.json",
        }
        created = {path.name for path in experiment_dir.iterdir() if path.is_file()}
        assert expected <= created

    def test_logreg_baseline_runs(self, tmp_path: Path) -> None:
        import runners.demo_deep_runner as runner_mod

        with patch("lib.runner.ROOT", tmp_path):
            init_demo()
            experiment_dir = tmp_path / "experiments" / "demo_deep"
            payload = save_candidate_result(
                experiment_dir,
                "logreg-baseline",
                runner_mod._load_splits,
                runner_mod.CANDIDATE_RUNNERS,
            )

        assert payload["candidate_id"] == "logreg-baseline"
        assert payload["objective_metric"] == "val_auc"
        results = json.loads((experiment_dir / "results.json").read_text())
        assert len(results) == 1

    def test_mlp_deep_beats_logreg_on_validation(self, tmp_path: Path) -> None:
        import runners.demo_deep_runner as runner_mod

        with patch("lib.runner.ROOT", tmp_path):
            init_demo()
            experiment_dir = tmp_path / "experiments" / "demo_deep"
            logreg = save_candidate_result(
                experiment_dir,
                "logreg-baseline",
                runner_mod._load_splits,
                runner_mod.CANDIDATE_RUNNERS,
            )
            mlp = save_candidate_result(
                experiment_dir,
                "mlp-deep",
                runner_mod._load_splits,
                runner_mod.CANDIDATE_RUNNERS,
            )

        assert mlp["objective_score"] > logreg["objective_score"]
