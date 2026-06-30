from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lib.runner import (
    init_experiment_dir,
    run_runner_main,
    runner_cli,
)
from lib.runner import (
    save_candidate_result as shared_save,
)
from runners.demo_classification_runner import init_demo

_RUNNER_MODULES = [
    "runners.demo_classification_runner",
    "runners.demo_regression_runner",
    "runners.demo_bootstrap_runner",
    "runners.demo_deep_runner",
]


@pytest.mark.parametrize("runner_module", _RUNNER_MODULES)
def test_creates_expected_files(runner_module: str, tmp_path: Path) -> None:
    if runner_module == "runners.demo_deep_runner" and importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed (needs the deep extra)")
    mod = importlib.import_module(runner_module)
    with patch("lib.runner.ROOT", tmp_path):
        d = mod.init_demo()

    expected = {
        "experiment.md",
        "research_journal.md",
        "research_sources.md",
        "results.json",
    }
    created = {f.name for f in d.iterdir() if f.is_file()}
    assert expected <= created


_BASELINE_PARAMS = [
    (
        "runners.demo_classification_runner",
        "experiments/demo_classification",
        "rule-baseline",
        "val_auc",
    ),
    (
        "runners.demo_regression_runner",
        "experiments/demo_regression",
        "rule-baseline",
        "val_r2",
    ),
    (
        "runners.demo_bootstrap_runner",
        "experiments/demo_bootstrap",
        "majority-baseline",
        "val_auc",
    ),
]


@pytest.mark.parametrize("runner_module,exp_subdir,candidate_id,objective_metric", _BASELINE_PARAMS)
def test_saves_rule_baseline(
    runner_module: str,
    exp_subdir: str,
    candidate_id: str,
    objective_metric: str,
    tmp_path: Path,
) -> None:
    mod = importlib.import_module(runner_module)
    exp_dir = tmp_path / exp_subdir
    with patch("lib.runner.ROOT", tmp_path):
        mod.init_demo()

    shared_save(exp_dir, candidate_id, mod._load_splits, mod.CANDIDATE_RUNNERS)

    results = json.loads((exp_dir / "results.json").read_text())
    assert len(results) == 1
    entry = results[0]
    assert entry["candidate_id"] == candidate_id
    assert entry["objective_metric"] == objective_metric
    assert isinstance(entry["objective_score"], (int, float))


def test_runner_parser_exposes_retired_candidate_commands() -> None:
    parser = runner_cli("demo", ["active"], ["old"])

    list_args = parser.parse_args(["list-retired-candidates"])
    assert list_args.command == "list-retired-candidates"

    run_args = parser.parse_args(
        [
            "run-retired-candidate",
            "--experiment",
            "experiments/demo",
            "--candidate",
            "old",
        ]
    )
    assert run_args.command == "run-retired-candidate"
    assert run_args.candidate == "old"


def test_runner_main_accepts_legacy_generated_runner_call(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["runner", "list-candidates"])

    exit_code = run_runner_main("demo", {"active": lambda _splits: {}}, lambda: None)

    assert exit_code == 0
    assert capsys.readouterr().out == "active\n"


class TestInitDemo:
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


class TestInitExperimentDirWithTemplate:
    """Tests for init_experiment_dir template_path parameter."""

    def test_uses_template_path_for_experiment_md(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Custom Template\nThis is custom.\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp", template_path=template)
        assert (d / "experiment.md").read_text() == "# Custom Template\nThis is custom.\n"

    def test_falls_back_to_generic_template(self, tmp_path: Path) -> None:
        generic = tmp_path / "experiments" / "templates" / "model-search.md"
        generic.parent.mkdir(parents=True)
        generic.write_text("# Generic Template\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp")
        assert (d / "experiment.md").read_text() == "# Generic Template\n"

    def test_uses_research_sources_template_path(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Experiment\n")
        rs_template = tmp_path / "rs_template.md"
        rs_template.write_text("# Custom Research Sources\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir(
                "test-exp",
                template_path=template,
                research_sources_template_path=rs_template,
            )
        assert (d / "research_sources.md").read_text() == "# Custom Research Sources\n"

    def test_creates_all_expected_files(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Experiment\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp", template_path=template)
        expected = {
            "experiment.md",
            "research_journal.md",
            "research_sources.md",
            "results.json",
        }
        created = {f.name for f in d.iterdir() if f.is_file()}
        assert created == expected


class TestInitExperimentDirThreeFolderLayout:
    """Universal three-folder output layout (outputs/, work/, scripts/)."""

    def test_creates_three_directories(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Experiment\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp", template_path=template)
        assert (d / "outputs").is_dir()
        assert (d / "work").is_dir()
        assert (d / "scripts").is_dir()

    def test_creates_gitkeep_in_each(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Experiment\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp", template_path=template)
        for sub in ("outputs", "work", "scripts"):
            assert (d / sub / ".gitkeep").is_file(), f"missing {sub}/.gitkeep"

    def test_idempotent_on_repeated_init(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Experiment\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp", template_path=template)
            # Drop a real file into outputs/ — re-running init must preserve it
            (d / "outputs" / "report.md").write_text("kept")
            init_experiment_dir("test-exp", template_path=template)
        assert (d / "outputs" / "report.md").read_text() == "kept"
        assert (d / "outputs" / ".gitkeep").exists()
        assert (d / "work").is_dir()
        assert (d / "scripts").is_dir()

    def test_force_refresh_preserves_three_folders(self, tmp_path: Path) -> None:
        template = tmp_path / "my_template.md"
        template.write_text("# Experiment\n")
        with patch("lib.runner.ROOT", tmp_path):
            d = init_experiment_dir("test-exp", template_path=template)
            init_experiment_dir("test-exp", template_path=template, force=True)
        for sub in ("outputs", "work", "scripts"):
            assert (d / sub).is_dir()
            assert (d / sub / ".gitkeep").is_file()
