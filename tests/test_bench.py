from __future__ import annotations

import csv
import json
from pathlib import Path

from loop.bench import (
    BenchmarkBudget,
    BenchmarkRequest,
    RunnerOutcome,
    aggregate_runner_outcome,
    run_benchmark,
    scaffold_runner_experiment,
)
from loop.invoke import RunnerConfig
from loop.loop_state import LoopState, write_state_file
from tests.loop.conftest import make_experiment_dir

_BUDGET = BenchmarkBudget(max_cycles=3, max_hours=None)


def _finish_experiment(
    experiment_dir: Path, *, score: float, referee: int, leakage_clean: bool = True
) -> None:
    """Simulate a completed loop run by writing state, results, and a scorecard."""
    state = LoopState.from_dict(
        {"experiment_id": experiment_dir.name, "status": "completed", "cycle_count": 3}
    )
    write_state_file(experiment_dir, state)
    (experiment_dir / "results.json").write_text(
        json.dumps(
            [{"candidate_id": "cand-a", "objective_score": score, "objective_metric": "val_auc"}]
        )
    )
    card_dir = experiment_dir / "cycles" / "0001"
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "scorecard.json").write_text(
        json.dumps(
            {
                "cycle_id": "0001",
                "overall": referee,
                "grade": "A" if referee >= 85 else "C",
                "criteria": [
                    {
                        "name": "leakage_split_clean",
                        "score": 1.0 if leakage_clean else 0.0,
                        "note": "",
                    }
                ],
                "signals": [],
            }
        )
    )


def test_scaffold_creates_fresh_experiment(tmp_path: Path) -> None:
    source = make_experiment_dir(tmp_path, journal="## Cycle 0001: real work\n")
    dest = tmp_path / "bench" / "claude" / source.name
    scaffold_runner_experiment(source, dest)

    assert (dest / "experiment.md").exists()
    assert (dest / "research_journal.md").exists()
    assert (dest / "research_sources.md").exists()
    assert json.loads((dest / "results.json").read_text()) == []
    # Fresh journal, not the source's real work.
    assert "real work" not in (dest / "research_journal.md").read_text()


def test_aggregate_reads_state_results_and_scorecards(tmp_path: Path) -> None:
    d = make_experiment_dir(tmp_path)
    _finish_experiment(d, score=0.72, referee=90)

    outcome = aggregate_runner_outcome("claude", d, elapsed_seconds=1.5)
    assert outcome.cycles == 3
    assert outcome.status == "completed"
    assert outcome.best_score == 0.72
    assert outcome.best_candidate == "cand-a"
    assert outcome.mean_referee == 90.0
    assert outcome.last_referee == 90
    assert outcome.leakage_clean is True
    assert outcome.elapsed_seconds == 1.5


def test_run_benchmark_ranks_by_referee_then_score(tmp_path: Path) -> None:
    source = make_experiment_dir(tmp_path, journal="## Cycle 0001: x\n")
    bench_dir = tmp_path / "bench_out"

    quality = {"claude": (0.80, 95), "codex": (0.90, 60)}

    def fake_run_one(experiment_dir: Path, config: RunnerConfig, budget: BenchmarkBudget) -> None:
        score, referee = quality[config.name]
        _finish_experiment(experiment_dir, score=score, referee=referee)

    report = run_benchmark(
        BenchmarkRequest(
            source_experiment_dir=source,
            runner_names=["claude", "codex"],
            budget=_BUDGET,
            bench_dir=bench_dir,
            timestamp="2026-01-01T00:00:00Z",
            run_one=fake_run_one,
            runner_config_for=lambda name: RunnerConfig(name=name, command=[name]),
        )
    )

    ranked = report.ranked()
    # claude has worse score but better referee conduct -> ranked first.
    assert [o.runner for o in ranked] == ["claude", "codex"]
    assert (bench_dir / "comparison.md").exists()
    assert (bench_dir / "comparison.csv").exists()

    rows = list(csv.DictReader((bench_dir / "comparison.csv").open()))
    assert rows[0]["runner"] == "claude"
    assert rows[0]["rank"] == "1"


def test_failing_runner_is_recorded_and_does_not_abort_others(tmp_path: Path) -> None:
    source = make_experiment_dir(tmp_path, journal="## Cycle 0001: x\n")
    bench_dir = tmp_path / "bench_out"

    def fake_run_one(experiment_dir: Path, config: RunnerConfig, budget: BenchmarkBudget) -> None:
        if config.name == "codex":
            raise RuntimeError("runner exploded")
        _finish_experiment(experiment_dir, score=0.7, referee=88)

    report = run_benchmark(
        BenchmarkRequest(
            source_experiment_dir=source,
            runner_names=["claude", "codex"],
            budget=_BUDGET,
            bench_dir=bench_dir,
            timestamp="2026-01-01T00:00:00Z",
            run_one=fake_run_one,
            runner_config_for=lambda name: RunnerConfig(name=name, command=[name]),
        )
    )

    by_runner = {o.runner: o for o in report.outcomes}
    assert by_runner["claude"].error is None
    assert by_runner["codex"].error is not None
    assert "runner exploded" in by_runner["codex"].error
    # Errored runner sorts last.
    assert report.ranked()[-1].runner == "codex"
    assert "error: RuntimeError" in (bench_dir / "comparison.md").read_text()


def test_markdown_flags_leakage(tmp_path: Path) -> None:
    report_outcome = RunnerOutcome(
        runner="claude",
        experiment_dir=tmp_path,
        cycles=2,
        status="completed",
        best_score=0.5,
        mean_referee=70.0,
        last_referee=70,
        leakage_clean=False,
    )
    from loop.bench import BenchmarkReport

    report = BenchmarkReport("exp", "2026-01-01T00:00:00Z", _BUDGET, [report_outcome])
    md = report.to_markdown()
    assert "| NO |" in md  # leakage-not-clean is surfaced loudly
