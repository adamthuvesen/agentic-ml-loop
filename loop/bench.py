"""Runner benchmark: run one experiment spec across multiple agent runners.

Each runner gets an isolated copy of the experiment (the experiment.md spec plus
fresh journal/sources/results), runs the loop to a shared cycle/time budget with
the advisory referee on, and the outcomes are aggregated into a ranked
comparison (Markdown + CSV) ordered by referee score, then validation score.

The loop invocation is injectable (``run_one``) so the aggregation and report
logic can be tested without launching real agent subprocesses.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from experiment import research_journal_template, research_sources_template
from lib.analysis import ranked_results
from lib.io import load_json, write_json, write_text

from .invoke import RunnerConfig, runner_config_from_args
from .loop_state import load_state

RunOne = Callable[[Path, RunnerConfig, "BenchmarkBudget"], None]


@dataclass(frozen=True)
class BenchmarkBudget:
    max_cycles: int | None
    max_hours: float | None


@dataclass
class RunnerOutcome:
    """Aggregated result of running one runner over the shared spec."""

    runner: str
    experiment_dir: Path
    cycles: int = 0
    status: str = "unknown"
    best_score: float | None = None
    best_candidate: str | None = None
    mean_referee: float | None = None
    last_referee: int | None = None
    leakage_clean: bool = True
    elapsed_seconds: float = 0.0
    error: str | None = None

    def rank_key(self) -> tuple[float, float]:
        """Sort key: referee conduct first, then leaderboard quality."""
        return (
            self.mean_referee if self.mean_referee is not None else -1.0,
            self.best_score if self.best_score is not None else float("-inf"),
        )


@dataclass
class BenchmarkReport:
    experiment_id: str
    timestamp: str
    budget: BenchmarkBudget
    outcomes: list[RunnerOutcome] = field(default_factory=list)

    def ranked(self) -> list[RunnerOutcome]:
        """Outcomes best-first; errored runs sort last."""
        ok = [o for o in self.outcomes if o.error is None]
        failed = [o for o in self.outcomes if o.error is not None]
        ok.sort(key=lambda o: o.rank_key(), reverse=True)
        return ok + failed

    def to_markdown(self) -> str:
        budget_bits = []
        if self.budget.max_cycles is not None:
            budget_bits.append(f"max_cycles={self.budget.max_cycles}")
        if self.budget.max_hours is not None:
            budget_bits.append(f"max_hours={self.budget.max_hours}")
        budget_text = ", ".join(budget_bits) or "no budget limit"

        lines = [
            f"# Runner Benchmark: {self.experiment_id}",
            "",
            f"- Generated: `{self.timestamp}`",
            f"- Budget: {budget_text}",
            f"- Runners: {len(self.outcomes)}",
            "",
            "Ranked by mean referee score (scientific conduct), then best validation score.",
            "",
            "| Rank | Runner | Cycles | Best score | Best candidate | Mean referee | "
            "Last referee | Leakage clean | Status | Time (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for rank, o in enumerate(self.ranked(), start=1):
            best_score = f"{o.best_score:.4f}" if o.best_score is not None else "—"
            mean_ref = f"{o.mean_referee:.1f}" if o.mean_referee is not None else "—"
            last_ref = str(o.last_referee) if o.last_referee is not None else "—"
            status = f"error: {o.error}" if o.error else o.status
            lines.append(
                f"| {rank} | `{o.runner}` | {o.cycles} | {best_score} | "
                f"{o.best_candidate or '—'} | {mean_ref} | {last_ref} | "
                f"{'yes' if o.leakage_clean else 'NO'} | {status} | {o.elapsed_seconds:.1f} |"
            )
        return "\n".join(lines) + "\n"

    def to_csv_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for rank, o in enumerate(self.ranked(), start=1):
            rows.append(
                {
                    "rank": rank,
                    "runner": o.runner,
                    "cycles": o.cycles,
                    "best_score": o.best_score if o.best_score is not None else "",
                    "best_candidate": o.best_candidate or "",
                    "mean_referee": (
                        round(o.mean_referee, 2) if o.mean_referee is not None else ""
                    ),
                    "last_referee": o.last_referee if o.last_referee is not None else "",
                    "leakage_clean": o.leakage_clean,
                    "status": o.error or o.status,
                    "elapsed_seconds": round(o.elapsed_seconds, 1),
                }
            )
        return rows


def scaffold_runner_experiment(source_experiment_dir: Path, dest_dir: Path) -> Path:
    """Copy the experiment spec into ``dest_dir`` with fresh loop-managed files."""
    experiment_id = source_experiment_dir.name
    dest_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        dest_dir / "experiment.md",
        (source_experiment_dir / "experiment.md").read_text(encoding="utf-8"),
    )
    write_text(dest_dir / "research_journal.md", research_journal_template(experiment_id))
    write_text(dest_dir / "research_sources.md", research_sources_template(experiment_id))
    write_json(dest_dir / "results.json", [])
    return dest_dir


def _read_scorecards(experiment_dir: Path) -> list[dict[str, object]]:
    cards: list[dict[str, object]] = []
    for path in sorted((experiment_dir / "cycles").glob("*/scorecard.json")):
        try:
            cards.append(load_json(path))
        except (OSError, ValueError):
            continue
    return cards


def aggregate_runner_outcome(
    runner: str, experiment_dir: Path, elapsed_seconds: float
) -> RunnerOutcome:
    """Read the finished experiment dir into a comparable outcome."""
    outcome = RunnerOutcome(
        runner=runner, experiment_dir=experiment_dir, elapsed_seconds=elapsed_seconds
    )

    try:
        state = load_state(experiment_dir)
        outcome.cycles = state.cycle_count
        outcome.status = str(state.status)
    except (OSError, ValueError):
        pass

    try:
        results = load_json(experiment_dir / "results.json")
        ranked = ranked_results(results if isinstance(results, list) else [])
        if ranked:
            outcome.best_score = float(ranked[0]["objective_score"])
            outcome.best_candidate = str(ranked[0].get("candidate_id", "?"))
    except (OSError, ValueError):
        pass

    cards = _read_scorecards(experiment_dir)
    if cards:
        overalls = [int(c.get("overall", 0)) for c in cards]
        outcome.mean_referee = sum(overalls) / len(overalls)
        outcome.last_referee = overalls[-1]
        outcome.leakage_clean = all(
            criterion.get("score", 1.0) >= 1.0
            for c in cards
            for criterion in c.get("criteria", [])
            if criterion.get("name") == "leakage_split_clean"
        )
    return outcome


def _default_run_one(
    experiment_dir: Path, runner_config: RunnerConfig, budget: BenchmarkBudget
) -> None:
    """Run the full loop for one runner with the referee enabled."""
    # Imported here to avoid a circular import at module load (core imports bench-free).
    from .core import acquire_lock, initial_state, release_lock, run_loop
    from .hooks import RefereeCycleHooks

    state = initial_state(
        experiment_dir,
        budget.max_cycles,
        budget.max_hours,
        runner_config=runner_config,
    )
    lock_path = acquire_lock(experiment_dir)
    try:
        run_loop(experiment_dir, state, hooks=RefereeCycleHooks())
    finally:
        release_lock(lock_path)


def run_benchmark(
    source_experiment_dir: Path,
    runner_names: list[str],
    *,
    budget: BenchmarkBudget,
    bench_dir: Path,
    timestamp: str,
    run_one: RunOne | None = None,
    runner_config_for: Callable[[str], RunnerConfig] | None = None,
) -> BenchmarkReport:
    """Run ``runner_names`` over the spec and return a ranked report.

    Each runner executes in its own scaffolded copy under ``bench_dir/<runner>/``.
    A failing runner is recorded with its error and never aborts the others.
    """
    run_one = run_one or _default_run_one
    runner_config_for = runner_config_for or (lambda name: runner_config_from_args(runner=name))

    report = BenchmarkReport(
        experiment_id=source_experiment_dir.name,
        timestamp=timestamp,
        budget=budget,
    )
    for name in runner_names:
        dest = bench_dir / name / source_experiment_dir.name
        scaffold_runner_experiment(source_experiment_dir, dest)
        started = time.monotonic()
        error: str | None = None
        try:
            run_one(dest, runner_config_for(name), budget)
        except Exception as exc:  # noqa: BLE001 - record and continue to next runner
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started

        outcome = aggregate_runner_outcome(name, dest, elapsed)
        outcome.error = error
        report.outcomes.append(outcome)

    write_report(bench_dir, report)
    return report


def write_report(bench_dir: Path, report: BenchmarkReport) -> tuple[Path, Path]:
    """Write comparison.md and comparison.csv; return both paths."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    md_path = bench_dir / "comparison.md"
    csv_path = bench_dir / "comparison.csv"
    write_text(md_path, report.to_markdown())

    rows = report.to_csv_rows()
    fieldnames = [
        "rank",
        "runner",
        "cycles",
        "best_score",
        "best_candidate",
        "mean_referee",
        "last_referee",
        "leakage_clean",
        "status",
        "elapsed_seconds",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return md_path, csv_path
