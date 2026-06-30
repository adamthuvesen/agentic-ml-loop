from __future__ import annotations

import argparse
import fcntl
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment import (
    LOOP_MANAGED_DIRECTORIES,
    LOOP_MANAGED_FILES,
    research_sources_template,
)
from lib.io import load_json, write_json
from lib.result_schema import validate_result_entry

ROOT = Path(__file__).resolve().parent.parent


def experiment_dir_from_arg(value: str) -> Path:
    """Resolve a CLI ``--experiment`` argument to an absolute path.

    Accepts either a path that already exists (used as-is) or a value relative to
    the repo root, so both ``experiments/demo_classification`` and an absolute path
    work from any working directory.
    """
    path = Path(value)
    if path.exists():
        return path.resolve()
    return (ROOT / value).resolve()


def _load_existing_results(results_path: Path) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    payload = load_json(results_path)
    if not isinstance(payload, list):
        raise ValueError(f"{results_path} must contain a JSON list of result objects")
    for index, item in enumerate(payload):
        _raise_for_result_errors(
            validate_result_entry(
                item,
                f"{results_path}[{index}]",
                strict_completion=False,
            )
        )
    return payload


def _raise_for_result_errors(messages: list[str]) -> None:
    errors = [message for message in messages if not message.startswith("warning: ")]
    if errors:
        raise ValueError("; ".join(errors))


def _raise_if_selection_frozen(experiment_dir: Path) -> None:
    state_path = experiment_dir / "loop_state.json"
    if not state_path.exists():
        return
    state = load_json(state_path)
    if not isinstance(state, dict) or not (
        state.get("selection_frozen") or state.get("final_holdout_accessed")
    ):
        return
    frozen_ids = state.get("frozen_candidate_ids") or []
    frozen_text = ", ".join(str(candidate_id) for candidate_id in frozen_ids) or "n/a"
    raise ValueError(
        "Selection is frozen for this experiment; refusing to write results.json. "
        f"Frozen candidates: {frozen_text}. Use the guarded final-holdout path for "
        "test scoring or unfreeze explicitly before more validation search."
    )


def save_candidate_result(
    experiment_dir: Path,
    candidate_id: str,
    splits_loader: Callable[[], Any],
    candidate_runners: dict[str, Callable],
) -> dict[str, Any]:
    """Run one candidate and merge its validated result into ``results.json``.

    Loads splits lazily via *splits_loader*, runs the runner registered under
    *candidate_id*, validates the result payload, then writes it under an
    exclusive file lock so concurrent runners cannot corrupt ``results.json``.
    A prior result for the same candidate is replaced, and entries are kept
    sorted by ``objective_score`` (descending).

    Returns the saved result payload. Raises ``ValueError`` for an unknown
    candidate or an invalid result, and ``FileNotFoundError`` if the experiment
    directory is missing.
    """
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    if candidate_id not in candidate_runners:
        known = ", ".join(sorted(candidate_runners)) or "none"
        raise ValueError(f"Unknown candidate '{candidate_id}'. Available: {known}")
    _raise_if_selection_frozen(experiment_dir)

    results_path = experiment_dir / "results.json"
    _load_existing_results(results_path)

    splits = splits_loader()
    candidate = candidate_runners[candidate_id](splits)
    payload = candidate.result_payload()
    _raise_for_result_errors(
        validate_result_entry(
            payload,
            f"candidate {candidate_id!r} result",
            strict_completion=True,
        )
    )

    lock_path = results_path.with_suffix(".json.lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        existing = _load_existing_results(results_path)
        existing = [r for r in existing if r.get("candidate_id") != candidate.candidate_id]
        existing.append(payload)
        existing.sort(key=lambda x: x.get("objective_score", float("-inf")), reverse=True)
        write_json(results_path, existing)
    return payload


def _remove_loop_managed_artifacts(experiment_dir: Path) -> None:
    if not experiment_dir.exists():
        return
    for artifact in sorted(LOOP_MANAGED_FILES):
        (experiment_dir / artifact).unlink(missing_ok=True)
    for dirname in sorted(LOOP_MANAGED_DIRECTORIES):
        path = experiment_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def _seed_text_file(path: Path, content: Callable[[], str], *, force: bool) -> None:
    if force or not path.exists():
        path.write_text(content(), encoding="utf-8")


def _experiment_template_text(template_path: Path | None) -> str:
    source = template_path or (ROOT / "experiments" / "templates" / "model-search.md")
    return source.read_text(encoding="utf-8")


def _research_journal_text(experiment_id: str) -> str:
    return (
        f"# Research Journal: {experiment_id}\n\n"
        "Write one entry per cycle. Include what you set out to learn, what you found,\n"
        "and what it means for next steps.\n"
    )


def _research_sources_text(
    experiment_id: str,
    research_sources_template_path: Path | None,
) -> str:
    if research_sources_template_path:
        return research_sources_template_path.read_text(encoding="utf-8")
    return research_sources_template(experiment_id)


def _seed_experiment_files(
    experiment_dir: Path,
    experiment_id: str,
    template_path: Path | None,
    research_sources_template_path: Path | None,
    *,
    force: bool,
) -> None:
    _seed_text_file(
        experiment_dir / "experiment.md",
        lambda: _experiment_template_text(template_path),
        force=force,
    )
    _seed_text_file(
        experiment_dir / "research_journal.md",
        lambda: _research_journal_text(experiment_id),
        force=force,
    )
    _seed_text_file(
        experiment_dir / "research_sources.md",
        lambda: _research_sources_text(experiment_id, research_sources_template_path),
        force=force,
    )
    if force or not (experiment_dir / "results.json").exists():
        write_json(experiment_dir / "results.json", [])


def _ensure_three_folder_layout(experiment_dir: Path) -> None:
    # Universal three-folder output layout. Idempotent: existing files are preserved.
    for subdir in ("outputs", "work", "scripts"):
        target = experiment_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        gitkeep = target / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.write_text("", encoding="utf-8")


def init_experiment_dir(
    experiment_id: str,
    template_path: Path | None = None,
    research_sources_template_path: Path | None = None,
    force: bool = False,
) -> Path:
    """Create or refresh an experiment directory.

    Args:
        experiment_id: Experiment slug used as the directory name.
        template_path: Path to a template file whose content seeds ``experiment.md``.
            Falls back to the generic ``experiments/templates/model-search.md``.
        research_sources_template_path: Optional path to a template for
            ``research_sources.md``.  Falls back to the default generated scaffold.
        force: When True, remove existing artifacts before re-seeding.
    """
    experiment_dir = ROOT / "experiments" / experiment_id
    if force:
        _remove_loop_managed_artifacts(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _seed_experiment_files(
        experiment_dir,
        experiment_id,
        template_path,
        research_sources_template_path,
        force=force,
    )
    _ensure_three_folder_layout(experiment_dir)
    return experiment_dir


def runner_cli(
    experiment_id: str,
    candidate_choices: list[str],
    retired_candidate_choices: list[str] | None = None,
) -> argparse.ArgumentParser:
    """Build the argparse parser shared by every demo runner entrypoint.

    Wires up the ``list-candidates``, ``run-candidate``, and ``init-demo``
    subcommands. When *retired_candidate_choices* is non-empty, the matching
    ``list-retired-candidates`` / ``run-retired-candidate`` subcommands are added
    so archived candidates stay runnable for reproducibility.
    """
    retired_candidate_choices = retired_candidate_choices or []
    parser = argparse.ArgumentParser(description=f"{experiment_id} runner for agentic-ml-loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-candidates", help="List bundled candidates")
    if retired_candidate_choices:
        subparsers.add_parser(
            "list-retired-candidates",
            help="List archived candidates preserved for reproducibility",
        )

    run_parser = subparsers.add_parser(
        "run-candidate", help="Run one candidate and update artifacts"
    )
    run_parser.add_argument(
        "--experiment", required=True, help="Experiment directory or relative path"
    )
    run_parser.add_argument(
        "--candidate",
        required=True,
        choices=sorted(candidate_choices),
        help="Candidate id to run",
    )
    if retired_candidate_choices:
        retired_parser = subparsers.add_parser(
            "run-retired-candidate",
            help="Run one archived candidate and update artifacts",
        )
        retired_parser.add_argument(
            "--experiment", required=True, help="Experiment directory or relative path"
        )
        retired_parser.add_argument(
            "--candidate",
            required=True,
            choices=sorted(retired_candidate_choices),
            help="Retired candidate id to run",
        )

    init_parser = subparsers.add_parser("init-demo", help="Create or refresh the experiment")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh artifacts even if the experiment exists",
    )
    return parser


@dataclass(frozen=True)
class RunnerMainConfig:
    experiment_id: str
    candidate_runners: dict[str, Callable]
    dataset_loader: Callable[[], Any]
    template_path: Path | None = None
    research_sources_template_path: Path | None = None
    retired_candidate_runners: dict[str, Callable] | None = None


def _coerce_runner_main_config(
    config_or_experiment_id: RunnerMainConfig | str | None,
    legacy_args: tuple[Any, ...],
    legacy_kwargs: dict[str, Any],
) -> RunnerMainConfig:
    if isinstance(config_or_experiment_id, RunnerMainConfig):
        if legacy_args or legacy_kwargs:
            raise TypeError("run_runner_main() received both RunnerMainConfig and legacy arguments")
        return config_or_experiment_id

    values = dict(legacy_kwargs)
    if config_or_experiment_id is None:
        try:
            config_or_experiment_id = values.pop("experiment_id")
        except KeyError as exc:
            raise TypeError("run_runner_main() missing required experiment_id") from exc

    legacy_fields = [
        "candidate_runners",
        "dataset_loader",
        "template_path",
        "research_sources_template_path",
        "retired_candidate_runners",
    ]
    if len(legacy_args) > len(legacy_fields):
        raise TypeError("run_runner_main() got too many positional arguments")
    positional_values = dict(zip(legacy_fields, legacy_args, strict=False))
    duplicate = sorted(set(positional_values) & set(values))
    if duplicate:
        raise TypeError(
            "run_runner_main() got multiple values for argument(s): " + ", ".join(duplicate)
        )

    unknown = sorted(set(values) - set(legacy_fields))
    if unknown:
        raise TypeError("run_runner_main() got unexpected keyword arguments: " + ", ".join(unknown))
    values = {**positional_values, **values}
    missing = [key for key in ("candidate_runners", "dataset_loader") if key not in values]
    if missing:
        raise TypeError(
            "run_runner_main() missing required legacy arguments: " + ", ".join(missing)
        )

    return RunnerMainConfig(
        experiment_id=str(config_or_experiment_id),
        candidate_runners=values["candidate_runners"],
        dataset_loader=values["dataset_loader"],
        template_path=values.get("template_path"),
        research_sources_template_path=values.get("research_sources_template_path"),
        retired_candidate_runners=values.get("retired_candidate_runners"),
    )


def run_runner_main(
    config: RunnerMainConfig | str | None = None,
    *legacy_args: Any,
    **legacy_kwargs: Any,
) -> int:
    """Parse argv and dispatch a demo runner subcommand. Returns a process exit code.

    Shared ``main`` for the ``runners/<id>_runner.py`` entrypoints: lists
    candidates, initializes the experiment directory, or runs a (possibly retired)
    candidate and prints its objective score as JSON.
    """
    config = _coerce_runner_main_config(config, legacy_args, legacy_kwargs)
    retired_candidate_runners = config.retired_candidate_runners or {}
    parser = runner_cli(
        config.experiment_id,
        list(config.candidate_runners.keys()),
        list(retired_candidate_runners.keys()),
    )
    args = parser.parse_args()

    if args.command == "list-candidates":
        for cid in config.candidate_runners:
            print(cid)
        return 0

    if args.command == "list-retired-candidates":
        for cid in retired_candidate_runners:
            print(cid)
        return 0

    if args.command == "init-demo":
        path = init_experiment_dir(
            config.experiment_id,
            config.template_path,
            config.research_sources_template_path,
            force=args.force,
        )
        print(f"Initialized {config.experiment_id} experiment at {path}")
        return 0

    if args.command in {"run-candidate", "run-retired-candidate"}:
        experiment_dir = experiment_dir_from_arg(args.experiment)
        runners = (
            retired_candidate_runners
            if args.command == "run-retired-candidate"
            else config.candidate_runners
        )
        result = save_candidate_result(
            experiment_dir, args.candidate, config.dataset_loader, runners
        )
        print(
            json.dumps(
                {
                    "candidate_id": result["candidate_id"],
                    "objective_metric": result["objective_metric"],
                    "objective_score": result["objective_score"],
                },
                indent=2,
            )
        )
        return 0

    raise ValueError(f"Unknown command: {args.command}")
