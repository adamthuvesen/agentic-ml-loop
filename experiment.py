from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lib.io import load_json, read_text
from lib.paths import DATA_DIRNAME
from lib.result_schema import validate_result_entries

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
JOURNAL_FILE_NAME = "research_journal.md"
RESEARCH_SOURCES_FILE_NAME = "research_sources.md"
RESULTS_FILE_NAME = "results.json"
LEARNINGS_FILE_NAME = "learnings.md"
DIAGNOSTICS_DIR_NAME = "diagnostics"
DIAGNOSTICS_SUMMARY_FILE_NAME = "summary.md"
DIAGNOSTICS_REPORT_FILE_NAME = "report.json"
EVALUATION_REVIEW_FILE_NAME = "evaluation_review.md"
EVALUATION_REVIEW_REPORT_FILE_NAME = "evaluation_review.json"
REQUIRED_FILES = (
    "experiment.md",
    "research_journal.md",
    "results.json",
)

LOOP_MANAGED_FILES: frozenset[str] = frozenset(
    {
        "experiment.md",
        "research_journal.md",
        "research_sources.md",
        "results.json",
        "evaluation_review.md",
        "evaluation_review.json",
        "feedback.json",
        "notebook.yaml",
        "loop_state.json",
        ".loop.lock",
        "status.md",
        "results.json.lock",
    }
)

LOOP_MANAGED_DIRECTORIES: frozenset[str] = frozenset(
    {
        "cycles",
        "diagnostics",
        "outputs",
        "work",
        "scripts",
    }
)

# Materialized input data (a warehouse snapshot + its manifest) lives in
# ``data/`` (lib.paths.DATA_DIRNAME). It is allowed in the experiment root but
# deliberately NOT loop-managed: a force re-init must never delete an expensive
# pulled snapshot.
#
# Anything outside this allowlist surfaces as a soft "stray files" warning so the
# agent steers new artefacts into outputs/, work/, or scripts/.
EXPERIMENT_ROOT_ALLOWLIST: frozenset[str] = (
    LOOP_MANAGED_FILES | LOOP_MANAGED_DIRECTORIES | {DATA_DIRNAME}
)


def journal_path(experiment_dir: Path) -> Path:
    return experiment_dir / JOURNAL_FILE_NAME


def research_sources_path(experiment_dir: Path) -> Path:
    return experiment_dir / RESEARCH_SOURCES_FILE_NAME


def results_file(experiment_dir: Path) -> Path:
    return experiment_dir / RESULTS_FILE_NAME


def diagnostics_dir(experiment_dir: Path) -> Path:
    return experiment_dir / DIAGNOSTICS_DIR_NAME


def diagnostics_summary_path(experiment_dir: Path) -> Path:
    return diagnostics_dir(experiment_dir) / DIAGNOSTICS_SUMMARY_FILE_NAME


def diagnostics_report_path(experiment_dir: Path) -> Path:
    return diagnostics_dir(experiment_dir) / DIAGNOSTICS_REPORT_FILE_NAME


def evaluation_review_path(experiment_dir: Path) -> Path:
    return experiment_dir / EVALUATION_REVIEW_FILE_NAME


def evaluation_review_report_path(experiment_dir: Path) -> Path:
    return experiment_dir / EVALUATION_REVIEW_REPORT_FILE_NAME


def learnings_file() -> Path:
    return ROOT / LEARNINGS_FILE_NAME


def read_diagnostics_summary(experiment_dir: Path) -> str | None:
    """Read diagnostics summary markdown if present and non-empty."""
    path = diagnostics_summary_path(experiment_dir)
    if not path.exists():
        return None
    content = read_text(path).strip()
    return content if content else None


def read_evaluation_review(experiment_dir: Path) -> str | None:
    """Read evaluation review markdown if present and non-empty."""
    path = evaluation_review_path(experiment_dir)
    if not path.exists():
        return None
    content = read_text(path).strip()
    return content if content else None


def get_cross_learnings_enabled(experiment_dir: Path) -> bool:
    """Check if cross-experiment learnings are enabled for this experiment.

    Looks for ``cross_learnings: false`` in experiment.md. Defaults to True.
    """
    spec_path = experiment_dir / "experiment.md"
    if not spec_path.exists():
        return True
    text = read_text(spec_path)
    match = re.search(r"^cross_learnings:\s*(false|no)\s*$", text, re.MULTILINE | re.IGNORECASE)
    return match is None


def get_min_cycles_before_complete(experiment_dir: Path) -> int | None:
    """Optional floor on EXPERIMENT_COMPLETE from ``experiment.md``.

    If present, the line::

        Minimum loop cycles before EXPERIMENT_COMPLETE: N

    requires at least N ``## Cycle NNNN:`` headings in ``research_journal.md``
    before the loop accepts ``<promise>EXPERIMENT_COMPLETE</promise>``.
    """
    spec_path = experiment_dir / "experiment.md"
    if not spec_path.exists():
        return None
    text = read_text(spec_path)
    match = re.search(
        r"Minimum\s+loop\s+cycles\s+before\s+EXPERIMENT_COMPLETE:\s*(\d+)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return int(match.group(1)) if match else None


def count_journal_cycles(experiment_dir: Path) -> int:
    """Count ``## Cycle NNNN:`` entries in ``research_journal.md``."""
    path = journal_path(experiment_dir)
    if not path.exists():
        return 0
    text = read_text(path)
    return len(re.findall(r"^## Cycle \d+:", text, re.MULTILINE))


def read_learnings() -> str | None:
    """Read the canonical repo-root ``learnings.md`` file."""
    path = learnings_file()
    if not path.exists():
        return None
    content = read_text(path).strip()
    return content if content else None


def research_journal_template(experiment_id: str) -> str:
    return f"""# Research Journal: {experiment_id}

Write one entry per cycle. Include what you set out to learn, what you found,
and what it means for next steps. Use whatever structure fits — a research
cycle looks different from a modeling cycle.
"""


def research_sources_template(experiment_id: str) -> str:
    return f"""# Research Sources: {experiment_id}

Durable, source-backed research memory for this experiment.

Use this file for reusable outside research: Kaggle write-ups, papers, blog posts,
docs, and similar-problem references that future cycles should build on.

Keep cycle-by-cycle reasoning, hypotheses, and results in `research_journal.md`.
Treat this as a living document: source cards preserve provenance, while
`Reusable Takeaways` should be rewritten when later findings narrow, qualify, or
contradict earlier conclusions.
Phrase takeaways as scoped heuristics with caveats, not universal laws.

## Reusable Takeaways

Rewrite these bullets as your understanding evolves. Do not leave contradictions
unresolved here; this section is the current summary for future cycles. If a
source card and a takeaway conflict, trust the takeaway and revise the card only
if needed for clarity.

- _none yet_

## Source Cards

Source cards are provenance, not final truth. They preserve what each source said
and why it mattered; `Reusable Takeaways` is what future cycles should follow.

### Source 001: <title>

- **Type:** kaggle | paper | blog | docs
- **URL:** <link>
- **Why relevant here:** <one or two sentences>
- **Key takeaways:** <short bullets or prose>
- **Applicability / caveats:** <where this helps or does not apply>
- **Ideas this suggests:** <candidate next moves>
- **Status:** used | deferred | ruled_out
"""


def get_objective_metric(experiment_dir: Path) -> str | None:
    """Extract the declared objective metric identifier from experiment.md.

    Looks for a parenthesized token (e.g. ``(val_auc)``) inside the
    ``## Objective Metric`` section. Returns ``None`` if the section or
    identifier is missing.
    """
    spec_path = experiment_dir / "experiment.md"
    if not spec_path.exists():
        return None
    text = read_text(spec_path)
    # Find the ## Objective Metric section
    section_match = re.search(
        r"^## Objective Metric\s*\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not section_match:
        return None
    section_text = section_match.group(1)
    # Extract parenthesized identifier like (val_auc) or (`val_auc`)
    token_match = re.search(r"\(`?([a-z][a-z0-9_]*)`?\)", section_text)
    return token_match.group(1) if token_match else None


def stray_root_entries(experiment_dir: Path) -> list[str]:
    """Return names of entries in the experiment root outside the allowlist.

    Hidden dotfiles other than ``.loop.lock`` are ignored — agents drop
    ad-hoc ``.DS_Store`` etc. and we do not want to flag those.
    """
    if not experiment_dir.is_dir():
        return []
    stray: list[str] = []
    for entry in sorted(experiment_dir.iterdir()):
        name = entry.name
        if name in EXPERIMENT_ROOT_ALLOWLIST:
            continue
        if name.startswith(".") and name != ".loop.lock":
            continue
        stray.append(name)
    return stray


def _snapshot_validation_errors(experiment_dir: Path) -> list[str]:
    """Check a materialized warehouse snapshot against its manifest, if present.

    Returns an empty list when the experiment has no ``dataset_manifest.json``
    (local/CSV experiments are unaffected), a soft warning when pyarrow or
    ``lib.sources`` is unavailable, and an error when the parquet has drifted
    from its manifest. Never raises — a validator must not crash on bad input.
    """
    manifest_path = experiment_dir / DATA_DIRNAME / "dataset_manifest.json"
    if not manifest_path.exists():
        return []
    snapshot_path = experiment_dir / DATA_DIRNAME / "snapshot.parquet"
    if not snapshot_path.exists():
        return [f"manifest present but snapshot missing: {snapshot_path}"]
    try:
        from lib.sources.manifest import DatasetManifest, verify_snapshot
    except ImportError:
        return ["warning: lib.sources unavailable; snapshot integrity not checked"]
    try:
        manifest = DatasetManifest.load(manifest_path)
    except (ValueError, json.JSONDecodeError) as exc:
        return [f"invalid dataset_manifest.json: {exc}"]
    try:
        verify_snapshot(snapshot_path, manifest)
    except ModuleNotFoundError:
        return ["warning: pyarrow not installed; snapshot integrity not checked"]
    except Exception as exc:  # report any drift/corruption without crashing validate
        return [f"snapshot integrity check failed: {exc}"]
    return []


def validate_experiment(experiment_dir: Path, strict_completion: bool = False) -> list[str]:
    errors: list[str] = []
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        return [f"Experiment directory does not exist: {experiment_dir}"]

    for filename in REQUIRED_FILES:
        if not (experiment_dir / filename).exists():
            errors.append(f"Missing required file: {experiment_dir / filename}")
    if errors:
        return errors

    stray = stray_root_entries(experiment_dir)
    if stray:
        joined = ", ".join(stray)
        errors.append(
            f"warning: stray files in experiment root ({joined}); "
            f"move deliverables to outputs/, scratch to work/, scripts to scripts/"
        )

    rf = results_file(experiment_dir)
    try:
        results = load_json(rf)
    except json.JSONDecodeError as exc:
        errors.append(f"{rf.name} is invalid JSON: {exc}")
        return errors
    if not isinstance(results, list):
        errors.append(f"{rf.name} must be a JSON list")
        return errors

    if strict_completion and not results:
        errors.append("strict completion requires at least one result entry")
    else:
        errors.extend(validate_result_entries(results, strict_completion=strict_completion))

    # Metric consistency checks
    declared_metric = get_objective_metric(experiment_dir)
    if declared_metric and isinstance(results, list) and results:
        # Check prefix convention
        if not (declared_metric.startswith("val_") or declared_metric.startswith("validation_")):
            msg = (
                f"declared objective metric '{declared_metric}' does not follow "
                f"val_/validation_ prefix convention"
            )
            errors.append(msg if strict_completion else f"warning: {msg}")

        # Check each result entry matches
        for entry in results:
            if not isinstance(entry, dict):
                continue
            entry_metric = entry.get("objective_metric")
            if entry_metric and entry_metric != declared_metric:
                cid = entry.get("candidate_id", "?")
                msg = (
                    f"candidate '{cid}' has objective_metric='{entry_metric}' "
                    f"but experiment declares '{declared_metric}'"
                )
                errors.append(msg if strict_completion else f"warning: {msg}")

    errors.extend(_snapshot_validation_errors(experiment_dir))

    return errors


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-search experiment helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate an experiment")
    validate_parser.add_argument("experiment_path", help="Path to the experiment directory")
    validate_parser.add_argument(
        "--strict-completion",
        action="store_true",
        help="Require at least one result entry",
    )
    return parser


def main() -> int:
    args = cli_parser().parse_args()

    if args.command == "validate":
        messages = validate_experiment(
            Path(args.experiment_path), strict_completion=args.strict_completion
        )
        warnings = [m for m in messages if m.startswith("warning: ")]
        errors = [m for m in messages if not m.startswith("warning: ")]
        if errors:
            print("Validation failed:")
            for error in errors:
                print(f"  - {error}")
            for warning in warnings:
                print(f"  - {warning}")
            return 1
        if warnings:
            print(f"Validation passed with warnings: {args.experiment_path}")
            for warning in warnings:
                print(f"  - {warning}")
            return 0
        print(f"Validation passed: {args.experiment_path}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
