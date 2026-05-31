from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment import journal_path, research_sources_path, results_file
from lib.utils import read_text, write_text


@dataclass
class AdvisorySnapshot:
    pre_cycle_diagnostics: set[Path]
    pre_cycle_evaluation_reviews: dict[Path, str]


@dataclass
class CycleBaselines:
    journal_backup: str
    experiment_md_backup: str
    results_backup: str
    sources_path: Path
    sources_backup: str | None
    before_snapshot: dict[str, Any]
    advisory: AdvisorySnapshot


def capture_advisory_snapshot(experiment_dir: Path) -> AdvisorySnapshot:
    """Record diagnostics and evaluation-review paths present before a cycle."""
    diagnostics_dir = experiment_dir / "diagnostics"
    pre_cycle_diagnostics = (
        {diagnostics_dir, *set(diagnostics_dir.glob("**/*"))} if diagnostics_dir.is_dir() else set()
    )
    pre_cycle_evaluation_reviews = {
        p: read_text(p)
        for p in (
            experiment_dir / "evaluation_review.md",
            experiment_dir / "evaluation_review.json",
        )
        if p.exists()
    }
    return AdvisorySnapshot(
        pre_cycle_diagnostics=pre_cycle_diagnostics,
        pre_cycle_evaluation_reviews=pre_cycle_evaluation_reviews,
    )


def capture_cycle_baselines(experiment_dir: Path) -> CycleBaselines:
    """Snapshot mutable experiment files at cycle start for retry rollback."""
    sources_path = research_sources_path(experiment_dir)
    return CycleBaselines(
        journal_backup=(
            read_text(journal_path(experiment_dir)) if journal_path(experiment_dir).exists() else ""
        ),
        experiment_md_backup=read_text(experiment_dir / "experiment.md"),
        results_backup=(
            read_text(results_file(experiment_dir))
            if results_file(experiment_dir).exists()
            else "[]"
        ),
        sources_path=sources_path,
        sources_backup=read_text(sources_path) if sources_path.exists() else None,
        before_snapshot=artifact_snapshot(experiment_dir),
        advisory=capture_advisory_snapshot(experiment_dir),
    )


def restore_cycle_baselines(experiment_dir: Path, baselines: CycleBaselines) -> None:
    """Restore experiment files to the cycle-start snapshot."""
    restore_artifacts(
        experiment_dir,
        baselines.journal_backup,
        baselines.experiment_md_backup,
        baselines.results_backup,
        baselines.sources_path,
        baselines.sources_backup,
        baselines.advisory.pre_cycle_diagnostics,
        baselines.advisory.pre_cycle_evaluation_reviews,
    )


def sha256_text(value: str) -> str:
    """Return a hex SHA-256 digest of ``value``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact_snapshot(experiment_dir: Path) -> dict[str, Any]:
    """Capture a comparable view of experiment artifacts for progress and retries."""
    journal_file = journal_path(experiment_dir)
    journal_text = read_text(journal_file) if journal_file.exists() else ""

    sources = research_sources_path(experiment_dir)
    sources_text = read_text(sources) if sources.exists() else ""

    rf = results_file(experiment_dir)
    results_text = read_text(rf) if rf.exists() else "[]"

    try:
        results = json.loads(results_text)
    except json.JSONDecodeError:
        results = []

    results_by_id: dict[str, dict[str, Any]] = {}
    for item in results:
        if isinstance(item, dict):
            cid = str(item.get("candidate_id", "")).strip()
            if cid:
                results_by_id[cid] = item

    return {
        "journal_hash": sha256_text(journal_text),
        "sources_hash": sha256_text(sources_text),
        "results_hash": sha256_text(results_text),
        "candidate_ids": sorted(results_by_id),
        "results_by_id": results_by_id,
    }


def compute_progress(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Describe what changed between two `artifact_snapshot` results."""
    reasons: list[str] = []

    if before["journal_hash"] != after["journal_hash"]:
        reasons.append("journal_updated")
    if before["sources_hash"] != after["sources_hash"]:
        reasons.append("research_sources_updated")

    new_candidates = sorted(set(after["candidate_ids"]) - set(before["candidate_ids"]))
    if new_candidates:
        reasons.append(f"new_candidates:{', '.join(new_candidates)}")

    for cid in sorted(set(before["candidate_ids"]) & set(after["candidate_ids"])):
        if before["results_by_id"][cid] != after["results_by_id"][cid]:
            reasons.append(f"changed:{cid}")

    return reasons


def restore_artifacts(
    experiment_dir: Path,
    journal_backup: str,
    experiment_md_backup: str,
    results_backup: str,
    sources_path: Path,
    sources_backup: str | None,
    pre_cycle_diagnostics: set[Path] | None = None,
    pre_cycle_evaluation_reviews: dict[Path, str] | None = None,
) -> None:
    """Reset mutable files and advisory artifacts to the cycle-start snapshot."""
    write_text(journal_path(experiment_dir), journal_backup)
    write_text(experiment_dir / "experiment.md", experiment_md_backup)
    write_text(results_file(experiment_dir), results_backup)
    if sources_backup is not None:
        write_text(sources_path, sources_backup)
    else:
        sources_path.unlink(missing_ok=True)

    for name in ("evaluation_review.md", "evaluation_review.json"):
        p = experiment_dir / name
        if pre_cycle_evaluation_reviews is not None and p in pre_cycle_evaluation_reviews:
            write_text(p, pre_cycle_evaluation_reviews[p])
        else:
            p.unlink(missing_ok=True)

    diagnostics_dir = experiment_dir / "diagnostics"
    if diagnostics_dir.is_dir():
        if pre_cycle_diagnostics is not None:
            for f in sorted(
                diagnostics_dir.glob("**/*"),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                if f in pre_cycle_diagnostics:
                    continue
                if f.is_file() or f.is_symlink():
                    f.unlink(missing_ok=True)
                elif f.is_dir():
                    with contextlib.suppress(OSError):
                        f.rmdir()
            if diagnostics_dir not in pre_cycle_diagnostics:
                with contextlib.suppress(OSError):
                    diagnostics_dir.rmdir()
        else:
            shutil.rmtree(diagnostics_dir)
