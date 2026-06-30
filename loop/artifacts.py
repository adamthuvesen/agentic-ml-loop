from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from experiment import journal_path, research_sources_path, results_file
from lib.io import read_text, write_text

logger = logging.getLogger(__name__)


class RollbackError(RuntimeError):
    """Raised when restoring cycle-start artifacts fails.

    Surfaced loudly so a failed rollback can never masquerade as a clean cycle;
    the supervisor aborts rather than continuing on corrupted state.
    """


class ArtifactSnapshot(TypedDict):
    """Comparable view of experiment artifacts for progress detection and retries."""

    journal_hash: str
    sources_hash: str
    results_hash: str
    candidate_ids: list[str]
    results_by_id: dict[str, dict[str, Any]]


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
    before_snapshot: ArtifactSnapshot
    advisory: AdvisorySnapshot


@dataclass(frozen=True)
class RestoreArtifactsRequest:
    experiment_dir: Path
    journal_backup: str
    experiment_md_backup: str
    results_backup: str
    sources_path: Path
    sources_backup: str | None
    pre_cycle_diagnostics: set[Path] | None = None
    pre_cycle_evaluation_reviews: dict[Path, str] | None = None


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
        RestoreArtifactsRequest(
            experiment_dir=experiment_dir,
            journal_backup=baselines.journal_backup,
            experiment_md_backup=baselines.experiment_md_backup,
            results_backup=baselines.results_backup,
            sources_path=baselines.sources_path,
            sources_backup=baselines.sources_backup,
            pre_cycle_diagnostics=baselines.advisory.pre_cycle_diagnostics,
            pre_cycle_evaluation_reviews=baselines.advisory.pre_cycle_evaluation_reviews,
        )
    )


def sha256_text(value: str) -> str:
    """Return a hex SHA-256 digest of ``value``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def artifact_snapshot(experiment_dir: Path) -> ArtifactSnapshot:
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


def compute_progress(before: ArtifactSnapshot, after: ArtifactSnapshot) -> list[str]:
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


def _restore_sources_file(sources_path: Path, sources_backup: str | None) -> None:
    if sources_backup is not None:
        write_text(sources_path, sources_backup)
    else:
        sources_path.unlink(missing_ok=True)


def _restore_evaluation_reviews(
    experiment_dir: Path,
    pre_cycle_evaluation_reviews: dict[Path, str] | None,
) -> None:
    for name in ("evaluation_review.md", "evaluation_review.json"):
        path = experiment_dir / name
        if pre_cycle_evaluation_reviews is not None and path in pre_cycle_evaluation_reviews:
            write_text(path, pre_cycle_evaluation_reviews[path])
        else:
            path.unlink(missing_ok=True)


def _remove_attempt_diagnostic_paths(
    diagnostics_dir: Path,
    pre_cycle_diagnostics: set[Path],
) -> None:
    for path in sorted(
        diagnostics_dir.glob("**/*"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        if path in pre_cycle_diagnostics:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            with contextlib.suppress(OSError):
                path.rmdir()


def _restore_diagnostics(
    experiment_dir: Path,
    pre_cycle_diagnostics: set[Path] | None,
) -> None:
    diagnostics_dir = experiment_dir / "diagnostics"
    if not diagnostics_dir.is_dir():
        return
    if pre_cycle_diagnostics is None:
        shutil.rmtree(diagnostics_dir)
        return

    _remove_attempt_diagnostic_paths(diagnostics_dir, pre_cycle_diagnostics)
    if diagnostics_dir not in pre_cycle_diagnostics:
        with contextlib.suppress(OSError):
            diagnostics_dir.rmdir()


def restore_artifacts(request: RestoreArtifactsRequest) -> None:
    """Reset mutable files and advisory artifacts to the cycle-start snapshot.

    Raises :class:`RollbackError` if any restore operation fails, so a corrupted
    rollback aborts the cycle loudly instead of leaving partial state behind.
    """
    try:
        write_text(journal_path(request.experiment_dir), request.journal_backup)
        write_text(request.experiment_dir / "experiment.md", request.experiment_md_backup)
        write_text(results_file(request.experiment_dir), request.results_backup)
        _restore_sources_file(request.sources_path, request.sources_backup)
        _restore_evaluation_reviews(request.experiment_dir, request.pre_cycle_evaluation_reviews)
        _restore_diagnostics(request.experiment_dir, request.pre_cycle_diagnostics)
    except OSError as exc:
        logger.error("Failed to restore cycle artifacts in %s: %s", request.experiment_dir, exc)
        raise RollbackError(
            f"Could not restore cycle-start artifacts in {request.experiment_dir}: {exc}"
        ) from exc
