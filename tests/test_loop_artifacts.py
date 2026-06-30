from pathlib import Path
from unittest.mock import patch

import pytest

from loop import artifact_snapshot, compute_progress
from loop.artifacts import RestoreArtifactsRequest, RollbackError
from loop.core import _restore_artifacts
from tests.loop.conftest import _make_experiment


class TestArtifactSnapshot:
    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "research_journal.md").write_text("# Journal\n")
        (d / "results.json").write_text('[{"candidate_id": "a", "objective_score": 0.8}]\n')
        snap = artifact_snapshot(d)
        assert set(snap.keys()) == {
            "journal_hash",
            "sources_hash",
            "results_hash",
            "candidate_ids",
            "results_by_id",
        }
        assert snap["candidate_ids"] == ["a"]
        assert "a" in snap["results_by_id"]


class TestComputeProgress:
    def test_new_candidate(self) -> None:
        before = {
            "journal_hash": "abc",
            "sources_hash": "sources-a",
            "results_hash": "x",
            "candidate_ids": [],
            "results_by_id": {},
        }
        after = {
            "journal_hash": "abc",
            "sources_hash": "sources-a",
            "results_hash": "y",
            "candidate_ids": ["foo"],
            "results_by_id": {"foo": {"candidate_id": "foo"}},
        }
        reasons = compute_progress(before, after)
        assert "new_candidates:foo" in reasons

    def test_journal_updated_counts_as_progress(self) -> None:
        before = {
            "journal_hash": "aaa",
            "sources_hash": "sources-a",
            "results_hash": "x",
            "candidate_ids": [],
            "results_by_id": {},
        }
        after = {
            "journal_hash": "bbb",
            "sources_hash": "sources-a",
            "results_hash": "x",
            "candidate_ids": [],
            "results_by_id": {},
        }
        reasons = compute_progress(before, after)
        assert reasons == ["journal_updated"]

    def test_research_sources_updated_counts_as_progress(self) -> None:
        before = {
            "journal_hash": "aaa",
            "sources_hash": "sources-a",
            "results_hash": "x",
            "candidate_ids": [],
            "results_by_id": {},
        }
        after = {
            "journal_hash": "aaa",
            "sources_hash": "sources-b",
            "results_hash": "x",
            "candidate_ids": [],
            "results_by_id": {},
        }
        reasons = compute_progress(before, after)
        assert reasons == ["research_sources_updated"]

    def test_no_change(self) -> None:
        snap = {
            "journal_hash": "abc",
            "sources_hash": "sources-a",
            "results_hash": "x",
            "candidate_ids": ["a"],
            "results_by_id": {"a": {"candidate_id": "a"}},
        }
        assert compute_progress(snap, snap) == []


class TestRestoreArtifactsAdvisoryCleanup:
    """Verify _restore_artifacts handles advisory artifacts on rollback."""

    def test_deletes_attempt_created_advisory_artifacts(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"
        # Create advisory artifacts that did not exist before the attempt.
        (d / "evaluation_review.md").write_text("attempt review")
        (d / "evaluation_review.json").write_text("{}")
        diag = d / "diagnostics"
        diag.mkdir()
        (diag / "report.json").write_text("{}")

        _restore_artifacts(
            RestoreArtifactsRequest(
                experiment_dir=d,
                journal_backup="# Journal\n",
                experiment_md_backup="# Experiment\n",
                results_backup="[]\n",
                sources_path=sources_path,
                sources_backup="# Research Sources\n",
                pre_cycle_evaluation_reviews={},
            )
        )

        assert not (d / "evaluation_review.md").exists()
        assert not (d / "evaluation_review.json").exists()
        assert not diag.exists()

    def test_pre_existing_evaluation_review_survives_retry(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"
        review_md = d / "evaluation_review.md"
        review_json = d / "evaluation_review.json"
        review_md.write_text("pre-cycle review")
        review_json.write_text('{"pre_cycle": true}')
        pre_cycle_reviews = {
            review_md: review_md.read_text(),
            review_json: review_json.read_text(),
        }

        review_md.write_text("attempt review")
        review_json.write_text('{"attempt": true}')

        _restore_artifacts(
            RestoreArtifactsRequest(
                experiment_dir=d,
                journal_backup="# Journal\n",
                experiment_md_backup="# Experiment\n",
                results_backup="[]\n",
                sources_path=sources_path,
                sources_backup="# Research Sources\n",
                pre_cycle_evaluation_reviews=pre_cycle_reviews,
            )
        )

        assert review_md.read_text() == "pre-cycle review"
        assert review_json.read_text() == '{"pre_cycle": true}'

    def test_no_error_when_advisory_artifacts_missing(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"

        # Should not raise when advisory artifacts don't exist
        _restore_artifacts(
            RestoreArtifactsRequest(
                experiment_dir=d,
                journal_backup="# Journal\n",
                experiment_md_backup="# Experiment\n",
                results_backup="[]\n",
                sources_path=sources_path,
                sources_backup="# Research Sources\n",
            )
        )

    def test_attempt_created_sources_file_is_removed_when_absent_before(
        self, tmp_path: Path
    ) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"
        sources_path.unlink()
        sources_path.write_text("# Attempt sources\n")

        _restore_artifacts(
            RestoreArtifactsRequest(
                experiment_dir=d,
                journal_backup="# Journal\n",
                experiment_md_backup="# Experiment\n",
                results_backup="[]\n",
                sources_path=sources_path,
                sources_backup=None,
            )
        )

        assert not sources_path.exists()

    def test_attempt_created_diagnostics_dir_is_removed_when_absent_before(
        self, tmp_path: Path
    ) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"
        diag = d / "diagnostics"
        nested = diag / "nested"
        nested.mkdir(parents=True)
        (nested / "attempt_report.json").write_text("{}")

        _restore_artifacts(
            RestoreArtifactsRequest(
                experiment_dir=d,
                journal_backup="# Journal\n",
                experiment_md_backup="# Experiment\n",
                results_backup="[]\n",
                sources_path=sources_path,
                sources_backup="# Research Sources\n",
                pre_cycle_diagnostics=set(),
            )
        )

        assert not diag.exists()

    def test_pre_seeded_diagnostic_survives_retry(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"
        diag = d / "diagnostics"
        diag.mkdir()
        pre_seeded = diag / "baseline_report.json"
        pre_seeded.write_text('{"seeded": true}')
        pre_cycle_diagnostics = set(diag.glob("**/*"))

        # Simulate an attempt writing a new diagnostic file
        (diag / "attempt_artifact.json").write_text("{}")

        _restore_artifacts(
            RestoreArtifactsRequest(
                experiment_dir=d,
                journal_backup="# Journal\n",
                experiment_md_backup="# Experiment\n",
                results_backup="[]\n",
                sources_path=sources_path,
                sources_backup=None,
                pre_cycle_diagnostics=pre_cycle_diagnostics,
            )
        )

        assert pre_seeded.exists(), "pre-seeded diagnostic should survive restore"
        assert not (diag / "attempt_artifact.json").exists()


class TestRestoreArtifactsFailureIsLoud:
    """A failed restore must raise RollbackError, never silently continue."""

    def test_write_failure_raises_rollback_error(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        sources_path = d / "research_sources.md"

        with (
            patch("loop.artifacts.write_text", side_effect=OSError("disk full")),
            pytest.raises(RollbackError, match="Could not restore"),
        ):
            _restore_artifacts(
                RestoreArtifactsRequest(
                    experiment_dir=d,
                    journal_backup="# Journal\n",
                    experiment_md_backup="# Experiment\n",
                    results_backup="[]\n",
                    sources_path=sources_path,
                    sources_backup="# Research Sources\n",
                )
            )
