import json
from pathlib import Path
from unittest.mock import patch

from loop import (
    DefaultCycleHooks,
    artifact_snapshot,
    build_cycle_prompt,
    journal_mentions_error_analysis,
)
from tests.loop.conftest import _make_experiment


class TestDefaultCycleHooks:
    def test_pre_cycle_matches_build_cycle_prompt(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        hooks = DefaultCycleHooks()

        result = hooks.pre_cycle(d, "0001", {"cycle_count": 0})

        assert result.prompt_text == build_cycle_prompt(d, "0001").assemble()
        assert result.prompt_text == result.cycle_prompt.assemble()

    def test_post_cycle_computes_progress_from_snapshots(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        hooks = DefaultCycleHooks()
        before = artifact_snapshot(d)
        (d / "results.json").write_text(
            json.dumps([{"candidate_id": "new-model", "objective_score": 0.7}]) + "\n"
        )
        after = artifact_snapshot(d)

        result = hooks.post_cycle(d, "0001", before, after, "", "CYCLE_DONE")

        assert result.progress_reasons == ["new_candidates:new-model"]
        assert result.learnings_extracted is False

    def test_post_cycle_extracts_learnings_on_completion(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        hooks = DefaultCycleHooks()
        snapshot = artifact_snapshot(d)

        with (
            patch("loop.hooks.get_cross_learnings_enabled", return_value=True),
            patch("loop.hooks.extract_and_append_learnings", return_value=True),
        ):
            result = hooks.post_cycle(
                d,
                "0001",
                snapshot,
                snapshot,
                "<promise>EXPERIMENT_COMPLETE</promise>",
                "EXPERIMENT_COMPLETE",
            )

        assert result.learnings_extracted is True


class TestJournalMentionsErrorAnalysis:
    def test_detects_false_negative_mention(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            journal="# Journal\n\n## Cycle 0001: review\n\nLooked at false negatives.\n",
        )
        assert journal_mentions_error_analysis(d) is True

    def test_detects_error_analysis_heading(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            journal="# Journal\n\n## Cycle 0001: error analysis\n\nDid error analysis.\n",
        )
        assert journal_mentions_error_analysis(d) is True

    def test_no_match_on_generic_journal(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            journal="# Journal\n\n## Cycle 0001: baseline\n\nTrained model.\n",
        )
        assert journal_mentions_error_analysis(d) is False
