import json
from pathlib import Path
from unittest.mock import patch

from loop import (
    DefaultCycleHooks,
    PostCycleContext,
    PostCycleResult,
    artifact_snapshot,
    cycle_prompt,
    journal_mentions_error_analysis,
)
from loop.hooks import call_post_cycle_hook
from tests.loop.conftest import _make_experiment


class TestDefaultCycleHooks:
    def test_pre_cycle_matches_cycle_prompt(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        hooks = DefaultCycleHooks()

        result = hooks.pre_cycle(d, "0001", {"cycle_count": 0})

        assert result.prompt_text == cycle_prompt(d, "0001").assemble()
        assert result.prompt_text == result.cycle_prompt.assemble()

    def test_post_cycle_computes_progress_from_snapshots(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        hooks = DefaultCycleHooks()
        before = artifact_snapshot(d)
        (d / "results.json").write_text(
            json.dumps([{"candidate_id": "new-model", "objective_score": 0.7}]) + "\n"
        )
        after = artifact_snapshot(d)

        result = hooks.post_cycle(
            PostCycleContext(
                experiment_dir=d,
                cycle_id="0001",
                before_snapshot=before,
                after_snapshot=after,
                output="",
                marker="CYCLE_DONE",
            )
        )

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
                PostCycleContext(
                    experiment_dir=d,
                    cycle_id="0001",
                    before_snapshot=snapshot,
                    after_snapshot=snapshot,
                    output="<promise>EXPERIMENT_COMPLETE</promise>",
                    marker="EXPERIMENT_COMPLETE",
                )
            )

        assert result.learnings_extracted is True

    def test_post_cycle_adapter_passes_context_to_custom_hook(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        snapshot = artifact_snapshot(d)

        class CustomHooks:
            def post_cycle(self, context: PostCycleContext) -> PostCycleResult:
                assert context.experiment_dir == d
                assert context.before_snapshot == snapshot
                assert context.after_snapshot == snapshot
                assert context.output == ""
                assert context.marker == "CYCLE_DONE"
                return PostCycleResult(progress_reasons=[f"custom:{context.cycle_id}"])

        result = call_post_cycle_hook(
            CustomHooks(),
            PostCycleContext(
                experiment_dir=d,
                cycle_id="0001",
                before_snapshot=snapshot,
                after_snapshot=snapshot,
                output="",
                marker="CYCLE_DONE",
            ),
        )

        assert result.progress_reasons == ["custom:0001"]


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
