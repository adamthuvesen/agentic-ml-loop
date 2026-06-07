import json
from pathlib import Path
from unittest.mock import call, patch

from loop import CyclePrompt, cycle_prompt
from tests.loop.conftest import _make_experiment


class TestBuildCyclePrompt:
    def test_returns_cycle_prompt_object(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001")
        assert isinstance(prompt, CyclePrompt)
        assert prompt.static_sections
        assert prompt.dynamic_sections

    def test_static_sections_stable_across_cycles(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "baseline", "objective_score": 0.6}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        prompt_a = cycle_prompt(d, "0002")
        prompt_b = cycle_prompt(d, "0003")
        assert prompt_a.static_sections == prompt_b.static_sections

    def test_dynamic_sections_change_with_results(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt_before = cycle_prompt(d, "0001")
        (d / "results.json").write_text(
            json.dumps([{"candidate_id": "model-a", "objective_score": 0.75}]) + "\n"
        )
        prompt_after = cycle_prompt(d, "0001")
        assert prompt_before.dynamic_sections != prompt_after.dynamic_sections

    def test_includes_minimum_cycles_contract_when_specified(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "experiment.md").write_text(
            "Minimum loop cycles before EXPERIMENT_COMPLETE: 6\n\n# Experiment\n"
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "Minimum cycles contract" in prompt
        assert "**6** completed journal cycles" in prompt

    def test_workspace_has_paths(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "## Your Workspace" in prompt
        assert f"- Experiment: `{d.resolve()}`" in prompt
        assert "- One-shot cycle scripts:" in prompt
        assert "- Long-lived module" in prompt
        assert "before doing fresh research" in prompt

    def test_includes_output_paths_section(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "## Output paths" in prompt
        assert "outputs/" in prompt
        assert "work/" in prompt
        assert "scripts/" in prompt
        assert "lib.paths" in prompt

    def test_output_paths_falls_back_when_program_md_lacks_section(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from loop import prompts as prompts_module

        fake_program = tmp_path / "program.md"
        fake_program.write_text("# Program\n\n## Researcher Identity\n\nBe rigorous.\n")
        monkeypatch.setattr(prompts_module, "PROGRAM_MD_PATH", fake_program)

        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "## Output paths" in prompt
        # Fallback content names the helper API
        assert "outputs_dir" in prompt
        assert "scripts_dir" in prompt

    def test_includes_conditional_sources_update_nudge(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "update `research_sources.md`" in prompt
        assert "Reusable Takeaways" in prompt
        assert "current — rewrite" in prompt

    def test_first_cycle_includes_research_text(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Research Phase Required" in prompt

    def test_includes_advisory_signals_when_diagnostics_report_exists(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        diagnostics_dir = d / "diagnostics"
        diagnostics_dir.mkdir()
        report = {
            "split_comparison": [
                {
                    "split": "validation",
                    "top_feature_drift": [
                        {"column": "age", "delta": 2.5, "scaled_delta": 1.5},
                    ],
                }
            ],
            "missingness": [],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diagnostics_dir / "report.json").write_text(json.dumps(report))
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Advisory Signals" in prompt
        assert "age" in prompt

    def test_includes_advisory_signals_when_evaluation_review_report_exists(
        self, tmp_path: Path
    ) -> None:
        d = _make_experiment(tmp_path)
        review = {
            "concerns": [
                {
                    "kind": "instability",
                    "title": "Leaderboard Instability Concern",
                    "concern": "Noisy leaderboard.",
                    "priority": 80.5,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Advisory Signals" in prompt
        assert "Noisy leaderboard" in prompt

    def test_includes_retrieved_learnings_warm_start(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        with patch(
            "loop.prompts.cross_experiment_learnings_context",
            return_value=(
                "### Warm-Start Note\n\n"
                "- Prior: Start with a simple baseline.\n\n"
                "### Relevant Excerpts\n\n"
                "#### Pattern A\n\n"
                "- Example prior.\n"
            ),
        ):
            prompt = cycle_prompt(d, "0001").assemble()
        assert "Cross-Experiment Learnings" in prompt
        assert "Warm-Start Note" in prompt
        assert "Start with a simple baseline." in prompt

    def test_omits_advisory_signals_when_no_findings(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Advisory Signals" not in prompt

    def test_omits_learnings_when_no_relevant_match(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        with patch(
            "loop.prompts.cross_experiment_learnings_context",
            return_value=None,
        ):
            prompt = cycle_prompt(d, "0001").assemble()
        assert "Cross-Experiment Learnings" not in prompt

    def test_cross_learnings_opt_out_skips_warm_start(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        (d / "experiment.md").write_text("cross_learnings: false\n# Experiment\n")
        with patch(
            "loop.prompts.cross_experiment_learnings_context",
            return_value="### Warm-Start Note\n\n- Prior: Should not appear.\n",
        ):
            prompt = cycle_prompt(d, "0001").assemble()
        assert "Cross-Experiment Learnings" not in prompt

    def test_existing_results_omits_research_text(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.5}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "Research Phase Required" not in prompt

    def test_stall_warning_includes_research_mandate(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.5}],
            journal="# Journal\n\n## Cycle 0001: stuff\n\nDone.\n",
        )
        state = {
            "consecutive_no_progress_cycles": 3,
            "status": "running",
            "cycle_count": 3,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
            "stop_reason": None,
            "last_successful_cycle_id": None,
        }
        (d / "loop_state.json").write_text(json.dumps(state) + "\n")
        prompt = cycle_prompt(d, "0004").assemble()
        assert "Stall Warning" in prompt
        assert "Change Your Approach" in prompt
        assert "Kaggle" in prompt

    def test_research_nudge_when_no_external_sources(self, tmp_path: Path) -> None:
        """Agent has done cycles but never searched the web — signal fires."""
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.5}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        # research_sources.md has only the untouched scaffold placeholder
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Reusable Takeaways\n\n- none\n\n"
            "## Source Cards\n\n### Source 001: <title>\n\n- placeholder\n"
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "Advisory Signals" in prompt
        assert "No external research is recorded" in prompt

    def test_stale_research_signal_with_thin_sources(self, tmp_path: Path) -> None:
        """Several cycles with very thin research should emit a stale-research signal."""
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.5}],
            journal=(
                "# Journal\n\n"
                "## Cycle 0001: baseline\n\nDone.\n\n"
                "## Cycle 0002: tuning\n\nDone.\n\n"
                "## Cycle 0003: review\n\nDone.\n\n"
                "## Cycle 0004: retune\n\nDone.\n"
            ),
        )
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: Kaggle solution\n\n- real research\n"
        )
        prompt = cycle_prompt(d, "0005").assemble()
        assert "Advisory Signals" in prompt
        assert "External research looks thin" in prompt

    def test_analysis_protocol_injected_after_first_cycle(self, tmp_path: Path) -> None:
        """Cycle 2+ should include the analysis protocol."""
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.5}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: Seed\n\n- seeded\n\n"
            "### Source 002: Kaggle\n\n- real\n"
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "Before You Choose What To Do" in prompt
        assert "Analyze results" in prompt
        assert "Consult external knowledge" in prompt
        assert "Form hypothesis" in prompt

    def test_analysis_protocol_not_on_first_cycle(self, tmp_path: Path) -> None:
        """First cycle has no prior results to analyze — skip the protocol."""
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Before You Choose What To Do" not in prompt

    def test_first_cycle_research_fires_with_preseeded_results(self, tmp_path: Path) -> None:
        """Pre-seeded results.json should NOT suppress the first-cycle research gate."""
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "rule-baseline", "objective_score": 0.70}],
            journal="# Journal\n",  # No cycle entries
        )
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Research Phase Required" in prompt


class TestErrorAnalysisNudge:
    def test_nudge_fires_when_candidates_exist_without_error_analysis(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {"candidate_id": "a", "objective_score": 0.7},
                {"candidate_id": "b", "objective_score": 0.72},
                {"candidate_id": "c", "objective_score": 0.75},
            ],
            journal=(
                "# Journal\n\n## Cycle 0001: baseline\n\nTrained LR.\n\n"
                "## Cycle 0002: xgboost\n\nTrained XGB.\n"
            ),
        )
        # Need external sources to avoid research nudge dominating
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0003").assemble()
        assert "Advisory Signals" in prompt
        assert "neither error analysis nor diagnostics are recorded yet" in prompt

    def test_nudge_suppressed_when_error_analysis_done(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {"candidate_id": "a", "objective_score": 0.7},
                {"candidate_id": "b", "objective_score": 0.72},
                {"candidate_id": "c", "objective_score": 0.75},
            ],
            journal=(
                "# Journal\n\n## Cycle 0001: baseline\n\nTrained LR.\n\n"
                "## Cycle 0002: error analysis\n\nExamined false negatives.\n"
            ),
        )
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0003").assemble()
        assert "neither error analysis nor diagnostics are recorded yet" not in prompt

    def test_nudge_does_not_fire_with_few_candidates(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.7}],
            journal=(
                "# Journal\n\n## Cycle 0001: baseline\n\nDone.\n\n## Cycle 0002: tuning\n\nDone.\n"
            ),
        )
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0003").assemble()
        assert "neither error analysis nor diagnostics are recorded yet" not in prompt


class TestCompletionRigorCheck:
    def test_present_after_first_cycle(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.7}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "Before Declaring EXPERIMENT_COMPLETE" in prompt
        assert "meaningfully different model families" in prompt

    def test_absent_on_first_cycle(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "Before Declaring EXPERIMENT_COMPLETE" not in prompt


class TestCyclePrompt:
    def test_assemble_includes_boundary_marker(self) -> None:
        prompt = CyclePrompt(
            static_sections=["# Static", "Always here"],
            dynamic_sections=["# Dynamic", "Changes now"],
        )
        assembled = prompt.assemble()
        assert "# --- dynamic context below ---" in assembled
        assert assembled.index("Always here") < assembled.index("# --- dynamic context below ---")
        assert assembled.index("# --- dynamic context below ---") < assembled.index("# Dynamic")

    def test_estimated_tokens_matches_assembled_length(self) -> None:
        prompt = CyclePrompt(
            static_sections=["a" * 120],
            dynamic_sections=["b" * 80],
        )
        assembled = prompt.assemble()
        assert prompt.estimated_tokens == len(assembled) // 4

    def test_static_hash_stable_and_sensitive_to_changes(self) -> None:
        prompt_a = CyclePrompt(
            static_sections=["one", "two"],
            dynamic_sections=["alpha"],
        )
        prompt_b = CyclePrompt(
            static_sections=["one", "two"],
            dynamic_sections=["beta"],
        )
        prompt_c = CyclePrompt(
            static_sections=["one", "three"],
            dynamic_sections=["alpha"],
        )
        assert prompt_a.static_hash == prompt_b.static_hash
        assert prompt_a.static_hash != prompt_c.static_hash

    def test_progressive_truncation_drops_sections_in_priority_order(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {"candidate_id": f"candidate-{i}", "objective_score": 0.9 - i * 0.01}
                for i in range(5)
            ],
            journal=(
                "# Journal\n\n"
                + "\n\n".join(
                    f"## Cycle {i:04d}: cycle {i}\n\n" + ("very long note " * 400)
                    for i in range(1, 4)
                )
                + "\n"
            ),
        )
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: Seed\n\n- seeded\n\n"
            "### Source 002: Real\n\n- real\n"
        )
        diagnostics_dir = d / "diagnostics"
        diagnostics_dir.mkdir()
        (diagnostics_dir / "report.json").write_text(
            json.dumps(
                {
                    "split_comparison": [
                        {
                            "split": "validation",
                            "top_feature_drift": [
                                {
                                    "column": "very_large_feature_name",
                                    "delta": 9.9,
                                    "scaled_delta": 3.2,
                                }
                            ],
                        }
                    ],
                    "missingness": [],
                    "subgroup_slices": [],
                    "interaction_candidates": [],
                }
            )
        )

        with (
            patch(
                "loop.prompts.cross_experiment_learnings_context",
                return_value="Warm start:\n" + ("prior " * 300),
            ),
            patch("builtins.print") as mock_print,
        ):
            prompt = cycle_prompt(d, "0004", max_tokens=500)

        assembled = prompt.assemble()
        assert "## Cycle 0001:" not in assembled
        assert "## Cycle 0002:" not in assembled
        assert "## Cycle 0003:" in assembled
        assert assembled.count("- `candidate-") == 3
        assert "## Advisory Signals" not in assembled
        assert "## Cross-Experiment Learnings" not in assembled
        assert mock_print.call_args_list[:4] == [
            call("WARNING: prompt over budget; truncating journal entries to 1"),
            call("WARNING: prompt over budget; truncating results to top 3"),
            call("WARNING: prompt over budget; dropping advisory signals"),
            call("WARNING: prompt over budget; dropping cross-experiment learnings"),
        ]


class TestDomainAndRigorInPrompt:
    def test_first_cycle_includes_domain_reasoning(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "business context" in prompt

    def test_analysis_protocol_includes_calibration(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.7}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        (d / "research_sources.md").write_text(
            "# Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "Calibration" in prompt
        assert "Significance" in prompt


class TestPromptLineBudget:
    def test_mature_experiment_within_350_lines(self, tmp_path: Path) -> None:
        """A mature experiment with results, journal, and advisory signals fits in 350 lines."""
        journal_entries = "\n\n".join(
            f"## Cycle {i:04d}: cycle {i}\n\nTested something.\n\n**Hypothesis:** Try approach {i}"
            for i in range(1, 6)
        )
        results = [
            {
                "candidate_id": f"model-{i}",
                "objective_score": 0.7 + i * 0.01,
                "objective_metric": "val_auc",
                "metrics": {
                    "train": {"auc": 0.85 + i * 0.01},
                    "validation": {"auc": 0.7 + i * 0.01},
                },
            }
            for i in range(1, 6)
        ]
        d = _make_experiment(
            tmp_path,
            results=results,
            journal=f"# Journal\n\n{journal_entries}\n",
        )
        # Add diagnostics report for advisory signals
        diagnostics_dir = d / "diagnostics"
        diagnostics_dir.mkdir()
        report = {
            "split_comparison": [
                {
                    "split": "validation",
                    "top_feature_drift": [
                        {"column": "age", "delta": 2.5, "scaled_delta": 1.5},
                    ],
                }
            ],
            "missingness": [],
            "subgroup_slices": [],
            "interaction_candidates": [],
        }
        (diagnostics_dir / "report.json").write_text(json.dumps(report))
        # Add evaluation review for advisory signals
        review = {
            "concerns": [
                {
                    "kind": "instability",
                    "title": "Concern",
                    "concern": "Noisy leaderboard.",
                    "priority": 80.5,
                }
            ]
        }
        (d / "evaluation_review.json").write_text(json.dumps(review))
        # Sources with some real research
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n## Source Cards\n\n"
            "### Source 001: Seed\n\n- seeded\n\n"
            "### Source 002: Kaggle\n\n- real\n"
        )

        with patch(
            "loop.prompts.cross_experiment_learnings_context",
            return_value=(
                "### Warm-Start Note\n\n"
                "- Prior: Start with a simple baseline.\n\n"
                "### Relevant Excerpts\n\n"
                "#### Pattern A\n\n"
                "- Example prior.\n"
            ),
        ):
            prompt = cycle_prompt(d, "0006").assemble()

        line_count = len(prompt.splitlines())
        assert line_count <= 350, (
            f"Prompt is {line_count} lines, expected ≤350. Prompt starts with: {prompt[:200]!r}"
        )

    def test_first_cycle_shorter_than_mature(self, tmp_path: Path) -> None:
        """First cycle with minimal context should be shorter than the mature case."""
        (tmp_path / "first").mkdir()
        d_first = _make_experiment(tmp_path / "first")
        prompt_first = cycle_prompt(d_first, "0001").assemble()

        (tmp_path / "mature").mkdir()
        d_mature = _make_experiment(
            tmp_path / "mature",
            results=[{"candidate_id": "a", "objective_score": 0.7}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        prompt_mature = cycle_prompt(d_mature, "0002").assemble()

        assert len(prompt_first.splitlines()) < len(prompt_mature.splitlines())


class TestAdvisorySignalsInPrompt:
    def test_prompt_includes_signal_block_when_signals_exist(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[
                {
                    "candidate_id": "a",
                    "objective_score": 0.701,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.68, 0.72]},
                },
                {
                    "candidate_id": "b",
                    "objective_score": 0.695,
                    "objective_metric": "val_auc",
                    "hyperparameters": {"val_auc_ci_95": [0.67, 0.71]},
                },
            ],
            journal="# Journal\n\n## Cycle 0001: compare\n\nDone.\n",
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "## Advisory Signals" in prompt
        assert "Advisory only" in prompt

    def test_prompt_omits_signal_block_when_no_signal_evidence(self, tmp_path: Path) -> None:
        d = _make_experiment(
            tmp_path,
            results=[{"candidate_id": "a", "objective_score": 0.5}],
            journal="# Journal\n\n## Cycle 0001: baseline\n\nDone.\n",
        )
        (d / "research_sources.md").write_text(
            "# Research Sources\n\n### Source 001: Seed\n\n- s\n\n### Source 002: Real\n\n- r\n"
        )
        prompt = cycle_prompt(d, "0002").assemble()
        assert "## Advisory Signals" not in prompt
