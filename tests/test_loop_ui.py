from pathlib import Path
from unittest.mock import patch

from loop import (
    LOCK_PATH_NAME,
    RESEARCHER_FRAMING,
    cycle_prompt,
    format_elapsed,
    latest_hypothesis,
    load_researcher_identity,
    write_status_markdown,
)
from tests.loop.conftest import _make_experiment


class TestFormatElapsed:
    def test_seconds(self) -> None:
        assert format_elapsed(45) == "45s"

    def test_minutes_seconds(self) -> None:
        assert format_elapsed(125) == "2m 5s"

    def test_hours_minutes(self) -> None:
        assert format_elapsed(3725) == "1h 2m"


class TestLatestHypothesis:
    def test_extracts_hypothesis_line(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Exp\n")
        (d / "research_journal.md").write_text(
            "# Journal\n\n## Cycle 0001: Test cycle\n\n**Hypothesis:** XGBoost beats logistic regression\n"
        )
        assert latest_hypothesis(d) == "XGBoost beats logistic regression"

    def test_falls_back_to_heading(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Exp\n")
        (d / "research_journal.md").write_text(
            "# Journal\n\n## Cycle 0001: Feature engineering pass\n\nSome notes.\n"
        )
        assert latest_hypothesis(d) == "Feature engineering pass"

    def test_empty_journal(self, tmp_path: Path) -> None:
        d = tmp_path / "exp"
        d.mkdir()
        (d / "experiment.md").write_text("# Exp\n")
        (d / "research_journal.md").write_text("# Journal\n")
        assert latest_hypothesis(d) == ""


class TestStatusMarkdown:
    def test_includes_runner_identity_from_state(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = {
            "status": "idle",
            "cycle_count": 0,
            "last_successful_cycle_id": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "runner_name": "claude",
            "runner_model": "claude-opus-4-7[1m]",
            "runner_effort": "xhigh",
            "stop_reason": None,
        }

        write_status_markdown(d, state)

        status = (d / "status.md").read_text()
        assert "claude (claude-opus-4-7[1m], effort=xhigh)" in status

    def test_includes_active_attempt_details(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path, results=[{"candidate_id": "a", "objective_score": 0.5}])
        state = {
            "status": "running",
            "cycle_count": 1,
            "last_successful_cycle_id": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": None,
            "active_cycle_id": "0002",
            "active_started_at": "2025-01-01T00:00:20+00:00",
            "active_attempt": 2,
        }
        write_status_markdown(d, state)
        status = (d / "status.md").read_text()
        assert "Active cycle" in status
        assert "0002" in status
        assert "Active attempt" in status
        assert "2/3" in status
        assert "Active started at" in status
        assert "Active for" in status

    def test_marks_running_state_without_lock_as_interrupted(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        state = {
            "status": "running",
            "cycle_count": 1,
            "last_successful_cycle_id": None,
            "started_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:30+00:00",
            "stop_reason": None,
            "active_cycle_id": "0001",
            "active_started_at": "2025-01-01T00:00:05+00:00",
            "active_attempt": 1,
        }
        assert not (d / LOCK_PATH_NAME).exists()
        write_status_markdown(d, state)
        status = (d / "status.md").read_text()
        assert "may have been interrupted" in status


class TestLoadResearcherIdentity:
    """Tests for load_researcher_identity() — extracts identity from program.md."""

    def test_extracts_section_from_program_md(self, tmp_path: Path) -> None:
        program = tmp_path / "program.md"
        program.write_text(
            "# Autonomous ML Research Program\n\n"
            "## Researcher Identity\n\n"
            "You are a scientist.\n"
            "Form hypotheses and test them.\n\n"
            "## Research Before Code\n\n"
            "Do research first.\n"
        )
        with patch("loop.prompts.PROGRAM_MD_PATH", program):
            result = load_researcher_identity()
        assert "You are a scientist." in result
        assert "Form hypotheses and test them." in result
        assert "Do research first." not in result

    def test_falls_back_when_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "program.md"
        with patch("loop.prompts.PROGRAM_MD_PATH", missing):
            result = load_researcher_identity()
        assert result == RESEARCHER_FRAMING

    def test_falls_back_when_section_missing(self, tmp_path: Path) -> None:
        program = tmp_path / "program.md"
        program.write_text(
            "# Autonomous ML Research Program\n\n## Research Before Code\n\nDo research first.\n"
        )
        with patch("loop.prompts.PROGRAM_MD_PATH", program):
            result = load_researcher_identity()
        assert result == RESEARCHER_FRAMING

    def test_caps_at_fifteen_lines(self, tmp_path: Path) -> None:
        lines = [f"Line {i}" for i in range(1, 25)]
        program = tmp_path / "program.md"
        program.write_text(
            "# Autonomous ML Research Program\n\n"
            "## Researcher Identity\n\n" + "\n".join(lines) + "\n\n## Next Section\n"
        )
        with patch("loop.prompts.PROGRAM_MD_PATH", program):
            result = load_researcher_identity()
        assert "Line 15" in result
        assert "Line 16" not in result

    def test_prompt_assembly_uses_identity(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        program = tmp_path / "program.md"
        program.write_text(
            "# Autonomous ML Research Program\n\n"
            "## Researcher Identity\n\n"
            "You are a rigorous scientist.\n\n"
            "## Research Before Code\n\n"
            "Research first.\n"
        )
        with patch("loop.prompts.PROGRAM_MD_PATH", program):
            prompt = cycle_prompt(d, "0001").assemble()
        assert "You are a rigorous scientist." in prompt
        assert "ML researcher running an autonomous experiment" not in prompt


class TestSummarizeGuidelines:
    def test_returns_summary_when_guidelines_exist(self, tmp_path: Path) -> None:
        from loop.prompts import summarize_guidelines

        guidelines = tmp_path / "guidelines.md"
        guidelines.write_text(
            "# Guidelines\n\n"
            "1. **`objective_score` = validation metric.**\n"
            "7. Report uncertainty (bootstrap CI) before claiming a winner.\n"
        )
        with patch("loop.prompts.GUIDELINES_PATH", guidelines):
            result = summarize_guidelines()
        assert "## Guidelines" in result
        assert "objective_score" in result
        assert "bootstrap CI" in result
        assert "Full rules:" in result

    def test_returns_empty_when_missing(self, tmp_path: Path) -> None:
        from loop.prompts import summarize_guidelines

        missing = tmp_path / "guidelines.md"
        with patch("loop.prompts.GUIDELINES_PATH", missing):
            result = summarize_guidelines()
        assert result == ""

    def test_prompt_includes_guidelines_summary(self, tmp_path: Path) -> None:
        d = _make_experiment(tmp_path)
        prompt = cycle_prompt(d, "0001").assemble()
        assert "## Guidelines" in prompt
        assert "objective_score" in prompt
