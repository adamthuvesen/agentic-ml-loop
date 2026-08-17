from pathlib import Path

from loop.contracts import cycle_contract_errors, read_completion_marker


def _status_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cycle_status.json"
    path.write_text(body, encoding="utf-8")
    return path


class TestReadCompletionMarker:
    def test_reads_a_valid_status(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(_status_file(tmp_path, '{"status": "CYCLE_DONE"}'))

        assert marker == "CYCLE_DONE"
        assert errors == []

    def test_ignores_extra_keys(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(
            _status_file(tmp_path, '{"status": "EXPERIMENT_COMPLETE", "why": "space exhausted"}')
        )

        assert marker == "EXPERIMENT_COMPLETE"
        assert errors == []

    def test_reports_a_missing_file(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(tmp_path / "cycle_status.json")

        assert marker == ""
        assert errors == ["missing completion status file: cycle_status.json"]

    def test_reports_unparseable_json(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(_status_file(tmp_path, "{oops"))

        assert marker == ""
        assert errors[0].startswith("cycle_status.json is not readable JSON:")

    def test_rejects_a_non_object_payload(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(_status_file(tmp_path, '"CYCLE_DONE"'))

        assert marker == ""
        assert errors == ["cycle_status.json must hold a JSON object"]

    def test_rejects_a_blank_status(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(_status_file(tmp_path, '{"status": "  "}'))

        assert marker == ""
        assert errors == ["cycle_status.json needs a non-empty `status`"]

    def test_rejects_an_unknown_status(self, tmp_path: Path) -> None:
        marker, errors = read_completion_marker(_status_file(tmp_path, '{"status": "DONE"}'))

        assert marker == "DONE"
        assert errors == ["unknown completion marker: DONE"]


class TestCycleContractErrors:
    def test_collects_specific_contract_failures(self) -> None:
        errors = cycle_contract_errors(
            returncode=2,
            marker_errors=["missing completion status file: cycle_status.json"],
            validation_errors=[
                "warning: stray files in experiment root (x.csv)",
                "results.json[0] requires finite numeric objective_score",
            ],
            journal_updated=False,
            experiment_md_changed=True,
        )

        assert errors == [
            "runner exited with return code 2",
            "missing completion status file: cycle_status.json",
            "results.json[0] requires finite numeric objective_score",
            "research_journal.md was not updated",
            "experiment.md changed during the cycle",
        ]

    def test_ignores_warnings(self) -> None:
        errors = cycle_contract_errors(
            returncode=0,
            marker_errors=[],
            validation_errors=["warning: missing optional metric"],
            journal_updated=True,
            experiment_md_changed=False,
        )

        assert errors == []

    def test_deduplicates_errors(self) -> None:
        errors = cycle_contract_errors(
            returncode=0,
            marker_errors=["missing completion status file: cycle_status.json"],
            validation_errors=["missing completion status file: cycle_status.json"],
            journal_updated=True,
            experiment_md_changed=False,
        )

        assert errors == ["missing completion status file: cycle_status.json"]
