from loop.core import cycle_contract_errors, extract_completion_marker


class TestExtractCompletionMarker:
    def test_extracts_single_valid_marker(self) -> None:
        marker, errors = extract_completion_marker("Done.\n<promise>CYCLE_DONE</promise>\n")

        assert marker == "CYCLE_DONE"
        assert errors == []

    def test_reports_missing_marker(self) -> None:
        marker, errors = extract_completion_marker("Done.")

        assert marker == ""
        assert errors == ["missing completion marker"]

    def test_rejects_multiple_markers(self) -> None:
        marker, errors = extract_completion_marker(
            "<promise>CYCLE_DONE</promise>\nActually done.\n<promise>EXPERIMENT_COMPLETE</promise>"
        )

        assert marker == ""
        assert errors == ["expected exactly one completion marker, found 2"]

    def test_rejects_unknown_marker(self) -> None:
        marker, errors = extract_completion_marker("<promise>DONE</promise>")

        assert marker == "DONE"
        assert errors == ["unknown completion marker: DONE"]


class TestCycleContractErrors:
    def test_collects_specific_contract_failures(self) -> None:
        errors = cycle_contract_errors(
            returncode=2,
            marker_errors=["missing completion marker"],
            validation_errors=[
                "warning: stray files in experiment root (x.csv)",
                "results.json[0] requires finite numeric objective_score",
            ],
            journal_updated=False,
            experiment_md_changed=True,
        )

        assert errors == [
            "runner exited with return code 2",
            "missing completion marker",
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
            marker_errors=["missing completion marker"],
            validation_errors=["missing completion marker"],
            journal_updated=True,
            experiment_md_changed=False,
        )

        assert errors == ["missing completion marker"]
