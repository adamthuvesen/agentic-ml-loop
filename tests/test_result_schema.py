from __future__ import annotations

from lib.result_schema import (
    has_minimum_result_fields,
    validate_result_entries,
    validate_result_entry,
)


def test_validate_result_entry_reports_required_fields() -> None:
    messages = validate_result_entry({}, "results.json[0]", strict_completion=True)

    assert "results.json[0] requires non-empty string candidate_id" in messages
    assert "results.json[0] requires finite numeric objective_score" in messages
    assert "results.json[0] requires non-empty string objective_metric" in messages


def test_validate_result_entry_warns_on_missing_metric_outside_strict_mode() -> None:
    messages = validate_result_entry(
        {"candidate_id": "a", "objective_score": 0.5},
        "results.json[0]",
        strict_completion=False,
    )

    assert messages == [
        "warning: results.json[0] requires non-empty string objective_metric"
    ]


def test_validate_result_entry_reports_nested_non_finite_numbers() -> None:
    messages = validate_result_entry(
        {
            "candidate_id": "a",
            "objective_score": 0.5,
            "objective_metric": "val_auc",
            "metrics": {"validation": {"auc": float("nan")}},
        },
        "results.json[0]",
    )

    assert (
        "results.json[0].metrics.validation.auc must be finite JSON number" in messages
    )


def test_validate_result_entries_requires_one_valid_entry_in_strict_mode() -> None:
    messages = validate_result_entries([{"foo": 1}], strict_completion=True)

    assert "strict completion requires at least one valid result entry" in messages


def test_has_minimum_result_fields_does_not_require_metric() -> None:
    assert has_minimum_result_fields({"candidate_id": "a", "objective_score": 0.5})
