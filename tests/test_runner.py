from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lib.runner import save_candidate_result


class DummyCandidate:
    candidate_id = "candidate-a"

    def result_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "objective_metric": "val_auc",
            "objective_score": 0.7,
        }


def _candidate_runner(_splits: object) -> DummyCandidate:
    return DummyCandidate()


class BadCandidate:
    def result_payload(self) -> dict[str, Any]:
        return {
            "candidate_id": "bad",
            "objective_metric": "val_auc",
            "objective_score": "0.7",
        }


def _bad_candidate_runner(_splits: object) -> BadCandidate:
    return BadCandidate()


def test_save_candidate_result_rejects_missing_experiment_dir(tmp_path: Path) -> None:
    calls: list[str] = []

    with pytest.raises(FileNotFoundError, match="Experiment directory"):
        save_candidate_result(
            tmp_path / "missing",
            "candidate-a",
            lambda: calls.append("loaded"),
            {"candidate-a": _candidate_runner},
        )

    assert calls == []


def test_save_candidate_result_rejects_unknown_candidate_before_loading_data(
    tmp_path: Path,
) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    calls: list[str] = []

    with pytest.raises(ValueError, match="Unknown candidate 'missing'"):
        save_candidate_result(
            exp_dir,
            "missing",
            lambda: calls.append("loaded"),
            {"candidate-a": _candidate_runner},
        )

    assert calls == []


def test_save_candidate_result_rejects_malformed_results_before_loading_data(
    tmp_path: Path,
) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text('{"candidate_id": "old"}\n')
    calls: list[str] = []

    with pytest.raises(ValueError, match="JSON list of result objects"):
        save_candidate_result(
            exp_dir,
            "candidate-a",
            lambda: calls.append("loaded"),
            {"candidate-a": _candidate_runner},
        )

    assert calls == []


@pytest.mark.parametrize(
    ("results_json", "message"),
    [
        ('[{"objective_score": 0.5}]\n', "candidate_id"),
        ('[{"candidate_id": "old", "objective_score": "0.5"}]\n', "objective_score"),
        ("[1]\n", "must be an object"),
    ],
)
def test_save_candidate_result_rejects_invalid_result_entries_before_loading_data(
    tmp_path: Path,
    results_json: str,
    message: str,
) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text(results_json)
    calls: list[str] = []

    with pytest.raises(ValueError, match=message):
        save_candidate_result(
            exp_dir,
            "candidate-a",
            lambda: calls.append("loaded"),
            {"candidate-a": _candidate_runner},
        )

    assert calls == []


def test_save_candidate_result_rejects_invalid_candidate_payload(
    tmp_path: Path,
) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()

    with pytest.raises(ValueError, match="objective_score"):
        save_candidate_result(
            exp_dir,
            "bad",
            lambda: object(),
            {"bad": _bad_candidate_runner},
        )

    assert not (exp_dir / "results.json").exists()


def test_save_candidate_result_replaces_existing_candidate_result(
    tmp_path: Path,
) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text(
        '[{"candidate_id": "candidate-a", "objective_score": 0.1}, '
        '{"candidate_id": "candidate-b", "objective_score": 0.9}]\n'
    )

    payload = save_candidate_result(
        exp_dir,
        "candidate-a",
        lambda: object(),
        {"candidate-a": _candidate_runner},
    )

    assert payload["objective_score"] == 0.7
    assert (exp_dir / "results.json.lock").exists()
    results = json.loads((exp_dir / "results.json").read_text())
    assert [entry["candidate_id"] for entry in results] == [
        "candidate-b",
        "candidate-a",
    ]
    assert results[1]["objective_score"] == 0.7


def test_save_candidate_result_rejects_writes_after_selection_freeze(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "results.json").write_text(
        '[{"candidate_id": "candidate-a", "objective_score": 0.6}]\n'
    )
    (exp_dir / "loop_state.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp",
                "status": "completed",
                "selection_frozen": True,
                "frozen_candidate_ids": ["candidate-a"],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="Selection is frozen"):
        save_candidate_result(
            exp_dir,
            "candidate-a",
            lambda: object(),
            {"candidate-a": _candidate_runner},
        )

    results = json.loads((exp_dir / "results.json").read_text())
    assert results == [{"candidate_id": "candidate-a", "objective_score": 0.6}]
