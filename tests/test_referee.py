from __future__ import annotations

import json
from pathlib import Path

from lib.referee import (
    grade_cycle,
    latest_scorecard_line,
    write_scorecard,
)
from loop.hooks import PostCycleContext, RefereeCycleHooks
from tests.loop.conftest import make_experiment_dir

_EMPTY_SNAPSHOT = {
    "journal_hash": "OLD",
    "sources_hash": "OLD",
    "results_hash": "OLD",
    "candidate_ids": [],
    "results_by_id": {},
}


def _scores(scorecard) -> dict[str, float]:
    return {c.name: c.score for c in scorecard.criteria}


def test_good_cycle_scores_high(tmp_path: Path) -> None:
    journal = (
        "## Cycle 0001: Test logreg-poly\n"
        "**Hypothesis:** polynomial features beat the baseline.\n"
        "Bootstrap 95% CI overlaps the baseline, so the gain is within noise.\n"
    )
    d = make_experiment_dir(
        tmp_path,
        results=[
            {"candidate_id": "logreg-poly", "objective_score": 0.61, "objective_metric": "val_auc"}
        ],
        journal=journal,
    )
    after = {
        **_EMPTY_SNAPSHOT,
        "journal_hash": "NEW",
        "results_hash": "NEW",
        "candidate_ids": ["logreg-poly"],
    }
    card = grade_cycle(d, "0001", before=_EMPTY_SNAPSHOT, after=after, output_text="")

    scores = _scores(card)
    assert scores["journal_updated"] == 1.0
    assert scores["hypothesis_framed"] == 1.0
    assert scores["evidence_or_understanding"] == 1.0
    assert scores["noise_awareness"] == 1.0
    assert card.grade == "A"
    assert card.overall >= 85


def test_new_candidate_without_uncertainty_loses_noise_points(tmp_path: Path) -> None:
    journal = "## Cycle 0001: Try xgboost\n**Hypothesis:** xgboost will win.\nGot 0.70 val auc, shipping it.\n"
    d = make_experiment_dir(
        tmp_path,
        results=[{"candidate_id": "xgb", "objective_score": 0.70, "objective_metric": "val_auc"}],
        journal=journal,
    )
    after = {
        **_EMPTY_SNAPSHOT,
        "journal_hash": "NEW",
        "results_hash": "NEW",
        "candidate_ids": ["xgb"],
    }
    card = grade_cycle(d, "0001", before=_EMPTY_SNAPSHOT, after=after, output_text="")

    assert _scores(card)["noise_awareness"] == 0.0
    assert card.overall < 100


def test_journal_not_updated_is_penalized(tmp_path: Path) -> None:
    d = make_experiment_dir(tmp_path, journal="## Cycle 0001: stub\n")
    # before == after: nothing changed this cycle.
    card = grade_cycle(d, "0001", before=_EMPTY_SNAPSHOT, after=_EMPTY_SNAPSHOT, output_text="")
    scores = _scores(card)
    assert scores["journal_updated"] == 0.0
    assert scores["evidence_or_understanding"] == 0.0
    assert card.grade in {"C", "D"}


def test_write_and_read_scorecard_roundtrip(tmp_path: Path) -> None:
    d = make_experiment_dir(tmp_path, journal="## Cycle 0003: x\n**Hypothesis:** y\n")
    after = {**_EMPTY_SNAPSHOT, "journal_hash": "NEW"}
    card = grade_cycle(d, "0003", before=_EMPTY_SNAPSHOT, after=after, output_text="")

    path = write_scorecard(d, card)
    assert path == d / "cycles" / "0003" / "scorecard.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["cycle_id"] == "0003"
    assert on_disk["overall"] == card.overall

    line = latest_scorecard_line(d)
    assert line is not None
    assert "cycle 0003" in line
    assert f"{card.overall}/100" in line


def test_latest_scorecard_line_none_when_absent(tmp_path: Path) -> None:
    d = make_experiment_dir(tmp_path)
    assert latest_scorecard_line(d) is None


def test_referee_hooks_persist_scorecard(tmp_path: Path) -> None:
    journal = "## Cycle 0001: t\n**Hypothesis:** h\n"
    d = make_experiment_dir(tmp_path, journal=journal)
    before = _EMPTY_SNAPSHOT
    after = {**_EMPTY_SNAPSHOT, "journal_hash": "NEW"}

    hooks = RefereeCycleHooks()
    result = hooks.post_cycle(
        PostCycleContext(
            experiment_dir=d,
            cycle_id="0001",
            before_snapshot=before,
            after_snapshot=after,
            output="**Hypothesis:** h",
            marker="CYCLE_DONE",
        )
    )

    assert result.scorecard is not None
    assert (d / "cycles" / "0001" / "scorecard.json").exists()
    # Default hook behavior (progress reasons) is preserved by the subclass.
    assert result.progress_reasons == ["journal_updated"]
