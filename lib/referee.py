"""Advisory research referee: grade a single cycle's scientific conduct.

The referee is deterministic and **advisory** — it never blocks the loop. It
aggregates the existing signal producers (:mod:`lib.signals`,
:mod:`lib.evaluation_review`, :mod:`lib.diagnostics`) plus a few lightweight
per-cycle checks into a transparent scorecard, so a human (or a benchmark
comparing runners) can see whether the agent behaved like a scientist this
cycle: framed a hypothesis, logged evidence, respected uncertainty, and didn't
ignore leakage or skip error analysis.

The rubric is intentionally small. Per ``program.md``: "Add new process rules
only when they prevent a real failure mode. A huge prompt full of unaudited
rules is not discipline; it is fog with bullet points."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment import journal_path
from lib.signals import (
    build_advisory_signals,
    journal_mentions_error_analysis,
)
from lib.utils import write_json

# Keyword sets for the lightweight journal/narrative checks.
_HYPOTHESIS_TERMS = (
    "hypothesis",
    "expect",
    "predict",
    "success criterion",
    "falsif",
    "we think",
    "the idea is",
)
_UNCERTAINTY_TERMS = (
    "confidence interval",
    "ci_95",
    "ci 95",
    "95% ci",
    "bootstrap",
    "within noise",
    "within the noise",
    "not significant",
    "significan",  # significant / significance
    "uncertaint",
    "std",
    "standard deviation",
    "overlap",
)
# Advisory-signal text fragments that indicate an unaddressed leakage/split risk.
_LEAKAGE_SPLIT_TERMS = ("leakage", "split reliability", "split may be", "target rate differs")
_MISSING_ANALYSIS_TERMS = ("neither error analysis nor", "inspect failures")

_GRADE_BANDS = ((85, "A"), (70, "B"), (55, "C"), (0, "D"))


@dataclass
class CriterionScore:
    """One rubric line: a 0.0 / 0.5 / 1.0 score with a short rationale."""

    name: str
    score: float
    note: str


@dataclass
class CycleScorecard:
    """Advisory grade for one cycle's scientific conduct (0-100 + letter)."""

    cycle_id: str
    overall: int
    grade: str
    criteria: list[CriterionScore]
    signals: list[str]

    def summary_line(self) -> str:
        """One-line summary for status output and logs."""
        return f"Referee: cycle {self.cycle_id} scored {self.overall}/100 (grade {self.grade})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "overall": self.overall,
            "grade": self.grade,
            "criteria": [{"name": c.name, "score": c.score, "note": c.note} for c in self.criteria],
            "signals": self.signals,
        }


def _grade_for(score: int) -> str:
    for threshold, letter in _GRADE_BANDS:
        if score >= threshold:
            return letter
    return "D"


def current_cycle_journal(experiment_dir: Path, cycle_id: str) -> str:
    """Return the journal section for ``cycle_id``, or the last section.

    Matches ``## Cycle NNNN:`` headings; falls back to the most recent entry so
    a differently-numbered heading still gets graded.
    """
    path = journal_path(experiment_dir)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?=^## Cycle \d{4}:)", text, flags=re.MULTILINE)
    sections = [p.strip() for p in parts if re.match(r"^## Cycle \d{4}:", p.strip())]
    if not sections:
        return ""
    for section in sections:
        if re.match(rf"^## Cycle 0*{int(cycle_id):d}\b", section) or section.startswith(
            f"## Cycle {cycle_id}:"
        ):
            return section
    return sections[-1]


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def grade_cycle(
    experiment_dir: Path,
    cycle_id: str,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    output_text: str = "",
) -> CycleScorecard:
    """Grade one cycle from artifact deltas + existing advisory signals.

    ``before``/``after`` are :class:`loop.artifacts.ArtifactSnapshot` dicts (the
    referee only reads their keys, so it stays free of any ``loop`` import).
    """
    journal_changed = before.get("journal_hash") != after.get("journal_hash")
    results_changed = before.get("results_hash") != after.get("results_hash")
    new_candidates = sorted(
        set(after.get("candidate_ids", [])) - set(before.get("candidate_ids", []))
    )

    narrative = f"{output_text}\n{current_cycle_journal(experiment_dir, cycle_id)}".lower()
    advisory = build_advisory_signals(experiment_dir)
    signal_texts = [text for _, text in advisory]
    signals_blob = " ".join(signal_texts).lower()

    criteria: list[CriterionScore] = []

    # 1. Evidence discipline: the journal must record the cycle.
    criteria.append(
        CriterionScore(
            "journal_updated",
            1.0 if journal_changed else 0.0,
            "journal entry recorded" if journal_changed else "journal not updated this cycle",
        )
    )

    # 2. Hypothesis framing.
    if _has_any(narrative, _HYPOTHESIS_TERMS):
        criteria.append(CriterionScore("hypothesis_framed", 1.0, "hypothesis/expectation stated"))
    elif journal_changed:
        criteria.append(
            CriterionScore("hypothesis_framed", 0.5, "journal updated but no explicit hypothesis")
        )
    else:
        criteria.append(CriterionScore("hypothesis_framed", 0.0, "no hypothesis found"))

    # 3. Produced evidence or understanding.
    if results_changed or new_candidates:
        note = (
            f"new candidates: {', '.join(new_candidates)}" if new_candidates else "results updated"
        )
        criteria.append(CriterionScore("evidence_or_understanding", 1.0, note))
    elif journal_changed and len(narrative) > 400:
        criteria.append(
            CriterionScore(
                "evidence_or_understanding", 0.5, "understanding-only cycle (no results)"
            )
        )
    else:
        criteria.append(
            CriterionScore("evidence_or_understanding", 0.0, "no new results or substantive notes")
        )

    # 4. Noise awareness when results moved.
    if not (results_changed or new_candidates):
        criteria.append(
            CriterionScore("noise_awareness", 1.0, "no new candidate to judge against noise")
        )
    elif _has_any(narrative, _UNCERTAINTY_TERMS):
        criteria.append(CriterionScore("noise_awareness", 1.0, "uncertainty/CI discussed"))
    else:
        criteria.append(
            CriterionScore(
                "noise_awareness",
                0.0,
                "new candidates without any uncertainty / CI / significance discussion",
            )
        )

    # 5. Leakage / split reliability addressed (penalize an unaddressed concern).
    leakage_flagged = _has_any(signals_blob, _LEAKAGE_SPLIT_TERMS)
    if not leakage_flagged:
        criteria.append(CriterionScore("leakage_split_clean", 1.0, "no leakage/split concern"))
    elif _has_any(narrative, ("leak", "split", "drift")):
        criteria.append(
            CriterionScore("leakage_split_clean", 0.5, "concern raised and acknowledged in journal")
        )
    else:
        criteria.append(
            CriterionScore(
                "leakage_split_clean", 0.0, "leakage/split concern flagged but not addressed"
            )
        )

    # 6. Error analysis when saturating (advisory "missing analysis" signal fired).
    missing_analysis = _has_any(signals_blob, _MISSING_ANALYSIS_TERMS)
    if not missing_analysis or journal_mentions_error_analysis(experiment_dir):
        criteria.append(CriterionScore("analysis_when_saturating", 1.0, "error analysis present"))
    else:
        criteria.append(
            CriterionScore(
                "analysis_when_saturating",
                0.0,
                "candidates accumulating without recorded error analysis",
            )
        )

    overall = round(100 * sum(c.score for c in criteria) / len(criteria))
    return CycleScorecard(
        cycle_id=cycle_id,
        overall=overall,
        grade=_grade_for(overall),
        criteria=criteria,
        signals=signal_texts,
    )


def scorecard_path(experiment_dir: Path, cycle_id: str) -> Path:
    """Path to a cycle's scorecard JSON under ``cycles/<id>/``."""
    return experiment_dir / "cycles" / cycle_id / "scorecard.json"


def write_scorecard(experiment_dir: Path, scorecard: CycleScorecard) -> Path:
    """Persist a scorecard as JSON and return its path."""
    path = scorecard_path(experiment_dir, scorecard.cycle_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, scorecard.to_dict())
    return path


def latest_scorecard_line(experiment_dir: Path) -> str | None:
    """Return the most recent cycle's referee summary line, if any."""
    cycles_dir = experiment_dir / "cycles"
    if not cycles_dir.is_dir():
        return None
    scorecards = sorted(cycles_dir.glob("*/scorecard.json"))
    if not scorecards:
        return None
    from lib.utils import load_json

    try:
        data = load_json(scorecards[-1])
    except (OSError, ValueError):
        return None
    cycle_id = str(data.get("cycle_id", "?"))
    overall = data.get("overall", "?")
    grade = data.get("grade", "?")
    return f"Referee: cycle {cycle_id} scored {overall}/100 (grade {grade})"
