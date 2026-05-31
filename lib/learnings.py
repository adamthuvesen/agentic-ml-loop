from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from experiment import journal_path, learnings_file
from lib.utils import read_text, write_text

ROOT = Path(__file__).resolve().parent.parent

SECTION_RE = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)
_FROM_EXPERIMENT_RE = re.compile(r"^From\s+`?(?P<eid>[^`(\s]+)`?\s*", re.IGNORECASE)
MAX_RETRIEVED_EXCERPTS = 3
MAX_EXCERPT_BULLETS = 2
MIN_RELEVANCE_SCORE = 3.0


def replace_or_append_learnings(
    learnings_path: Path, experiment_id: str, new_section: str
) -> None:
    """Replace any existing ``## From <experiment_id>`` sections, then append *new_section*.

    Preserves the file header (everything before the first ``## `` heading) and
    any sections belonging to other experiments.  If the file doesn't exist yet
    it is created with *new_section* as the only content.
    """
    if not learnings_path.exists():
        write_text(learnings_path, new_section)
        return

    text = read_text(learnings_path)
    matches = list(SECTION_RE.finditer(text))

    if not matches:
        # No sections at all — just append.
        write_text(learnings_path, text.rstrip("\n") + "\n" + new_section)
        return

    header = text[: matches[0].start()]
    kept_sections: list[str] = []

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        title = match.group("title").strip()
        m = _FROM_EXPERIMENT_RE.match(title)
        if m and m.group("eid").lower() == experiment_id.lower():
            continue  # drop this section
        kept_sections.append(text[start:end])

    result = header.rstrip("\n")
    if kept_sections:
        result += "\n" + "".join(kept_sections).rstrip("\n")
    result += "\n" + new_section
    write_text(learnings_path, result)


@dataclass(frozen=True)
class ExperimentProfile:
    experiment_id: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class LearningsExcerpt:
    title: str
    body: str
    bullets: tuple[str, ...]
    explicit_tags: tuple[str, ...]
    inferred_tags: tuple[str, ...]
    matched_tags: tuple[str, ...]
    score: float


def build_experiment_profile(experiment_dir: Path) -> ExperimentProfile:
    experiment_path = experiment_dir / "experiment.md"
    if not experiment_path.exists():
        return ExperimentProfile(experiment_id=experiment_dir.name, tags=())

    text = read_text(experiment_path)
    tags = _infer_tags_from_text(
        " ".join(
            part
            for part in [
                _extract_section_text(text, "Problem Type"),
                _extract_section_text(text, "Objective Metric"),
                _extract_section_text(text, "Split Strategy"),
                _extract_section_text(text, "Data Profile"),
                _extract_section_text(text, "Goal"),
                _extract_section_text(text, "Constraints"),
                _extract_section_text(text, "Known Risks"),
            ]
            if part
        )
    )

    row_count = _extract_count(
        _extract_section_text(text, "Data Profile"),
        label="row count",
    )
    feature_count = _extract_count(
        _extract_section_text(text, "Data Profile"),
        label="feature count",
    )
    if row_count is not None:
        tags.add(_row_count_bucket(row_count))
    if feature_count is not None:
        tags.add(_feature_count_bucket(feature_count))

    return ExperimentProfile(
        experiment_id=experiment_dir.name, tags=tuple(sorted(tags))
    )


def learnings_profile_tags(experiment_dir: Path) -> list[str]:
    return list(build_experiment_profile(experiment_dir).tags)


def build_cross_experiment_learnings_context(experiment_dir: Path) -> str | None:
    learnings_path = learnings_file()
    if not learnings_path.exists():
        return None

    learnings_text = read_text(learnings_path).strip()
    if not learnings_text:
        return None

    retrieved = retrieve_relevant_learnings(
        experiment_dir, learnings_text=learnings_text
    )
    if not retrieved:
        return None

    profile = build_experiment_profile(experiment_dir)
    warm_start_lines = _warm_start_lines(profile, retrieved)
    lines = [
        "### Warm-Start Note",
        "",
        "Advisory priors from relevant past experiments. Verify before following them.",
        "",
    ]
    lines.extend([f"- {line}" for line in warm_start_lines])
    lines.extend(["", "### Relevant Excerpts", ""])
    for excerpt in retrieved:
        lines.append(f"#### {excerpt.title}")
        lines.append("")
        if excerpt.matched_tags:
            lines.append(
                "- Matched profile tags: "
                + ", ".join(f"`{tag}`" for tag in excerpt.matched_tags)
            )
        for bullet in excerpt.bullets[:MAX_EXCERPT_BULLETS]:
            lines.append(f"- {bullet}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def retrieve_relevant_learnings(
    experiment_dir: Path,
    *,
    learnings_text: str | None = None,
    limit: int = MAX_RETRIEVED_EXCERPTS,
) -> list[LearningsExcerpt]:
    learnings_text = learnings_text or _read_learnings_text()
    if not learnings_text:
        return []

    profile = build_experiment_profile(experiment_dir)
    if not profile.tags:
        return []

    excerpts: list[LearningsExcerpt] = []
    for title, section_body in _parse_sections(learnings_text):
        explicit_tags, body_without_tags = _extract_explicit_tags(section_body)
        inferred_tags = tuple(
            sorted(_infer_tags_from_text(f"{title}\n{body_without_tags}"))
        )
        matched_explicit = sorted(set(profile.tags) & set(explicit_tags))
        matched_inferred = sorted(
            (set(profile.tags) & set(inferred_tags)) - set(explicit_tags)
        )
        score = (3.0 * len(matched_explicit)) + (1.5 * len(matched_inferred))
        if score < MIN_RELEVANCE_SCORE:
            continue
        bullets = tuple(_section_bullets(body_without_tags))
        excerpts.append(
            LearningsExcerpt(
                title=title,
                body=body_without_tags.strip(),
                bullets=bullets or (_section_summary_line(body_without_tags),),
                explicit_tags=explicit_tags,
                inferred_tags=inferred_tags,
                matched_tags=tuple(matched_explicit + matched_inferred),
                score=score,
            )
        )

    excerpts.sort(
        key=lambda item: (item.score, len(item.matched_tags), item.title), reverse=True
    )
    return excerpts[:limit]


def _read_learnings_text() -> str | None:
    path = learnings_file()
    if not path.exists():
        return None
    content = read_text(path).strip()
    return content if content else None


def _parse_sections(text: str) -> list[tuple[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        title = match.group("title").strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


def _extract_explicit_tags(section_body: str) -> tuple[tuple[str, ...], str]:
    lines = section_body.splitlines()
    explicit_tags: tuple[str, ...] = ()
    cleaned_lines: list[str] = []
    consumed_tags = False
    for line in lines:
        stripped = line.strip()
        if not consumed_tags and stripped.lower().startswith("tags:"):
            tags_text = stripped.split(":", 1)[1]
            explicit_tags = tuple(
                sorted(tag.strip() for tag in tags_text.split(",") if tag.strip())
            )
            consumed_tags = True
            continue
        cleaned_lines.append(line)
    return explicit_tags, "\n".join(cleaned_lines).strip()


def _section_bullets(section_body: str) -> list[str]:
    bullets = []
    for line in section_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
    return bullets


def _section_summary_line(section_body: str) -> str:
    flattened = " ".join(
        line.strip() for line in section_body.splitlines() if line.strip()
    )
    return re.sub(r"\s+", " ", flattened).strip()


def _extract_section_text(markdown: str, heading: str) -> str | None:
    matches = list(SECTION_RE.finditer(markdown))
    for index, match in enumerate(matches):
        if match.group("title").strip() != heading:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        content = markdown[start:end].strip()
        if not content:
            return None
        normalized_lines = [
            re.sub(r"^\s*[-*]\s+", "", line).strip() for line in content.splitlines()
        ]
        cleaned = " ".join(line for line in normalized_lines if line)
        return re.sub(r"\s+", " ", cleaned).strip()
    return None


def _extract_count(section_text: str | None, *, label: str) -> int | None:
    if not section_text:
        return None
    match = re.search(
        rf"{re.escape(label)}[^0-9]*(\d[\d,]*)", section_text, re.IGNORECASE
    )
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def _row_count_bucket(row_count: int) -> str:
    if row_count < 10_000:
        return "small-data"
    if row_count < 100_000:
        return "medium-data"
    return "large-data"


def _feature_count_bucket(feature_count: int) -> str:
    if feature_count <= 20:
        return "small-feature-set"
    if feature_count <= 100:
        return "medium-feature-set"
    return "wide-feature-set"


def _infer_tags_from_text(text: str) -> set[str]:
    lower = text.lower()
    tags: set[str] = set()

    def add_if(tag: str, *needles: str) -> None:
        if any(needle in lower for needle in needles):
            tags.add(tag)

    add_if("classification", "classification", "auc", "precision", "recall", "log_loss")
    add_if(
        "regression", "regression", "rmse", "mae", "r^2", " r2", "r2 ", "_r2", "val_r2"
    )
    add_if("ranking", "captured_at_", "top 20%", "top-k", "rank", "expected expansion")
    add_if("temporal-split", "temporal split", "time split", "quarter", "month", "q4")
    add_if("random-split", "random split", "stratified split")
    add_if("holdout", "holdout", "test set")
    add_if("no-holdout", "no holdout")
    add_if("imbalance", "imbalanc", "positive rate", "class balance", "rare event")
    add_if("zero-inflated", "zero-inflated", "zero inflated")
    add_if("auc", "auc")
    add_if("r2", "r^2", " r2", "r2 ", "_r2", "val_r2")
    add_if("top-k", "captured_at_", "top 20%", "top-k")
    add_if("calibration", "calibration", "brier")
    add_if("logistic-regression", "logistic regression", "logreg", "elasticnet lr")
    add_if(
        "linear-model",
        "linear model",
        "logistic regression",
        "ridge",
        "lasso",
        "elasticnet",
    )
    add_if("tree-model", "lightgbm", "xgboost", "gbdt", "tree model", "trees ")
    add_if("lightgbm", "lightgbm", "lgbm")
    add_if("xgboost", "xgboost", "xgb")
    add_if(
        "feature-engineering",
        "feature engineering",
        "engineered feature",
        "ratio feature",
        "interaction",
    )
    add_if(
        "aggregated-features",
        "pre-aggregated",
        "90-day window",
        "growth rates",
        "rolled-up",
        "summary features",
    )
    add_if(
        "value-modeling",
        "predict how much",
        "magnitude",
        "amount",
        "value-aware",
        "captured_at_",
    )
    return tags


def _warm_start_lines(
    profile: ExperimentProfile, excerpts: list[LearningsExcerpt]
) -> list[str]:
    priors = [
        _compress_bullet(excerpt.bullets[0])
        for excerpt in excerpts
        if excerpt.bullets and excerpt.bullets[0]
    ][:2]

    suggestions: list[str] = []
    if "classification" in profile.tags and any(
        "logistic regression" in excerpt.body.lower()
        or "logreg" in excerpt.body.lower()
        for excerpt in excerpts
    ):
        suggestions.append(
            "Suggested first move: establish a simple logistic-regression baseline before spending cycles on larger trees."
        )
    if "temporal-split" in profile.tags:
        suggestions.append(
            "Suggested first move: inspect temporal shift early and treat changing relationships across time as structural evidence, not just noise."
        )
    if any(
        "bootstrap" in excerpt.body.lower() or "ci" in excerpt.body.lower()
        for excerpt in excerpts
    ):
        suggestions.append(
            "Suggested first move: run uncertainty checks before calling small leaderboard gaps real."
        )

    lines = [f"Prior: {prior}" for prior in priors]
    for suggestion in suggestions:
        if suggestion not in lines:
            lines.append(suggestion)
    if not lines:
        lines.append(
            "Use the retrieved excerpts below as advisory starting priors, then verify them against the current data."
        )
    return lines[:3]


def _compress_bullet(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized.rstrip(".") + "."


# --- Learnings extraction (moved from loop/core.py) ---

LEARNINGS_EXTRACTION_PROMPT = """\
You are reviewing a completed ML experiment's research journal to extract \
generalizable learnings.

Experiment: `{experiment_id}`

## Research Journal

{journal}

## Task

Extract 3-7 bullet points of **generalizable ML learnings** from this \
experiment — insights that would help a researcher working on a *different* \
dataset or problem.

Focus on:
- Techniques that worked well (or didn't) and why
- Data preprocessing insights that generalize
- Model selection heuristics
- Evaluation methodology insights
- Common pitfalls discovered

Do NOT include:
- Dataset-specific findings (specific feature names, thresholds, etc.)
- Results that only apply to this particular problem
- Obvious textbook knowledge

If there are no genuinely generalizable learnings, respond with exactly: \
NO_LEARNINGS

Output ONLY the bullet points (or NO_LEARNINGS), no headers or preamble.
"""

LEARNINGS_EXTRACTION_TIMEOUT = 120  # seconds


def extract_and_append_learnings(experiment_dir: Path) -> bool:
    """Extract generalizable learnings from a completed experiment.

    Calls the Claude CLI to analyze the research journal, then appends
    any extracted insights to ``learnings.md`` at the repo root with
    experiment ID and date attribution.
    """
    jpath = journal_path(experiment_dir)
    if not jpath.exists():
        return False

    journal_text = read_text(jpath)
    # Use str.replace instead of .format() so that journal text containing
    # Python dicts, JSON, or set literals with { / } never raises KeyError.
    prompt = LEARNINGS_EXTRACTION_PROMPT.replace(
        "{experiment_id}", experiment_dir.name
    ).replace("{journal}", journal_text)

    try:
        result = subprocess.run(
            ["claude", "--print"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=LEARNINGS_EXTRACTION_TIMEOUT,
            cwd=ROOT,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

    if result.returncode != 0:
        return False

    extracted = result.stdout.strip()
    if not extracted or "NO_LEARNINGS" in extracted:
        return False

    lf = learnings_file()
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tags = learnings_profile_tags(experiment_dir)
    tags_line = f"Tags: {', '.join(tags)}\n\n" if tags else ""
    entry = f"\n## From `{experiment_dir.name}` ({date})\n\n{tags_line}{extracted}\n"

    if not lf.exists():
        header = (
            "# Cross-Experiment Learnings\n\n"
            "Generalizable ML insights extracted from completed experiments.\n"
        )
        write_text(lf, header + entry)
    else:
        replace_or_append_learnings(lf, experiment_dir.name, entry)

    return True
