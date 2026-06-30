"""Generate animation script JSON from Agentic ML Loop experiment artifacts."""

from __future__ import annotations

import contextlib
import json
import re
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from lib.* import ...` works
# when this script is invoked as `python3 viz/generate.py`.


def _experiment_sections(text: str) -> dict[str, str]:
    wanted = ("Title", "Goal", "Objective Metric")
    found: dict[str, str] = {}
    for section in re.split(r"^## ", text, flags=re.MULTILINE):
        heading = section.strip().splitlines()[0] if section.strip() else ""
        for key in wanted:
            if heading.startswith(key):
                found[key] = section
                break
    return found


def _first_nonblank_body_line(section: str) -> str:
    for line in section.strip().splitlines()[1:]:
        if line.strip():
            return line.strip()
    return ""


def _body_paragraph(section: str) -> str:
    lines: list[str] = []
    for line in section.strip().splitlines()[1:]:
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        elif lines:
            break
    return " ".join(lines)


def _short_goal(goal: str) -> str:
    if len(goal) <= 80:
        return goal
    for delimiter, minimum in ((".", 30), (" — ", 20), (",", 30)):
        cut = goal[:80].rfind(delimiter)
        if cut > minimum:
            suffix = "." if delimiter != "." else delimiter
            return goal[:cut] + suffix
    return goal[:77] + "..."


def _visualization_title(title: str) -> str:
    if not title:
        return ""
    parts = title.split(":")
    objective = ":".join(parts[1:]).strip() if len(parts) > 1 else title
    obj_title = objective.title().replace("Auc", "AUC")
    return f"Autonomous Research Agent: {obj_title}"


def _objective_metric(section: str) -> str:
    match = re.search(r"`([^`]+)`", section)
    return match.group(1).split("—")[0].strip() if match else ""


def parse_experiment_md(path: Path) -> dict:
    text = path.read_text()
    sections = _experiment_sections(text)
    return {
        "name": _visualization_title(_first_nonblank_body_line(sections.get("Title", ""))),
        "goal": _short_goal(_body_paragraph(sections.get("Goal", ""))),
        "metric": _objective_metric(sections.get("Objective Metric", "")),
    }


def _round_metric(value: object, digits: int = 3) -> float | None:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return None


def _metric_family(objective_metric: str, validation_metrics: dict) -> str:
    metric = objective_metric.lower()
    validation_names = {str(name).lower() for name in validation_metrics}
    if metric.startswith("val_captured_at_") or any(
        name.startswith("captured_at_") for name in validation_names
    ):
        return "captured_at_k"
    if any(
        token in metric or token in validation_names
        for token in ("auc", "accuracy", "precision", "recall", "avg_precision")
    ) or validation_names & {"log_loss", "brier_score", "f1"}:
        return "classification"
    if any(token in metric for token in ("r2", "rmse", "mae", "mape")) or (
        validation_names & {"r2", "rmse", "mae", "mape"}
    ):
        return "regression"
    return "generic"


def _friendly_primary_metric(objective_metric: str, family: str) -> str:
    if family == "captured_at_k":
        return _friendly_metric(objective_metric)
    if family == "classification":
        return objective_metric.replace("val_", "validation ").replace("_", " ")
    if family == "regression":
        return objective_metric.replace("val_", "validation ").replace("_", " ")
    return objective_metric or "objective score"


def _normalize_candidate(entry: dict) -> dict:
    objective_metric = str(entry.get("objective_metric", ""))
    rounded_objective = _round_metric(entry.get("objective_score"))
    objective_score = rounded_objective if rounded_objective is not None else 0.0
    validation = entry.get("metrics", {}).get("validation", {})
    if not isinstance(validation, dict):
        validation = {}
    family = _metric_family(objective_metric, validation)
    metrics = {
        str(name): value
        for name, raw_value in validation.items()
        if (value := _round_metric(raw_value)) is not None
    }

    candidate = {
        "id": entry["candidate_id"],
        "family": entry.get("model_family", ""),
        "objective_metric": objective_metric,
        "objective_score": objective_score,
        "score": objective_score,
        "primary_metric": {
            "name": objective_metric,
            "label": _friendly_primary_metric(objective_metric, family),
            "value": objective_score,
            "family": family,
        },
        "metrics": metrics,
        "notes": entry.get("notes", ""),
    }

    if family == "captured_at_k":
        captured_fields = {
            "at_10": "captured_at_10pct",
            "at_20": "captured_at_20pct",
            "at_30": "captured_at_30pct",
            "at_50": "captured_at_50pct",
        }
        for out_key, metric_key in captured_fields.items():
            value = _round_metric(validation.get(metric_key))
            if value is not None:
                candidate[out_key] = value
        auc = _round_metric(validation.get("auc"))
        if auc is not None:
            candidate["auc"] = auc

    return candidate


def parse_results_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    candidates = [_normalize_candidate(entry) for entry in data]
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def parse_journal_cycles(path: Path) -> list[dict]:
    text = path.read_text()
    cycle_pattern = re.compile(r"^## Cycle (\d+):\s*(.+)$", re.MULTILINE)
    matches = list(cycle_pattern.finditer(text))

    cycles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]

        num = int(m.group(1))
        title = m.group(2).strip()

        objective = extract_objective(body)
        has_research = "### Research Phase" in body or "research" in title.lower()
        findings = extract_key_findings(body)
        table_candidates = extract_result_table(body)

        cycles.append(
            {
                "number": num,
                "title": title,
                "objective": objective,
                "has_research": has_research,
                "findings": findings,
                "candidates": table_candidates,
            }
        )

    return cycles


def extract_objective(body: str) -> str:
    m = re.search(r"\*\*Objective\*\*:\s*(.+?)(?:\n\n|\n---)", body, re.DOTALL)
    if m:
        text = m.group(1).strip().replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        return text
    return ""


def extract_key_findings(body: str) -> list[str]:
    findings = []
    m = re.search(r"\*\*Key findings?\*\*:?\s*\n((?:\s*\d+\..+\n?)+)", body)
    if m:
        for line in m.group(1).strip().splitlines():
            line = re.sub(r"^\s*\d+\.\s*", "", line).strip()
            line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            if line and len(line) > 10:
                short = line.split(".")[0] + "." if len(line) > 80 else line
                findings.append(short)
    if not findings:
        m = re.search(r"\*\*(?:Key finding|Definitive finding)\*\*:\s*(.+)", body)
        if m:
            findings.append(m.group(1).strip()[:100])
    return findings[:4]


def _clean_name(name: str) -> str:
    """Strip markdown artifacts from candidate names."""
    name = re.sub(r"\*+", "", name).strip()
    return re.sub(r"\s*\(.*?\)\s*$", "", name)


def extract_result_table(body: str) -> list[dict]:
    """Extract candidates from the RESULTS section only, not pre-run tables."""
    results_section = ""
    m = re.search(r"### (?:Results|Baseline Results).*?\n(.*?)(?=\n### |\Z)", body, re.DOTALL)
    results_section = m.group(1) if m else body

    candidates = []
    seen = set()
    tables = re.findall(r"(\|.+\|.+\|.+)\n\|[-\s|:]+\n((?:\|.+\n)+)", results_section)
    if tables:
        header_row, body = tables[0]
        header_cols = [c.strip().strip("*").lower() for c in header_row.split("|")[1:-1]]
        col_idx: dict[str, int] = {name: i for i, name in enumerate(header_cols)}
        at_20_idx = next(
            (col_idx[k] for k in ("at_20", "captured_at_20pct", "score") if k in col_idx),
            None,
        )
        for row in body.strip().splitlines():
            cols = [c.strip().strip("*").strip() for c in row.split("|")[1:-1]]
            if not cols:
                continue
            name = _clean_name(cols[0])
            if not name or name.startswith("---") or name == "Candidate":
                continue
            if name in seen:
                continue
            seen.add(name)
            at_20 = None
            if at_20_idx is not None and at_20_idx < len(cols):
                with contextlib.suppress(ValueError):
                    at_20 = float(cols[at_20_idx].replace("—", "").strip())
            candidates.append({"name": name, "at_20": at_20})
    return candidates


def _friendly_metric(metric: str) -> str:
    """Convert technical metric names to plain English."""
    friendly = {
        "val_captured_at_20pct": "how much revenue we capture in top picks",
        "val_captured_at_10pct": "revenue captured in top 10% of picks",
        "val_captured_at_30pct": "revenue captured in top 30% of picks",
        "val_accuracy": "prediction accuracy",
        "val_auc": "ranking quality",
        "val_f1": "prediction quality",
    }
    return friendly.get(metric, metric)


def _validation_total_for_experiment(experiment_dir: Path, metric: str) -> float | None:
    """Return the validation value total for captured-at-K experiments."""
    if not metric.startswith("val_captured_at_"):
        return None

    experiment_id = experiment_dir.name
    try:
        data_mod = __import__(
            f"lib.{experiment_id}.data",
            fromlist=["TOTAL_COLUMN", "load_dataset", "split_dataset"],
        )
    except ImportError:
        return None

    total_column = getattr(data_mod, "TOTAL_COLUMN", None)
    if total_column is None:
        return None

    try:
        splits = data_mod.split_dataset(data_mod.load_dataset())
    except (FileNotFoundError, ValueError, KeyError):
        return None

    if total_column not in splits.validation.columns:
        return None

    return float(splits.validation[total_column].sum())


_RESEARCH_TEXTS = {
    1: "Studying the data and exploring what simple approaches might work as a starting point",
    2: "Researching useful features for this problem and studying value-weighted classification",
    3: "Analyzing patterns in what worked so far and exploring new feature ideas",
    4: "Final review — checking if any unexplored approaches could beat the current leader",
}

_JOURNAL_TEXTS = {
    1: "Recording baseline results and initial observations in the research journal",
    2: "Writing down what worked, what didn't, and ideas for the next agent to try",
    3: "Documenting key learnings and updating the research journal with findings",
    4: "Final notes — summarizing all discoveries for the experiment record",
}


def _intro_scene(experiment: dict) -> dict:
    metric_label = _friendly_metric(experiment["metric"])
    return {
        "type": "intro",
        "station": "desk",
        "title": "Understanding the Goal",
        "text": experiment["goal"],
        "subtitle": f"Measuring: {metric_label}",
    }


def _hypothesis_scene(cycle: dict) -> dict:
    objective = cycle["objective"].replace("learned models", "trained models")
    return {
        "type": "hypothesis",
        "cycle": cycle["number"],
        "station": "desk",
        "title": f"Round {cycle['number']}: Planning",
        "text": objective,
    }


def _research_scene(cycle_number: int) -> dict:
    return {
        "type": "research",
        "cycle": cycle_number,
        "station": "library",
        "title": "Exploring Ideas",
        "text": _RESEARCH_TEXTS.get(cycle_number, "Reviewing results and exploring new approaches"),
    }


def _dedupe_preserving_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _new_candidate_names(table_candidates: list[dict], leaderboard: list[dict]) -> list[str]:
    existing_names = {entry["name"] for entry in leaderboard}
    names = [
        candidate["name"]
        for candidate in table_candidates
        if candidate["name"] not in existing_names
    ]
    return _dedupe_preserving_order(names or [candidate["name"] for candidate in table_candidates])


def _training_scene(cycle_number: int, candidate_names: list[str]) -> dict:
    approach_word = "es" if len(candidate_names) != 1 else ""
    return {
        "type": "training",
        "cycle": cycle_number,
        "station": "lab",
        "title": "Testing Models",
        "text": f"Running {len(candidate_names)} different approach{approach_word}",
        "candidates": candidate_names[:6],
    }


def _candidate_info(candidate_lookup: dict, candidate_name: str) -> dict:
    return candidate_lookup.get(
        candidate_name, candidate_lookup.get(candidate_name.replace(" ", "-"), {})
    )


def _leaderboard_entries(
    table_candidates: list[dict],
    candidate_lookup: dict,
    leaderboard: list[dict],
) -> tuple[list[dict], dict | None, dict | None]:
    best_this_cycle = None
    new_entries: list[dict] = []
    for candidate in table_candidates:
        candidate_name = candidate["name"]
        info = _candidate_info(candidate_lookup, candidate_name)
        score = candidate.get("at_20")
        if score is None and info:
            score = info.get("score")
        if score is None:
            continue
        entry = {"name": candidate_name, "score": round(score, 3)}
        if info and info.get("primary_metric"):
            entry["primary_metric"] = info["primary_metric"]
        new_entries.append(entry)
        if not [existing for existing in leaderboard if existing["name"] == candidate_name]:
            leaderboard.append(entry)
        if best_this_cycle is None or score > best_this_cycle["score"]:
            best_this_cycle = entry

    leaderboard.sort(key=lambda entry: entry["score"], reverse=True)
    overall_best = leaderboard[0] if leaderboard else None
    return new_entries, best_this_cycle, overall_best


def _cycle_result_text(
    *,
    best_this_cycle: dict | None,
    overall_best: dict | None,
    leaderboard: list[dict],
    previous_best_name: str | None,
) -> tuple[str, bool]:
    if best_this_cycle is None:
        return "", False
    is_new_best = (
        overall_best is not None
        and best_this_cycle["name"] == overall_best["name"]
        and best_this_cycle["name"] != previous_best_name
    )
    if is_new_best:
        previous_score = leaderboard[1]["score"] if len(leaderboard) > 1 else 0
        delta = best_this_cycle["score"] - previous_score
        return (
            f"New leader! {best_this_cycle['name']} scores {best_this_cycle['score']:.3f} "
            f"(+{delta:.3f} improvement)",
            True,
        )
    if overall_best and best_this_cycle["name"] != overall_best["name"]:
        return (
            f"No improvement — {overall_best['name']} still leads with {overall_best['score']:.3f}",
            False,
        )
    return f"Still leading: {overall_best['name']} with {overall_best['score']:.3f}", False


def _evaluation_scene(
    cycle_number: int,
    result_text: str,
    new_entries: list[dict],
    leaderboard: list[dict],
    is_new_best: bool,
) -> dict:
    return {
        "type": "evaluation",
        "cycle": cycle_number,
        "station": "whiteboard",
        "title": "Scoreboard Update",
        "text": result_text,
        "results": new_entries,
        "leaderboard": [dict(entry) for entry in leaderboard],
        "is_new_best": is_new_best,
    }


def _journal_scene(cycle: dict) -> dict:
    cycle_number = cycle["number"]
    findings_summary = cycle["findings"][:2] if cycle["findings"] else []
    return {
        "type": "journal",
        "cycle": cycle_number,
        "station": "desk",
        "title": "Recording Findings",
        "text": _JOURNAL_TEXTS.get(
            cycle_number, "Writing findings and learnings in the research journal"
        ),
        "findings": findings_summary,
    }


def _cycle_candidate_scenes(
    cycle: dict,
    candidate_lookup: dict,
    leaderboard: list[dict],
    previous_best_name: str | None,
) -> tuple[list[dict], str | None]:
    table_candidates = cycle["candidates"]
    if not table_candidates:
        return [], previous_best_name

    candidate_names = _new_candidate_names(table_candidates, leaderboard)
    scenes = [_training_scene(cycle["number"], candidate_names)]
    new_entries, best_this_cycle, overall_best = _leaderboard_entries(
        table_candidates, candidate_lookup, leaderboard
    )
    result_text, is_new_best = _cycle_result_text(
        best_this_cycle=best_this_cycle,
        overall_best=overall_best,
        leaderboard=leaderboard,
        previous_best_name=previous_best_name,
    )
    if overall_best:
        previous_best_name = overall_best["name"]
    scenes.append(
        _evaluation_scene(cycle["number"], result_text, new_entries, leaderboard, is_new_best)
    )
    scenes.append(_journal_scene(cycle))
    return scenes, previous_best_name


def _value_capture_summary(
    experiment_dir: Path,
    experiment: dict,
    all_candidates: list[dict],
    leaderboard: list[dict],
    best: dict,
) -> dict | None:
    total_validation_value = _validation_total_for_experiment(experiment_dir, experiment["metric"])
    baseline_entry = next((c for c in all_candidates if c["id"] == "rule-baseline"), None)
    best_entry = next((c for c in all_candidates if c["id"] == best["name"]), None)
    if not baseline_entry and leaderboard:
        baseline = leaderboard[-1]
        baseline_entry = next((c for c in all_candidates if c["id"] == baseline["name"]), None)
    if not baseline_entry or not best_entry or total_validation_value is None:
        return None

    improvement = best_entry["score"] - baseline_entry["score"]
    pct_improvement = (
        (improvement / baseline_entry["score"] * 100) if baseline_entry["score"] else 0
    )
    return {
        "baseline": {
            "name": baseline_entry["id"],
            "auc": baseline_entry.get("auc", 0),
            "at_20": baseline_entry.get("at_20", baseline_entry["score"]),
            "value_captured": round(
                baseline_entry.get("at_20", baseline_entry["score"]) * total_validation_value
            ),
        },
        "winner": {
            "name": best_entry["id"],
            "auc": best_entry.get("auc", 0),
            "at_20": best_entry.get("at_20", best_entry["score"]),
            "value_captured": round(
                best_entry.get("at_20", best_entry["score"]) * total_validation_value
            ),
        },
        "total_validation_value": total_validation_value,
        "extra_revenue": round(improvement * total_validation_value),
        "improvement_pct": round(pct_improvement, 1),
    }


def replay_scenes(
    experiment_dir: Path,
    experiment: dict,
    cycles: list[dict],
    all_candidates: list[dict],
) -> list[dict]:
    scenes: list[dict] = []
    candidate_lookup = {c["id"]: c for c in all_candidates}
    leaderboard: list[dict] = []

    scenes.append(_intro_scene(experiment))

    prev_best_name = None

    for cycle in cycles:
        scenes.append(_hypothesis_scene(cycle))
        scenes.append(_research_scene(cycle["number"]))
        candidate_scenes, prev_best_name = _cycle_candidate_scenes(
            cycle, candidate_lookup, leaderboard, prev_best_name
        )
        scenes.extend(candidate_scenes)

    best = leaderboard[0] if leaderboard else {"name": "?", "score": 0}
    summary = _value_capture_summary(experiment_dir, experiment, all_candidates, leaderboard, best)

    scenes.append(
        {
            "type": "finale",
            "station": "desk",
            "title": "Experiment Complete!",
            "text": f"Winner: {best['name']} with a score of {best['score']:.3f}",
            "leaderboard": [dict(e) for e in leaderboard],
            "summary": summary,
        }
    )

    return scenes


def generate(experiment_dir: str) -> None:
    exp_path = Path(experiment_dir)
    if not exp_path.is_dir():
        print(f"Error: {exp_path} is not a directory")
        sys.exit(1)

    experiment = parse_experiment_md(exp_path / "experiment.md")
    candidates = parse_results_json(exp_path / "results.json")
    cycles = parse_journal_cycles(exp_path / "research_journal.md")

    scenes = replay_scenes(exp_path, experiment, cycles, candidates)

    script = {
        "experiment": {
            "name": experiment["name"],
            "metric": experiment["metric"],
            "objective": experiment["goal"],
        },
        "scenes": scenes,
        "total_cycles": len(cycles),
        "total_candidates": len(candidates),
    }

    out_dir = Path(__file__).parent / "output" / exp_path.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "script.json"
    out_file.write_text(json.dumps(script, indent=2))
    print(f"Generated {out_file} ({len(scenes)} scenes)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python viz/generate.py experiments/<experiment_id>")
        sys.exit(1)
    generate(sys.argv[1])
