from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from experiment import (
    count_journal_cycles,
    get_cross_learnings_enabled,
    get_min_cycles_before_complete,
    journal_path,
    results_file,
)
from lib.io import load_json
from lib.learnings import cross_experiment_learnings_context
from lib.signals import (
    advisory_signals as collect_advisory_signals,
)
from lib.signals import (
    results_snapshot,
)

from .constants import ROOT, STATE_PATH_NAME

PROGRAM_MD_PATH = ROOT / "program.md"

CYCLE_DONE_MARKER = "<promise>CYCLE_DONE</promise>"
EXPERIMENT_COMPLETE_MARKER = "<promise>EXPERIMENT_COMPLETE</promise>"

RESEARCHER_FRAMING = (
    "You are an ML researcher running an autonomous experiment. Make meaningful\n"
    "progress each cycle — one clear objective, done well.\n"
    "\n"
    "- Research before code — search for prior work before writing any model\n"
    "- Form a working theory and actively try to break it\n"
    "- One clear objective per cycle, 1-3 candidates max\n"
    '- Be honest about uncertainty — "I don\'t know yet" beats a confident guess\n'
)

OUTPUT_PATHS_FALLBACK = (
    "## Output paths\n"
    "\n"
    "Every experiment uses a fixed three-folder layout under `experiments/<exp_id>/`:\n"
    "\n"
    "- `outputs/` — deliverables a stakeholder reads (final reports, shortlists, ranked CSVs).\n"
    "- `work/` — intermediate artefacts one cycle writes for another to read.\n"
    "- `scripts/` — one-shot Python scripts you write during cycles. Long-lived modules stay in `lib/<exp_id>/`.\n"
    "\n"
    "Use the helpers in `lib.paths`: `outputs_dir(exp_dir)`, `work_dir(exp_dir)`, `scripts_dir(exp_dir)` — each creates the directory on first call.\n"
    "Do not write CSVs, summaries, or scripts directly to the experiment root; the validator will surface that as a warning.\n"
)

CYCLE_ANALYSIS_PROTOCOL = (
    "## Before You Choose What To Do\n"
    "\n"
    "Write this analysis into the journal before coding.\n"
    "\n"
    "1. **Read artifacts** — `experiment.md`, `research_journal.md`, `research_sources.md`, `results.json`. Read fully, don't skim.\n"
    "2. **Analyze results** — Overfit gaps, diminishing returns, Calibration (Brier), Significance (bootstrap CI), untried approaches.\n"
    "3. **Consult external knowledge** — Check `research_sources.md`; if thin, do fresh web research.\n"
    "4. **Form hypothesis** — Pick the single most promising next move. Research/error-analysis cycles count.\n"
)

FIRST_CYCLE_RESEARCH = (
    "## First Cycle: Research Phase Required\n"
    "\n"
    "Before writing code: do web research (2-3 sources — Kaggle, papers, blogs),\n"
    "look up library APIs with context7, and record findings in `research_sources.md`.\n"
    "Consider the business context — what errors are more costly? Let domain reasoning\n"
    "shape metric choice and feature priorities.\n"
)

STALL_RESEARCH_NUDGE = (
    "## Stall Warning — Change Your Approach\n"
    "\n"
    "You've had {no_progress} consecutive cycles without measurable progress.\n"
    "Doing more of the same will not break this stall. STOP and change angle:\n"
    "\n"
    "- Search for how others solved similar problems — Kaggle solutions, papers.\n"
    "  What did the top solutions do differently from what you've tried?\n"
    "- Look for techniques or model families you haven't tested yet.\n"
    "- Record findings in `research_sources.md` before writing code.\n"
    "\n"
    "If after researching you believe the space is genuinely exhausted, declare\n"
    "the experiment complete.\n"
)

COMPLETION_RIGOR_CHECK = (
    "## Before Declaring EXPERIMENT_COMPLETE\n"
    "\n"
    "ALL must be true:\n"
    "- External research done — `research_sources.md` has concrete source cards\n"
    "- Tried ≥2 meaningfully different model families (not just hyperparam variants)\n"
    "- Error analysis done on best model — what does it get wrong?\n"
    "- Score differences checked for significance (bootstrap CI or similar)\n"
    "- Documented why untried approaches wouldn't help\n"
)

GUIDELINES_PATH = ROOT / "guidelines.md"

DEFAULT_MAX_PROMPT_TOKENS = 80_000


@dataclass
class CyclePrompt:
    """Structured cycle prompt with static/dynamic section separation."""

    static_sections: list[str]
    dynamic_sections: list[str]
    _assembled_cache: str | None = field(default=None, repr=False, compare=False)

    def assemble(self) -> str:
        """Join static and dynamic sections with a boundary marker."""
        if self._assembled_cache is not None:
            return self._assembled_cache
        static_text = "\n\n".join(s.strip() for s in self.static_sections if s.strip())
        dynamic_text = "\n\n".join(s.strip() for s in self.dynamic_sections if s.strip())
        parts = [static_text, "# --- dynamic context below ---"]
        if dynamic_text:
            parts.append(dynamic_text)
        self._assembled_cache = "\n\n".join(parts) + "\n"
        return self._assembled_cache

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate: len(assembled_text) // 4."""
        return len(self.assemble()) // 4

    def truncate_for_budget(
        self,
        experiment_dir: Path,
        cycle_id: str,
        max_tokens: int,
    ) -> CyclePrompt:
        """Return a copy within *max_tokens* by dropping lower-priority dynamic content."""
        steps: list[tuple[str, dict[str, object]]] = [
            ("truncating journal entries to 1", {"journal_n": 1}),
            ("truncating results to top 3", {"journal_n": 1, "top_n": 3}),
            (
                "dropping advisory signals",
                {"journal_n": 1, "top_n": 3, "include_advisory": False},
            ),
            (
                "dropping cross-experiment learnings",
                {
                    "journal_n": 1,
                    "top_n": 3,
                    "include_advisory": False,
                    "include_cross_learnings": False,
                },
            ),
            (
                "truncating guidelines excerpt",
                {
                    "journal_n": 1,
                    "top_n": 3,
                    "include_advisory": False,
                    "include_cross_learnings": False,
                    "guidelines_max_lines": 25,
                },
            ),
        ]
        current = self
        if current.estimated_tokens <= max_tokens:
            return current
        for message, kwargs in steps:
            print(f"WARNING: prompt over budget; {message}")
            current = _assemble_cycle_prompt(experiment_dir, cycle_id, **kwargs)
            if current.estimated_tokens <= max_tokens:
                return current
        print("WARNING: prompt still exceeds budget after all dynamic truncation steps")
        return current

    @property
    def static_hash(self) -> str:
        """SHA-256 hex digest of the joined static sections."""
        text = "\n\n".join(s.strip() for s in self.static_sections if s.strip())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_guidelines() -> str:
    """Return repo-root ``guidelines.md`` text, or empty string if missing."""
    if not GUIDELINES_PATH.exists():
        return ""
    return GUIDELINES_PATH.read_text(encoding="utf-8").strip()


def summarize_guidelines(*, max_lines: int | None = None) -> str:
    """Return guidelines text parsed from ``guidelines.md``.

    Returns empty string if the file is missing.
    """
    text = load_guidelines()
    if not text:
        return ""
    lines = text.splitlines()
    if max_lines is not None:
        lines = lines[:max_lines]
    body = "\n".join(lines).strip()
    return f"## Guidelines\n\n{body}\n\nFull rules: `{GUIDELINES_PATH.resolve()}`"


def load_researcher_identity() -> str:
    """Extract ``## Researcher Identity`` from ``program.md``.

    Returns the section text (capped at 15 lines).  Falls back to
    :data:`RESEARCHER_FRAMING` if the file is missing or the section isn't found.
    """
    if not PROGRAM_MD_PATH.exists():
        return RESEARCHER_FRAMING
    text = PROGRAM_MD_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"^## Researcher Identity\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        return RESEARCHER_FRAMING
    lines = match.group(1).strip().splitlines()
    return "\n".join(lines[:15]).strip()


def load_output_paths_section() -> str:
    """Extract ``## Output paths`` from ``program.md``.

    Returns the full section (heading + body, up to the next ``##``).
    Falls back to :data:`OUTPUT_PATHS_FALLBACK` if the file is missing or
    the section isn't found.
    """
    if not PROGRAM_MD_PATH.exists():
        return OUTPUT_PATHS_FALLBACK
    text = PROGRAM_MD_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"^(## Output paths\n.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        return OUTPUT_PATHS_FALLBACK
    return match.group(1).strip()


def read_last_journal_entries(experiment_dir: Path, n: int = 3) -> str:
    """Return the last n cycle entries from research_journal.md."""
    path = journal_path(experiment_dir)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"(?=^## Cycle \d{4}:)", text, flags=re.MULTILINE)
    cycle_parts = [p.strip() for p in parts if re.match(r"^## Cycle \d{4}:", p.strip())]
    return "\n\n".join(cycle_parts[-n:]).strip()


def top_results_lines(experiment_dir: Path, *, n: int = 5) -> list[str]:
    """Markdown bullet lines for top *n* candidates by ``objective_score``."""
    rf = results_file(experiment_dir)
    if not rf.exists():
        return ["- No results yet"]
    try:
        results = load_json(rf)
    except json.JSONDecodeError:
        return ["- No results yet"]
    if not results:
        return ["- No results yet"]
    ranked = sorted(
        [
            r
            for r in results
            if isinstance(r, dict) and isinstance(r.get("objective_score"), (int, float))
        ],
        key=lambda x: x.get("objective_score", float("-inf")),
        reverse=True,
    )
    if not ranked:
        return ["- No results yet"]
    return [
        f"- `{r.get('candidate_id', '?')}`: score={r.get('objective_score', '?')} {r.get('notes', '')}"
        for r in ranked[:n]
    ]


def latest_hypothesis(experiment_dir: Path) -> str:
    """Best-effort hypothesis line from the latest journal cycle entry."""
    last = read_last_journal_entries(experiment_dir, n=1)
    if not last:
        return ""
    for line in last.splitlines():
        if line.startswith("**Hypothesis:**"):
            return line.replace("**Hypothesis:**", "").strip()
    for line in last.splitlines():
        if line.startswith("## Cycle"):
            title = line.split(":", 1)
            return title[1].strip() if len(title) > 1 else ""
    return ""


def _assemble_cycle_prompt(
    experiment_dir: Path,
    cycle_id: str,
    *,
    journal_n: int = 3,
    top_n: int = 5,
    include_advisory: bool = True,
    include_cross_learnings: bool = True,
    guidelines_max_lines: int | None = None,
) -> CyclePrompt:
    """Build a CyclePrompt with configurable truncation parameters."""
    results = results_snapshot(experiment_dir)
    last_journal = read_last_journal_entries(experiment_dir, n=journal_n)
    journal_cycles = count_journal_cycles(experiment_dir)

    # ---- Static sections (identical across cycles for the same experiment) ----
    static_sections: list[str] = []

    # Identity
    static_sections.append("# ML Research Cycle\n\n" + load_researcher_identity())

    # Output paths convention (universal three-folder layout)
    static_sections.append(load_output_paths_section())

    # Guidelines summary
    guidelines_summary = summarize_guidelines(max_lines=guidelines_max_lines)
    if guidelines_summary:
        static_sections.append(guidelines_summary)

    # Workspace
    exp_scripts_dir = experiment_dir.resolve() / "scripts"
    lib_module_dir = (ROOT / "lib" / experiment_dir.name).resolve()
    static_sections.append(
        "\n".join(
            [
                "## Your Workspace",
                "",
                f"- Experiment: `{experiment_dir.resolve()}`",
                f"- One-shot cycle scripts: `{exp_scripts_dir}`",
                f"- Long-lived module (loader, context): `{lib_module_dir}`",
                "",
                "Read experiment.md, research_journal.md, and results.json before deciding what to do.",
                "Read research_sources.md before doing fresh research. Read program.md for research principles.",
            ]
        )
    )

    # Closing instructions / done markers
    static_sections.append(
        "\n".join(
            [
                "## When You're Done",
                "",
                "- Update your research journal with a new `## Cycle NNNN: <title>` entry for the current cycle.",
                "- If you used outside research, update `research_sources.md` with reusable findings. "
                "Keep `Reusable Takeaways` current — rewrite when new evidence changes the story.",
                "- If you tested models, add entries to `results.json` (each entry needs at least "
                "`candidate_id` and `objective_score`).",
                "- Save one-shot cycle scripts under `experiments/<exp>/scripts/` (use "
                "`scripts_dir(exp_dir)` from `lib.paths`). Long-lived modules live in "
                f"`{lib_module_dir}`.",
                "- Write deliverables under `outputs/` and intermediate scratch under `work/` "
                "(use `outputs_dir` / `work_dir` from `lib.paths`). Do not dump files in the experiment root.",
                f"- End with `{CYCLE_DONE_MARKER}` if continuing, or `{EXPERIMENT_COMPLETE_MARKER}` if genuinely done.",
            ]
        )
    )

    # ---- Dynamic sections (vary per cycle) ----
    dynamic_sections: list[str] = []

    # Cross-experiment learnings
    if include_cross_learnings and get_cross_learnings_enabled(experiment_dir):
        learnings_context = cross_experiment_learnings_context(experiment_dir)
        if learnings_context:
            dynamic_sections.append(
                "\n".join(
                    [
                        "## Cross-Experiment Learnings",
                        "",
                        "Relevant priors from `learnings.md`. Advisory — verify they fit this experiment.",
                        "",
                        learnings_context,
                    ]
                )
            )

    # Advisory signals
    if include_advisory:
        signal_items = collect_advisory_signals(experiment_dir, results=results)
        if signal_items:
            lines = [
                "## Advisory Signals",
                "",
                "Observations from current artifacts. Advisory only — use judgment.",
                "",
            ]
            lines.extend(f"- {signal}" for _, signal in signal_items)
            dynamic_sections.append("\n".join(lines))

    # Min-cycles contract
    min_before_complete = get_min_cycles_before_complete(experiment_dir)
    if min_before_complete is not None and journal_cycles < min_before_complete:
        dynamic_sections.append(
            "\n".join(
                [
                    "## Minimum cycles contract",
                    "",
                    f"This experiment requires **{min_before_complete}** completed journal cycles "
                    f"(`## Cycle NNNN:` headings) before you may use "
                    f"`{EXPERIMENT_COMPLETE_MARKER}`. Current journal cycle count: **{journal_cycles}**. "
                    f"Use `{CYCLE_DONE_MARKER}` until the contract is satisfied.",
                ]
            )
        )

    # Where you left off + results + recent journal
    briefing_lines = [
        "## Where You Left Off",
        "",
        f"- Cycle: `{cycle_id}`",
        f"- Experiment: `{experiment_dir.name}`",
        f"- Candidates tested: {len(results)}",
    ]
    if results:
        briefing_lines.extend(["", "### Results So Far", ""])
        briefing_lines.extend(top_results_lines(experiment_dir, n=top_n))
    if last_journal:
        briefing_lines.extend(["", "### Recent Research", "", last_journal])
    dynamic_sections.append("\n".join(briefing_lines))

    # Cycle analysis / first-cycle research
    if journal_cycles > 0:
        dynamic_sections.append(CYCLE_ANALYSIS_PROTOCOL)
    if journal_cycles == 0:
        dynamic_sections.append(FIRST_CYCLE_RESEARCH)

    # Stall nudge
    consecutive_no_progress = 0
    _state_path = experiment_dir / STATE_PATH_NAME
    if _state_path.exists():
        try:
            st = load_json(_state_path)
            consecutive_no_progress = st.get("consecutive_no_progress_cycles", 0)
        except json.JSONDecodeError:
            pass
    if consecutive_no_progress >= 2:
        dynamic_sections.append(STALL_RESEARCH_NUDGE.format(no_progress=consecutive_no_progress))

    # Completion rigor check
    if journal_cycles > 0:
        dynamic_sections.append(COMPLETION_RIGOR_CHECK)

    return CyclePrompt(static_sections=static_sections, dynamic_sections=dynamic_sections)


def cycle_prompt(
    experiment_dir: Path,
    cycle_id: str,
    *,
    max_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> CyclePrompt:
    """Assemble the full markdown prompt for one loop cycle.

    Returns a ``CyclePrompt`` with static/dynamic sections.  When the
    assembled prompt exceeds *max_tokens*, progressively truncates dynamic
    sections: journal (3->1), results (5->3), advisory signals, cross-learnings.
    """
    prompt = _assemble_cycle_prompt(experiment_dir, cycle_id)
    return prompt.truncate_for_budget(experiment_dir, cycle_id, max_tokens)
