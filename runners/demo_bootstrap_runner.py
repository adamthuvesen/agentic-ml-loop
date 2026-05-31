from __future__ import annotations

from pathlib import Path


from lib.demo_bootstrap.data import load_dataset, split_dataset
from lib.demo_bootstrap.modeling import CANDIDATE_RUNNERS
from lib.runner import init_experiment_dir, run_runner_main

EXPERIMENT_ID = "demo_bootstrap"
_TEMPLATES = Path(__file__).resolve().parents[1] / "experiments" / "templates"
TEMPLATE_PATH = _TEMPLATES / "demo_bootstrap.md"
RESEARCH_SOURCES_TEMPLATE_PATH = _TEMPLATES / "demo_bootstrap_research_sources.md"


def _load_splits():
    return split_dataset(load_dataset())


def init_demo(force: bool = False) -> Path:
    return init_experiment_dir(
        EXPERIMENT_ID, TEMPLATE_PATH, RESEARCH_SOURCES_TEMPLATE_PATH, force=force
    )


if __name__ == "__main__":
    raise SystemExit(
        run_runner_main(
            EXPERIMENT_ID,
            CANDIDATE_RUNNERS,
            _load_splits,
            TEMPLATE_PATH,
            RESEARCH_SOURCES_TEMPLATE_PATH,
        )
    )
