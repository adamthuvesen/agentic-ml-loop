from __future__ import annotations

from pathlib import Path


from lib.demo_regression.data import (
    load_demo_regression_dataset,
    split_demo_regression_dataset,
)
from lib.demo_regression.modeling import CANDIDATE_RUNNERS
from lib.runner import init_experiment_dir, run_runner_main

EXPERIMENT_ID = "demo_regression"
TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "templates"
    / "demo_regression.md"
)


def _load_splits():
    """Load and split the bundled synthetic demo regression dataset."""
    return split_demo_regression_dataset(load_demo_regression_dataset())


def init_demo(force: bool = False) -> Path:
    """Create or refresh the experiment directory; seed ``experiment.md`` (overwrite when ``force``)."""
    return init_experiment_dir(EXPERIMENT_ID, TEMPLATE_PATH, force=force)


if __name__ == "__main__":
    raise SystemExit(
        run_runner_main(EXPERIMENT_ID, CANDIDATE_RUNNERS, _load_splits, TEMPLATE_PATH)
    )
