from __future__ import annotations

from pathlib import Path

from lib.demo_deep.data import load_demo_dataset, split_demo_dataset
from lib.demo_deep.modeling import CANDIDATE_RUNNERS
from lib.runner import RunnerMainConfig, init_experiment_dir, run_runner_main

EXPERIMENT_ID = "demo_deep"
TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "templates" / "demo_deep.md"


def _load_splits():
    return split_demo_dataset(load_demo_dataset())


def init_demo(force: bool = False) -> Path:
    return init_experiment_dir(EXPERIMENT_ID, TEMPLATE_PATH, force=force)


if __name__ == "__main__":
    raise SystemExit(
        run_runner_main(
            RunnerMainConfig(
                experiment_id=EXPERIMENT_ID,
                candidate_runners=CANDIDATE_RUNNERS,
                dataset_loader=_load_splits,
                template_path=TEMPLATE_PATH,
            )
        )
    )
