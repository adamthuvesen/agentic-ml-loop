"""Canonical experiment output directory helpers.

Each experiment under ``experiments/<exp_id>/`` follows a three-folder layout:

- ``outputs/`` — deliverables a stakeholder reads (final reports, shortlists, ranked CSVs).
- ``work/``    — intermediate artefacts one cycle writes for another to read.
- ``scripts/`` — one-shot Python scripts the agent writes during cycles.
- ``data/``    — materialized input data (e.g. a warehouse snapshot + manifest).

Cycle scripts should import these helpers and write to the returned ``Path``
instead of ``experiment_dir / "..."`` directly.
"""

from __future__ import annotations

from pathlib import Path

OUTPUTS_DIRNAME = "outputs"
WORK_DIRNAME = "work"
SCRIPTS_DIRNAME = "scripts"
DATA_DIRNAME = "data"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/outputs``, creating it if needed."""
    return _ensure(Path(experiment_dir) / OUTPUTS_DIRNAME)


def work_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/work``, creating it if needed."""
    return _ensure(Path(experiment_dir) / WORK_DIRNAME)


def scripts_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/scripts``, creating it if needed."""
    return _ensure(Path(experiment_dir) / SCRIPTS_DIRNAME)


def data_dir(experiment_dir: Path) -> Path:
    """Return ``experiment_dir/data`` (materialized input snapshots), creating it if needed."""
    return _ensure(Path(experiment_dir) / DATA_DIRNAME)
