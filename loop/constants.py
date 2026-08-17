from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STALL_LIMIT = 3
DEFAULT_FAILURE_LIMIT = 3
DEFAULT_MAX_ATTEMPTS_PER_CYCLE = 3
STATE_PATH_NAME = "loop_state.json"
CYCLE_STATUS_FILENAME = "cycle_status.json"


def cycle_artifacts_dir(experiment_dir: Path, cycle_id: str) -> Path:
    """Return the per-cycle artifact directory.

    The prompt names the completion-status path and the supervisor reads it, so
    both derive it here rather than each spelling out the layout.
    """
    return experiment_dir / "cycles" / cycle_id
