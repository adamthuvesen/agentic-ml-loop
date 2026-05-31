from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STALL_LIMIT = 3
DEFAULT_FAILURE_LIMIT = 3
DEFAULT_MAX_ATTEMPTS_PER_CYCLE = 3
STATE_PATH_NAME = "loop_state.json"
