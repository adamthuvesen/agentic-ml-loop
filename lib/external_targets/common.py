from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from lib.candidate_result import CandidateResult

ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "RESULT_JSON:"


def resolve_external_repo(env_var: str, repo_name: str) -> Path:
    """Resolve a sibling external repo, allowing a local override."""
    candidates: list[Path] = []
    if override := os.getenv(env_var):
        candidates.append(Path(override).expanduser())
    candidates.extend(
        [
            ROOT.parent / repo_name,
            Path.home() / "dev" / "menti" / repo_name,
        ]
    )
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {repo_name}; checked {checked}")


def parse_helper_result(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(f"Evaluator did not emit {RESULT_PREFIX!r}. stdout:\n{stdout}")


def run_uv_project_helper(
    *,
    repo: Path,
    helper: Path,
    candidate_id: str,
    extra_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    command = [
        "uv",
        "run",
        "--project",
        str(repo),
        "python",
        str(helper),
        "--repo",
        str(repo),
        "--candidate",
        candidate_id,
    ]
    if extra_args:
        command.extend(extra_args)

    child_env = os.environ.copy()
    child_env.pop("VIRTUAL_ENV", None)
    if env:
        child_env.update(env)

    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=child_env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise RuntimeError(
            f"External evaluator failed for {candidate_id!r} in {repo} "
            f"(exit {completed.returncode})"
        )
    return parse_helper_result(completed.stdout)


def candidate_result_from_payload(payload: dict[str, Any]) -> CandidateResult:
    return CandidateResult(
        candidate_id=str(payload["candidate_id"]),
        model_family=str(payload["model_family"]),
        feature_set=str(payload["feature_set"]),
        objective_metric=str(payload["objective_metric"]),
        objective_score=float(payload["objective_score"]),
        split_strategy=str(payload["split_strategy"]),
        status=str(payload.get("status", "completed")),
        notes=str(payload["notes"]),
        metrics=payload["metrics"],
        hyperparameters=payload.get("hyperparameters", {}),
        selected_features=list(payload.get("selected_features", [])),
    )


def emit_result(payload: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, sort_keys=True)}")
