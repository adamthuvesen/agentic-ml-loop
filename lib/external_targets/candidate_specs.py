from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def validate_candidate_id(candidate_id: str) -> str:
    if not _CANDIDATE_ID_RE.match(candidate_id):
        raise ValueError(
            "candidate ids must use lowercase letters, digits, '.', '_' or '-' "
            f"and must not contain path separators: {candidate_id!r}"
        )
    return candidate_id


def candidate_spec_path(experiment_dir: Path, candidate_id: str) -> Path:
    return experiment_dir / "work" / "candidates" / f"{validate_candidate_id(candidate_id)}.json"


def load_candidate_spec(
    candidate_id: str,
    bundled_candidates: dict[str, dict[str, Any]],
    experiment_dir: Path | None,
) -> dict[str, Any]:
    if candidate_id in bundled_candidates:
        return dict(bundled_candidates[candidate_id])
    if experiment_dir is None:
        known = ", ".join(sorted(bundled_candidates))
        raise ValueError(
            f"Unknown bundled candidate {candidate_id!r}; known={known}. "
            "Pass --experiment-dir to load local specs from work/candidates/."
        )

    path = candidate_spec_path(experiment_dir, candidate_id)
    if not path.exists():
        known = ", ".join(sorted(bundled_candidates))
        raise ValueError(
            f"Unknown candidate {candidate_id!r}; expected bundled candidate "
            f"({known}) or local spec {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    declared_id = payload.setdefault("candidate_id", candidate_id)
    if declared_id != candidate_id:
        raise ValueError(f"{path} declares candidate_id={declared_id!r}, expected {candidate_id!r}")
    required = ("family", "feature_set", "mode", "notes")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"{path} missing required keys: {missing}")
    payload["local_candidate_spec"] = str(path)
    return payload


def apply_derived_features(
    frame: pd.DataFrame,
    feature_specs: list[dict[str, Any]] | None,
    *,
    forbidden_sources: set[str],
) -> pd.DataFrame:
    if not feature_specs:
        return frame
    out = frame.copy()
    for spec in feature_specs:
        if not isinstance(spec, dict):
            raise ValueError(f"Derived feature spec must be an object, got {spec!r}")
        name = str(spec.get("name", "")).strip()
        validate_candidate_id(name)
        if name in forbidden_sources:
            raise ValueError(f"Derived feature name {name!r} is forbidden")
        op = str(spec.get("op", "")).strip()
        if not op:
            raise ValueError(f"Derived feature {name!r} is missing op")
        out[name] = _compute_derived_feature(out, spec, forbidden_sources)
    return out


def _source(frame: pd.DataFrame, column: str, forbidden_sources: set[str]) -> pd.Series:
    if column in forbidden_sources:
        raise ValueError(f"Derived feature source {column!r} is forbidden")
    if column not in frame.columns:
        raise ValueError(f"Derived feature source {column!r} is missing")
    return pd.to_numeric(frame[column], errors="coerce")


def _finite(values: pd.Series | np.ndarray) -> pd.Series:
    series = pd.Series(values)
    return series.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _log1p_feature(source: pd.Series, spec: dict[str, Any]) -> pd.Series:
    lower = float(spec.get("lower", 0.0))
    return _finite(np.log1p(source.clip(lower=lower)))


def _sqrt_feature(source: pd.Series, spec: dict[str, Any]) -> pd.Series:
    lower = float(spec.get("lower", 0.0))
    return _finite(np.sqrt(source.clip(lower=lower)))


def _square_feature(source: pd.Series, _spec: dict[str, Any]) -> pd.Series:
    return _finite(source**2)


def _clip_feature(source: pd.Series, spec: dict[str, Any]) -> pd.Series:
    min_value = spec.get("min")
    max_value = spec.get("max")
    return _finite(
        source.clip(
            lower=float(min_value) if min_value is not None else None,
            upper=float(max_value) if max_value is not None else None,
        )
    )


_SINGLE_SOURCE_OPS: dict[str, Callable[[pd.Series, dict[str, Any]], pd.Series]] = {
    "log1p": _log1p_feature,
    "sqrt": _sqrt_feature,
    "square": _square_feature,
    "clip": _clip_feature,
}

_BINARY_OPS: dict[str, Callable[[pd.Series, pd.Series], pd.Series]] = {
    "ratio": lambda left, right: left / right.replace(0, np.nan),
    "difference": lambda left, right: left - right,
    "product": lambda left, right: left * right,
    "sum": lambda left, right: left + right,
}

_THRESHOLD_OPS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "greater_than": lambda source, threshold: (source > threshold).astype(float),
    "less_than": lambda source, threshold: (source < threshold).astype(float),
}


def _compute_derived_feature(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    forbidden_sources: set[str],
) -> pd.Series:
    op = str(spec["op"])
    if op in _SINGLE_SOURCE_OPS:
        source = _source(frame, str(spec["source"]), forbidden_sources)
        return _SINGLE_SOURCE_OPS[op](source, spec)

    if op in _BINARY_OPS:
        left = _source(frame, str(spec["left"]), forbidden_sources)
        right = _source(frame, str(spec["right"]), forbidden_sources)
        return _finite(_BINARY_OPS[op](left, right))

    if op in _THRESHOLD_OPS:
        source = _source(frame, str(spec["source"]), forbidden_sources)
        threshold = float(spec["threshold"])
        return _THRESHOLD_OPS[op](source, threshold)

    raise ValueError(f"Unsupported derived feature op {op!r}")
