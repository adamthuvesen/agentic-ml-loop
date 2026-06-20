from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from lib.io import load_json

SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")


def _require_string(payload: dict[str, Any], key: str, errors: list[str]) -> None:
    if not isinstance(payload.get(key), str) or not payload[key].strip():
        errors.append(f"{key} must be a non-empty string")


def _is_safe_relative_path(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def validate_fixture_manifest(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for an external-target fixture manifest."""
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("fixture_id", "target_repo", "description", "split_policy"):
        _require_string(payload, key, errors)

    data_files = payload.get("data_files")
    if not isinstance(data_files, list) or not data_files:
        errors.append("data_files must be a non-empty list")
    else:
        seen_paths: set[str] = set()
        for index, item in enumerate(data_files):
            prefix = f"data_files[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            path = item.get("path")
            if not _is_safe_relative_path(path):
                errors.append(f"{prefix}.path must be a safe relative path")
            elif str(path) in seen_paths:
                errors.append(f"{prefix}.path duplicates {path!r}")
            else:
                seen_paths.add(str(path))
            if not isinstance(item.get("rows"), int) or item["rows"] <= 0:
                errors.append(f"{prefix}.rows must be a positive integer")
            if not isinstance(item.get("sha256"), str) or not SHA256_RE.match(item["sha256"]):
                errors.append(f"{prefix}.sha256 must be a 64-character hex digest")

    expected_metrics = payload.get("expected_metrics")
    if not isinstance(expected_metrics, list) or not expected_metrics:
        errors.append("expected_metrics must be a non-empty list")
    else:
        for index, item in enumerate(expected_metrics):
            prefix = f"expected_metrics[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            for key in ("candidate_id", "split", "metric"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    errors.append(f"{prefix}.{key} must be a non-empty string")
            if not isinstance(item.get("value"), int | float):
                errors.append(f"{prefix}.value must be numeric")
            if not isinstance(item.get("tolerance"), int | float) or item["tolerance"] < 0:
                errors.append(f"{prefix}.tolerance must be a non-negative number")

    leakage_tests = payload.get("leakage_negative_tests", [])
    if not isinstance(leakage_tests, list):
        errors.append("leakage_negative_tests must be a list when present")
    else:
        for index, item in enumerate(leakage_tests):
            prefix = f"leakage_negative_tests[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{prefix} must be an object")
                continue
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                errors.append(f"{prefix}.name must be a non-empty string")
            columns = item.get("forbidden_columns")
            if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
                errors.append(f"{prefix}.forbidden_columns must be a list of strings")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an external-target fixture manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    payload = load_json(args.manifest)
    if not isinstance(payload, dict):
        print("error: fixture manifest must be a JSON object", file=sys.stderr)
        return 1
    errors = validate_fixture_manifest(payload)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Fixture manifest valid: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
