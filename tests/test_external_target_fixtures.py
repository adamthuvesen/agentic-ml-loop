from __future__ import annotations

from lib.external_targets.fixtures import validate_fixture_manifest


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "fixture_id": "demo-small-v1",
        "target_repo": "demo-target",
        "description": "Small fixture",
        "split_policy": "temporal train/validation/final",
        "data_files": [
            {
                "path": "fixtures/demo.csv",
                "rows": 12,
                "sha256": "a" * 64,
            }
        ],
        "expected_metrics": [
            {
                "candidate_id": "baseline",
                "split": "validation",
                "metric": "val_auc",
                "value": 0.5,
                "tolerance": 0.001,
            }
        ],
        "leakage_negative_tests": [{"name": "target_rejected", "forbidden_columns": ["target"]}],
    }


def test_validate_fixture_manifest_accepts_contract() -> None:
    assert validate_fixture_manifest(_valid_manifest()) == []


def test_validate_fixture_manifest_rejects_unsafe_paths_and_bad_hashes() -> None:
    manifest = _valid_manifest()
    manifest["data_files"] = [
        {
            "path": "../private.csv",
            "rows": 0,
            "sha256": "not-a-hash",
        }
    ]

    errors = validate_fixture_manifest(manifest)

    assert "data_files[0].path must be a safe relative path" in errors
    assert "data_files[0].rows must be a positive integer" in errors
    assert "data_files[0].sha256 must be a 64-character hex digest" in errors


def test_validate_fixture_manifest_rejects_duplicate_data_paths() -> None:
    manifest = _valid_manifest()
    manifest["data_files"] = [
        {"path": "fixtures/demo.csv", "rows": 12, "sha256": "a" * 64},
        {"path": "fixtures/demo.csv", "rows": 12, "sha256": "b" * 64},
    ]

    errors = validate_fixture_manifest(manifest)

    assert "data_files[1].path duplicates 'fixtures/demo.csv'" in errors
