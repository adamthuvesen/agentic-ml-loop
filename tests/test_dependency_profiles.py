from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAVY_OPTIONAL_PACKAGES = {
    "catboost",
    "lightgbm",
    "matplotlib",
    "optuna",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "shap",
    "xgboost",
}


def _package_name(requirement: str) -> str:
    return requirement.split("[", 1)[0].split(">=", 1)[0].split("==", 1)[0]


def test_base_dependencies_exclude_heavy_optional_packages() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {_package_name(req) for req in pyproject["project"]["dependencies"]}

    assert base.isdisjoint(HEAVY_OPTIONAL_PACKAGES)


def test_optional_dependency_profiles_cover_modeling() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]
    models = {_package_name(req) for req in optional["models"]}
    dev = {_package_name(req) for req in pyproject["dependency-groups"]["dev"]}

    assert {"lightgbm", "xgboost", "scikit-learn"} <= models
    assert {"pytest", "ruff"} <= dev


def test_ci_installs_model_capable_test_environment() -> None:
    workflow = (ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert "uv sync --extra models --group dev" in workflow
    assert "uv run --extra models pytest" in workflow
