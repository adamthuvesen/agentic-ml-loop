from __future__ import annotations

import pytest

from lib.notebook_recipe import NotebookRecipe, NotebookRecipeError


def _base_recipe(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "title": "Portable export",
        "recipe_type": "analysis_pipeline",
        "data_path_hint": "snapshot.parquet",
        "target_column": "target",
        "time_column": "quarter",
        "train_dates": ["2026-Q1"],
        "validation_dates": ["2026-Q2"],
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    "output_filename",
    [
        "../escaped.ipynb",
        "nested/../../escaped.ipynb",
        "/tmp/escaped.ipynb",
    ],
)
def test_output_filename_rejects_unsafe_paths(output_filename: str) -> None:
    with pytest.raises(NotebookRecipeError, match="output_filename"):
        NotebookRecipe.from_mapping(_base_recipe(output_filename=output_filename), "exp")


def test_output_filename_accepts_simple_notebook_name() -> None:
    recipe = NotebookRecipe.from_mapping(
        _base_recipe(output_filename="portable_export.ipynb"),
        "exp",
    )

    assert recipe.output_filename == "portable_export.ipynb"
