from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.notebook_codegen import _build_notebook_cells
from lib.notebook_recipe import load_recipe
from lib.paths import outputs_dir

__all__ = ["generate_notebook", "load_recipe"]


def generate_notebook(experiment_dir: Path, *, include_sensitive: bool = False) -> Path:
    experiment_dir = Path(experiment_dir).resolve()
    recipe = load_recipe(experiment_dir)
    cells = _build_notebook_cells(recipe, include_sensitive=include_sensitive)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output_path = outputs_dir(experiment_dir) / recipe.output_filename
    output_path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a portable experiment notebook."
    )
    parser.add_argument("experiment", help="Experiment directory, e.g. experiments/foo")
    parser.add_argument(
        "--include-sensitive",
        action="store_true",
        help="Include recipe-declared sensitive outputs in the generated notebook.",
    )
    args = parser.parse_args(argv)
    path = generate_notebook(
        Path(args.experiment), include_sensitive=args.include_sensitive
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
