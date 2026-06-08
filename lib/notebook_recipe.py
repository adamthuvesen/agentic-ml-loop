from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

RecipeType = Literal["final_model", "analysis_pipeline"]
ModelFamily = Literal["logistic_regression", "xgboost"]

SUPPORTED_RECIPE_TYPES: frozenset[str] = frozenset({"final_model", "analysis_pipeline"})
SUPPORTED_MODEL_FAMILIES: frozenset[str] = frozenset({"logistic_regression", "xgboost"})
RECIPE_FILENAME = "notebook.yaml"


class NotebookRecipeError(ValueError):
    """Raised when a portable notebook recipe is missing or invalid."""


@dataclass(frozen=True)
class NotebookRecipe:
    experiment_id: str
    title: str
    recipe_type: RecipeType
    data_path_hint: str
    output_filename: str
    target_column: str
    time_column: str
    eligibility_region_column: str | None = None
    eligibility_group_column: str | None = None
    eligible_regions: list[str] = field(default_factory=list)
    eligible_groups: list[str] = field(default_factory=list)
    train_dates: list[str] = field(default_factory=list)
    validation_dates: list[str] = field(default_factory=list)
    development_dates: list[str] = field(default_factory=list)
    test_dates: list[str] = field(default_factory=list)
    holdout_dates: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    selected_features: list[str] = field(default_factory=list)
    model_label: str | None = None
    model_family: ModelFamily = "logistic_regression"
    incumbent_score_column: str | None = None
    incumbent_probability_scale: float | None = None
    run_pipeline_default: bool = False
    safe_outputs: list[str] = field(default_factory=list)
    sensitive_outputs: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], experiment_id: str) -> NotebookRecipe:
        required = [
            "title",
            "recipe_type",
            "data_path_hint",
            "target_column",
            "time_column",
        ]
        missing = [key for key in required if key not in data or data[key] in ("", None)]
        if missing:
            raise NotebookRecipeError(
                "Notebook recipe is missing required fields: " + ", ".join(missing)
            )

        recipe_type = str(data["recipe_type"])
        if recipe_type not in SUPPORTED_RECIPE_TYPES:
            supported = ", ".join(sorted(SUPPORTED_RECIPE_TYPES))
            raise NotebookRecipeError(
                f"Unsupported recipe_type {recipe_type!r}; expected one of {supported}."
            )

        output_filename = _validate_output_filename(
            str(data.get("output_filename") or f"{experiment_id}_portable.ipynb")
        )

        recipe = cls(
            experiment_id=experiment_id,
            title=str(data["title"]),
            recipe_type=recipe_type,  # type: ignore[arg-type]
            data_path_hint=str(data["data_path_hint"]),
            output_filename=output_filename,
            target_column=str(data["target_column"]),
            time_column=str(data["time_column"]),
            eligibility_region_column=_optional_str(data.get("eligibility_region_column")),
            eligibility_group_column=_optional_str(data.get("eligibility_group_column")),
            eligible_regions=_string_list(data.get("eligible_regions")),
            eligible_groups=_string_list(data.get("eligible_groups")),
            train_dates=_string_list(data.get("train_dates")),
            validation_dates=_string_list(data.get("validation_dates")),
            development_dates=_string_list(data.get("development_dates")),
            test_dates=_string_list(data.get("test_dates")),
            holdout_dates=_string_list(data.get("holdout_dates")),
            exclude_columns=_string_list(data.get("exclude_columns")),
            selected_features=_string_list(data.get("selected_features")),
            model_label=_optional_str(data.get("model_label")),
            model_family=_parse_model_family(data.get("model_family")),
            incumbent_score_column=_optional_str(data.get("incumbent_score_column")),
            incumbent_probability_scale=_optional_float(data.get("incumbent_probability_scale")),
            run_pipeline_default=bool(data.get("run_pipeline_default", False)),
            safe_outputs=_string_list(data.get("safe_outputs")),
            sensitive_outputs=_string_list(data.get("sensitive_outputs")),
        )
        recipe.validate()
        return recipe

    def validate(self) -> None:
        if self.recipe_type == "final_model":
            if not self.selected_features:
                raise NotebookRecipeError("final_model recipes must declare selected_features.")
            if not self.development_dates or not self.test_dates:
                raise NotebookRecipeError(
                    "final_model recipes must declare development_dates and test_dates."
                )
            if self.model_family not in SUPPORTED_MODEL_FAMILIES:
                supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
                raise NotebookRecipeError(
                    f"Unsupported model_family {self.model_family!r}; expected one of {supported}."
                )
        if self.recipe_type == "analysis_pipeline" and (
            not self.train_dates or not self.validation_dates
        ):
            raise NotebookRecipeError(
                "analysis_pipeline recipes must declare train_dates and validation_dates."
            )

        for label, paths in (
            ("safe_outputs", self.safe_outputs),
            ("sensitive_outputs", self.sensitive_outputs),
        ):
            for raw_path in paths:
                path = Path(raw_path)
                if path.is_absolute() or ".." in path.parts:
                    raise NotebookRecipeError(
                        f"{label} contains unsafe relative path: {raw_path!r}."
                    )

        overlap = set(self.safe_outputs) & set(self.sensitive_outputs)
        if overlap:
            raise NotebookRecipeError(
                "Outputs cannot be both safe and sensitive: " + ", ".join(sorted(overlap))
            )


def load_recipe(experiment_dir: Path) -> NotebookRecipe:
    experiment_dir = Path(experiment_dir)
    recipe_path = experiment_dir / RECIPE_FILENAME
    if not recipe_path.exists():
        raise NotebookRecipeError(f"Notebook recipe not found at {recipe_path}.")
    data = _parse_simple_yaml(recipe_path.read_text(encoding="utf-8"))
    return NotebookRecipe.from_mapping(data, experiment_dir.name)


def _validate_output_filename(value: str) -> str:
    if not value.endswith(".ipynb"):
        raise NotebookRecipeError("output_filename must end with .ipynb.")
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or path.name != value
        or ".." in path.parts
        or "\\" in value
    ):
        raise NotebookRecipeError(
            f"output_filename must be a simple notebook filename, got {value!r}."
        )
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small top-level YAML subset used by notebook recipes.

    Supported forms:
    - ``key: value``
    - ``key:`` followed by an indented ``- value`` list
    - quoted or unquoted scalar strings and booleans
    """
    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith("  - "):
            if current_key is None:
                raise NotebookRecipeError("Found list item without a key.")
            current = data.setdefault(current_key, [])
            if not isinstance(current, list):
                raise NotebookRecipeError(f"Field {current_key!r} mixes scalar and list values.")
            current.append(_parse_scalar(raw_line[4:].strip()))
            continue
        if raw_line.startswith(" "):
            raise NotebookRecipeError(
                "Only top-level keys and two-space scalar lists are supported in notebook.yaml."
            )
        if ":" not in raw_line:
            raise NotebookRecipeError(f"Invalid recipe line: {raw_line!r}.")
        key, raw_value = raw_line.split(":", 1)
        current_key = key.strip()
        value = raw_value.strip()
        data[current_key] = [] if value == "" else _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _parse_model_family(value: Any) -> ModelFamily:
    if value in (None, ""):
        return "logistic_regression"
    candidate = str(value).strip().lower()
    if candidate not in SUPPORTED_MODEL_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_FAMILIES))
        raise NotebookRecipeError(
            f"Unsupported model_family {value!r}; expected one of {supported}."
        )
    return candidate  # type: ignore[return-value]
