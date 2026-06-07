from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORT_PREFIXES = (
    "lib.demo_bootstrap",
    "lib.demo_classification",
    "lib.demo_regression",
    "runners.",
)
BOUNDARY_PATHS = [
    *sorted((ROOT / "loop").glob("*.py")),
    *sorted(path for path in (ROOT / "lib").glob("*.py") if path.name != "__init__.py"),
    *sorted((ROOT / "viz").glob("*.py")),
]
ALLOWLIST = {
    # Dynamic experiment data loading for validation-total summaries is documented
    # and avoids a direct import of any concrete experiment package.
    ROOT / "viz" / "generate.py",
}


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _imported_names_from_module(path: Path, module_name: str) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            names.extend(alias.name for alias in node.names)
    return names


def test_framework_modules_do_not_import_concrete_experiments() -> None:
    violations: list[str] = []
    for path in BOUNDARY_PATHS:
        if path in ALLOWLIST:
            continue
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.relative_to(ROOT)} imports {module}")

    assert violations == []


def test_modules_do_not_reach_into_private_lib_io_helpers() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "lib").glob("*.py")) + sorted((ROOT / "loop").glob("*.py")):
        if path.name == "io.py":
            continue
        private_imports = [
            name for name in _imported_names_from_module(path, "lib.io") if name.startswith("_")
        ]
        if private_imports:
            violations.append(
                f"{path.relative_to(ROOT)} imports private lib.io names: "
                f"{', '.join(private_imports)}"
            )

    assert violations == []
