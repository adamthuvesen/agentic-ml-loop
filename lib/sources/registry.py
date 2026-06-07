"""Declarative source bundles + a tiny enable/list registry.

Each warehouse ships a bundle directory under ``bundles/<name>/`` with a
``source.json`` (how to reach it, the required read-only grant, auth modes — all
by reference, never a credential) and a ``SETUP.md``. Enabling a source records
intent in a local, gitignored user-config file and points at its SETUP.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.utils import load_json, write_json

BUNDLES_DIR = Path(__file__).resolve().parent / "bundles"
_CONFIG_PATH = Path.home() / ".config" / "agentic-ml-loop" / "sources.json"


@dataclass(frozen=True)
class Bundle:
    source_type: str
    display_name: str
    extra: str
    package: str
    driver: str
    time_travel: bool
    description: str
    read_only_grant: str
    auth_modes: list[str]
    credentials_via: str
    directory: Path

    @property
    def setup_path(self) -> Path:
        return self.directory / "SETUP.md"


def _load_bundle(directory: Path) -> Bundle:
    data = load_json(directory / "source.json")
    return Bundle(
        source_type=data["source_type"],
        display_name=data["display_name"],
        extra=data["extra"],
        package=data["package"],
        driver=data["driver"],
        time_travel=bool(data["time_travel"]),
        description=data["description"],
        read_only_grant=data["read_only_grant"],
        auth_modes=list(data["auth_modes"]),
        credentials_via=data["credentials_via"],
        directory=directory,
    )


def list_bundles() -> list[Bundle]:
    """Return all source bundles, sorted by source type."""
    if not BUNDLES_DIR.is_dir():
        return []
    bundles = [
        _load_bundle(child)
        for child in sorted(BUNDLES_DIR.iterdir())
        if (child / "source.json").exists()
    ]
    return bundles


def get_bundle(source_type: str) -> Bundle:
    for bundle in list_bundles():
        if bundle.source_type == source_type:
            return bundle
    raise KeyError(f"No source bundle named {source_type!r}")


def enabled_sources() -> set[str]:
    if not _CONFIG_PATH.exists():
        return set()
    data = load_json(_CONFIG_PATH)
    return set(data.get("enabled", [])) if isinstance(data, dict) else set()


def is_enabled(source_type: str) -> bool:
    return source_type in enabled_sources()


def enable_source(source_type: str) -> None:
    get_bundle(source_type)  # validate it exists
    enabled = enabled_sources()
    enabled.add(source_type)
    write_json(_CONFIG_PATH, {"enabled": sorted(enabled)})
