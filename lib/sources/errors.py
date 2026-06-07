"""Errors raised by the warehouse ingestion layer.

All inherit from ``SourceError`` so callers can catch the whole family.
"""

from __future__ import annotations


class SourceError(RuntimeError):
    """Base class for warehouse ingestion errors."""


class MissingSourceExtra(SourceError):
    """A warehouse adapter was requested but its optional dependency is absent.

    The message names the extra to install so the failure is actionable instead
    of a raw ``ImportError`` traceback.
    """

    def __init__(self, source_type: str, extra: str, package: str) -> None:
        self.source_type = source_type
        self.extra = extra
        self.package = package
        super().__init__(
            f"Source '{source_type}' needs the optional '{extra}' extra "
            f"(provides {package}). Install it with: "
            f"uv sync --extra {extra}   # or: pip install 'agentic-ml-loop[{extra}]'"
        )


class NonDeterministicQueryError(SourceError):
    """A pull query would not reproduce the same rows on a re-pull."""


class SnapshotIntegrityError(SourceError):
    """A snapshot's parquet no longer matches its manifest."""
