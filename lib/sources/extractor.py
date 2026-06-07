"""The ``WarehouseExtractor`` contract and the factory that builds one.

The whole package speaks one shape: run a read-only query, get a
``pyarrow.Table``. ``get_extractor`` is the single place that maps a source type
to its concrete adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from lib.sources.adapters import (
    ADBC_DBAPI,
    SOURCE_EXTRAS,
    SUPPORTED_SOURCE_TYPES,
    AdbcExtractor,
    DatabricksExtractor,
    DuckDbExtractor,
    RedshiftExtractor,
)

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["WarehouseExtractor", "get_extractor", "SUPPORTED_SOURCE_TYPES"]


@runtime_checkable
class WarehouseExtractor(Protocol):
    """Runs one read-only query against a warehouse and returns Arrow."""

    source_type: str
    driver: str

    def extract(self, query: str) -> pa.Table:
        """Execute *query* read-only and return the result as a ``pyarrow.Table``."""
        ...


def get_extractor(source_type: str, config: dict[str, Any] | None = None) -> WarehouseExtractor:
    """Build the adapter for *source_type*.

    Raises ``ValueError`` for an unknown source type. The returned adapter does
    not touch the network or import any client until ``extract`` is called.
    """
    if source_type not in SOURCE_EXTRAS:
        supported = ", ".join(SUPPORTED_SOURCE_TYPES)
        raise ValueError(f"Unknown source type {source_type!r}. Supported: {supported}")
    config = config or {}
    if source_type in ADBC_DBAPI:
        return AdbcExtractor(source_type, config)
    if source_type == "databricks":
        return DatabricksExtractor(config)
    if source_type == "redshift":
        return RedshiftExtractor(config)
    return DuckDbExtractor(config)
