"""Warehouse ingestion: pull a read-only query into a frozen parquet snapshot.

Edge-only by design — a one-time ``pull`` materializes a reproducible snapshot
plus a provenance manifest; the loop then runs offline against that snapshot.
Importing this package is cheap and offline: pyarrow and every warehouse client
are imported lazily inside the functions that need them.
"""

from __future__ import annotations

from lib.sources.adapters import SOURCE_EXTRAS, SUPPORTED_SOURCE_TYPES
from lib.sources.errors import (
    MissingSourceExtra,
    NonDeterministicQueryError,
    SnapshotIntegrityError,
    SourceError,
)
from lib.sources.extractor import WarehouseExtractor, get_extractor
from lib.sources.manifest import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    DatasetManifest,
    build_manifest,
    verify_snapshot,
)
from lib.sources.query import apply_as_of, ensure_deterministic
from lib.sources.registry import Bundle, get_bundle, list_bundles
from lib.sources.snapshot import freeze_snapshot, read_snapshot

__all__ = [
    "Bundle",
    "DatasetManifest",
    "MANIFEST_FILENAME",
    "MissingSourceExtra",
    "NonDeterministicQueryError",
    "SNAPSHOT_FILENAME",
    "SOURCE_EXTRAS",
    "SUPPORTED_SOURCE_TYPES",
    "SnapshotIntegrityError",
    "SourceError",
    "WarehouseExtractor",
    "apply_as_of",
    "build_manifest",
    "ensure_deterministic",
    "freeze_snapshot",
    "get_bundle",
    "get_extractor",
    "list_bundles",
    "read_snapshot",
    "verify_snapshot",
]
