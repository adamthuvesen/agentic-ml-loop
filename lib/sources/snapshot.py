"""Freeze a warehouse query into a reproducible local parquet snapshot.

``freeze_snapshot`` is the edge step: guard the query for determinism, apply the
as-of, extract Arrow, write ``snapshot.parquet`` + ``dataset_manifest.json``.
``read_snapshot`` is what an experiment's ``data.py`` calls to load the frozen
data (verifying integrity first).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.paths import DATA_DIRNAME
from lib.sources.extractor import WarehouseExtractor, get_extractor
from lib.sources.manifest import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    DatasetManifest,
    build_manifest,
    verify_snapshot,
)
from lib.sources.query import apply_as_of, ensure_deterministic

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["freeze_snapshot", "read_snapshot", "SNAPSHOT_FILENAME", "MANIFEST_FILENAME"]


def freeze_snapshot(
    *,
    source_type: str,
    query: str,
    out_dir: Path,
    config: dict[str, Any] | None = None,
    as_of: str | None = None,
    sample_seed: int | None = None,
    allow_nondeterministic: bool = False,
    extractor: WarehouseExtractor | None = None,
) -> DatasetManifest:
    """Extract *query* read-only and freeze it under *out_dir*.

    Returns the written :class:`DatasetManifest`. Pass *extractor* to inject a
    pre-built or fake adapter (used in tests); otherwise one is built from
    *source_type* and *config*.
    """
    ensure_deterministic(query, allow_nondeterministic=allow_nondeterministic)
    rewritten_query, recorded_as_of = apply_as_of(query, source_type, as_of)

    extractor = extractor or get_extractor(source_type, config)
    table = extractor.extract(rewritten_query)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(table, out_dir / SNAPSHOT_FILENAME)

    manifest = build_manifest(
        table,
        source_type=source_type,
        driver=extractor.driver,
        query=query,
        as_of=recorded_as_of,
        sample_seed=sample_seed,
    )
    manifest.write(out_dir / MANIFEST_FILENAME)
    return manifest


def read_snapshot(experiment_dir: Path, *, verify: bool = True) -> pd.DataFrame:
    """Load an experiment's frozen snapshot as a pandas DataFrame.

    Verifies the snapshot against its manifest first (when both are present);
    raises :class:`SnapshotIntegrityError` on drift.
    """
    import pyarrow.parquet as pq

    data_dir = Path(experiment_dir) / DATA_DIRNAME
    snapshot_path = data_dir / SNAPSHOT_FILENAME
    manifest_path = data_dir / MANIFEST_FILENAME
    if not snapshot_path.exists():
        raise FileNotFoundError(f"No snapshot at {snapshot_path}")
    if verify and manifest_path.exists():
        verify_snapshot(snapshot_path, DatasetManifest.load(manifest_path))
    return pq.read_table(str(snapshot_path)).to_pandas()


def _write_parquet(table: Any, path: Path) -> None:
    import pyarrow.parquet as pq

    # store_schema=True (pyarrow's default, set explicitly) embeds the Arrow
    # schema in the parquet footer so verify_snapshot's schema hash round-trips.
    pq.write_table(table, str(path), compression="zstd", store_schema=True)
