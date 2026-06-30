"""Freeze a warehouse query into a reproducible local parquet snapshot.

``freeze_snapshot`` is the edge step: guard the query for determinism, apply the
as-of, extract Arrow, write ``snapshot.parquet`` + ``dataset_manifest.json``.
``read_snapshot`` is what an experiment's ``data.py`` calls to load the frozen
data (verifying integrity first).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.paths import DATA_DIRNAME
from lib.sources.extractor import WarehouseExtractor, get_extractor
from lib.sources.manifest import (
    MANIFEST_FILENAME,
    SNAPSHOT_FILENAME,
    DatasetManifest,
    DatasetManifestRequest,
    dataset_manifest_from_table,
    verify_snapshot,
)
from lib.sources.query import apply_as_of, require_deterministic_query

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "SnapshotFreezeRequest",
    "freeze_snapshot",
    "read_snapshot",
    "SNAPSHOT_FILENAME",
    "MANIFEST_FILENAME",
]


@dataclass(frozen=True)
class SnapshotFreezeRequest:
    source_type: str
    query: str
    out_dir: Path
    config: dict[str, Any] | None = None
    as_of: str | None = None
    sample_seed: int | None = None
    allow_nondeterministic: bool = False
    extractor: WarehouseExtractor | None = None


def freeze_snapshot(
    request: SnapshotFreezeRequest | None = None, **legacy_kwargs: Any
) -> DatasetManifest:
    """Extract *query* read-only and freeze it under *out_dir*.

    Returns the written :class:`DatasetManifest`. Pass *extractor* to inject a
    pre-built or fake adapter (used in tests); otherwise one is built from
    *source_type* and *config*.
    """
    request = _coerce_snapshot_freeze_request(request, legacy_kwargs)
    require_deterministic_query(
        request.query, allow_nondeterministic=request.allow_nondeterministic
    )
    rewritten_query, recorded_as_of = apply_as_of(request.query, request.source_type, request.as_of)

    extractor = request.extractor or get_extractor(request.source_type, request.config)
    table = extractor.extract(rewritten_query)

    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(table, out_dir / SNAPSHOT_FILENAME)

    manifest = dataset_manifest_from_table(
        DatasetManifestRequest(
            table=table,
            source_type=request.source_type,
            driver=extractor.driver,
            query=request.query,
            as_of=recorded_as_of,
            sample_seed=request.sample_seed,
        )
    )
    manifest.write(out_dir / MANIFEST_FILENAME)
    return manifest


def _coerce_snapshot_freeze_request(
    request: SnapshotFreezeRequest | None,
    legacy_kwargs: dict[str, Any],
) -> SnapshotFreezeRequest:
    if request is not None:
        if legacy_kwargs:
            raise TypeError(
                "freeze_snapshot() received both a SnapshotFreezeRequest and "
                "legacy keyword arguments"
            )
        return request

    allowed = {
        "source_type",
        "query",
        "out_dir",
        "config",
        "as_of",
        "sample_seed",
        "allow_nondeterministic",
        "extractor",
    }
    unknown = sorted(set(legacy_kwargs) - allowed)
    if unknown:
        raise TypeError("freeze_snapshot() got unexpected keyword arguments: " + ", ".join(unknown))

    missing = [key for key in ("source_type", "query", "out_dir") if key not in legacy_kwargs]
    if missing:
        raise TypeError(
            "freeze_snapshot() missing required keyword arguments: " + ", ".join(missing)
        )

    return SnapshotFreezeRequest(
        source_type=legacy_kwargs["source_type"],
        query=legacy_kwargs["query"],
        out_dir=legacy_kwargs["out_dir"],
        config=legacy_kwargs.get("config"),
        as_of=legacy_kwargs.get("as_of"),
        sample_seed=legacy_kwargs.get("sample_seed"),
        allow_nondeterministic=bool(legacy_kwargs.get("allow_nondeterministic", False)),
        extractor=legacy_kwargs.get("extractor"),
    )


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
