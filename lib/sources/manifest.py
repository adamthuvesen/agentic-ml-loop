"""The dataset snapshot manifest and its integrity check.

The manifest is the reproducible recipe for a frozen snapshot: source, exact
query, as-of, and a schema/row fingerprint. ``verify_snapshot`` re-reads the
parquet footer and fails loudly if the file drifted from its manifest, so a
stale or mutated snapshot can never silently poison the leaderboard.

Implemented as a frozen dataclass (matching the repo's stdlib + dataclass style)
with strict load validation rather than pulling in a new modelling dependency.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.io import load_json, utc_now, write_json
from lib.sources.errors import SnapshotIntegrityError

if TYPE_CHECKING:
    import pyarrow as pa

SNAPSHOT_FILENAME = "snapshot.parquet"
MANIFEST_FILENAME = "dataset_manifest.json"
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class DatasetManifest:
    """Provenance for one frozen snapshot. All fields are required on load."""

    manifest_version: int
    source_type: str
    driver: str
    query: str
    as_of: str | None
    row_count: int
    column_count: int
    columns: list[str]
    schema_hash: str
    sample_seed: int | None
    snapshot_filename: str
    pulled_at: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def write(self, path: Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        data = load_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a JSON object")
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path} has unknown manifest keys: {sorted(unknown)}")
        missing = known - set(data)
        if missing:
            raise ValueError(f"{path} is missing manifest keys: {sorted(missing)}")
        return cls(**data)


@dataclass(frozen=True)
class DatasetManifestRequest:
    table: pa.Table
    source_type: str
    driver: str
    query: str
    as_of: str | None
    sample_seed: int | None
    snapshot_filename: str = SNAPSHOT_FILENAME


def schema_fingerprint(schema: pa.Schema) -> str:
    """A stable hash of field names + types (ignores schema metadata)."""
    parts = [f"{field.name}:{field.type}" for field in schema]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def dataset_manifest_from_table(
    request: DatasetManifestRequest | Any | None = None, **legacy_kwargs: Any
) -> DatasetManifest:
    """Build a manifest from an extracted Arrow *table* (does not import pyarrow)."""
    request = _coerce_dataset_manifest_request(request, legacy_kwargs)
    schema = request.table.schema
    return DatasetManifest(
        manifest_version=MANIFEST_VERSION,
        source_type=request.source_type,
        driver=request.driver,
        query=request.query,
        as_of=request.as_of,
        row_count=request.table.num_rows,
        column_count=len(schema),
        columns=[field.name for field in schema],
        schema_hash=schema_fingerprint(schema),
        sample_seed=request.sample_seed,
        snapshot_filename=request.snapshot_filename,
        pulled_at=utc_now(),
    )


def _coerce_dataset_manifest_request(
    request: DatasetManifestRequest | Any | None,
    legacy_kwargs: dict[str, Any],
) -> DatasetManifestRequest:
    if isinstance(request, DatasetManifestRequest):
        if legacy_kwargs:
            raise TypeError(
                "dataset_manifest_from_table() received both a "
                "DatasetManifestRequest and legacy keyword arguments"
            )
        return request

    table = request if request is not None else legacy_kwargs.pop("table", None)
    if table is None:
        raise TypeError("dataset_manifest_from_table() missing required table")

    allowed = {
        "source_type",
        "driver",
        "query",
        "as_of",
        "sample_seed",
        "snapshot_filename",
    }
    unknown = sorted(set(legacy_kwargs) - allowed)
    if unknown:
        raise TypeError(
            "dataset_manifest_from_table() got unexpected keyword arguments: " + ", ".join(unknown)
        )

    missing = [
        key
        for key in ("source_type", "driver", "query", "as_of", "sample_seed")
        if key not in legacy_kwargs
    ]
    if missing:
        raise TypeError(
            "dataset_manifest_from_table() missing required keyword arguments: "
            + ", ".join(missing)
        )

    return DatasetManifestRequest(
        table=table,
        source_type=legacy_kwargs["source_type"],
        driver=legacy_kwargs["driver"],
        query=legacy_kwargs["query"],
        as_of=legacy_kwargs["as_of"],
        sample_seed=legacy_kwargs["sample_seed"],
        snapshot_filename=legacy_kwargs.get("snapshot_filename", SNAPSHOT_FILENAME),
    )


def verify_snapshot(snapshot_path: Path, manifest: DatasetManifest) -> None:
    """Raise :class:`SnapshotIntegrityError` if the parquet drifted from *manifest*.

    Reads only the parquet footer (schema + row count) — no full data load.
    Raises ``ModuleNotFoundError`` if pyarrow is not installed, so callers can
    treat "can't check" differently from "check failed".
    """
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(str(snapshot_path))
    schema = metadata.schema.to_arrow_schema()
    problems: list[str] = []
    if metadata.num_rows != manifest.row_count:
        problems.append(f"row count {metadata.num_rows} != manifest {manifest.row_count}")
    actual_hash = schema_fingerprint(schema)
    if actual_hash != manifest.schema_hash:
        problems.append("schema hash mismatch (columns or types changed)")
    if problems:
        raise SnapshotIntegrityError(f"{Path(snapshot_path).name}: " + "; ".join(problems))
