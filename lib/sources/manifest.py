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

from lib.sources.errors import SnapshotIntegrityError
from lib.utils import load_json, utc_now, write_json

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


def schema_fingerprint(schema: pa.Schema) -> str:
    """A stable hash of field names + types (ignores schema metadata)."""
    parts = [f"{field.name}:{field.type}" for field in schema]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def build_manifest(
    table: pa.Table,
    *,
    source_type: str,
    driver: str,
    query: str,
    as_of: str | None,
    sample_seed: int | None,
    snapshot_filename: str = SNAPSHOT_FILENAME,
) -> DatasetManifest:
    """Build a manifest from an extracted Arrow *table* (does not import pyarrow)."""
    schema = table.schema
    return DatasetManifest(
        manifest_version=MANIFEST_VERSION,
        source_type=source_type,
        driver=driver,
        query=query,
        as_of=as_of,
        row_count=table.num_rows,
        column_count=len(schema),
        columns=[field.name for field in schema],
        schema_hash=schema_fingerprint(schema),
        sample_seed=sample_seed,
        snapshot_filename=snapshot_filename,
        pulled_at=utc_now(),
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
