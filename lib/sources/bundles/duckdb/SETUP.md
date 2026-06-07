# DuckDB source

Tier-0, fully creds-free. Use it to prove the ingestion path end to end with no
server and no credentials, and as the bundled demo source.

## Install

```bash
uv sync --extra models   # duckdb + pyarrow already live in the models extra
```

## Auth

None. DuckDB reads a local file or `.duckdb` database. The extractor opens it
**read-only** (`read_only=True`) when it is a real file.

## As-of

DuckDB has no time travel. Model an as-of with a `WHERE` predicate on a time
column; the value is recorded in the manifest for provenance.

## Caveats

- An in-memory database (`:memory:`) cannot be opened read-only.
- Types map cleanly to Arrow; decimals and timestamps round-trip to parquet.

## Example

```bash
python -m lib.sources pull \
  --experiment experiments/demo_warehouse \
  --source duckdb \
  --database lib/sources/bundles/duckdb/demo.duckdb \
  --query "SELECT * FROM events ORDER BY snapshot_date"
```
