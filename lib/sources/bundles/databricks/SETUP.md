# Databricks source

Bulk extract from a Databricks SQL Warehouse via the native connector (its ADBC
driver is not a pip-clean Python wheel yet). Arrow-native through CloudFetch.

## Install

```bash
uv sync --extra databricks   # databricks-sql-connector[pyarrow] + pyarrow
```

## Auth (keyless local)

OAuth U2M (browser) is the keyless local default. Bootstrap a profile once:

```bash
databricks auth login --host https://<workspace-host>
```

Then pass connection kwargs (`server_hostname`, `http_path`, and the OAuth
`auth_type`) via `--conn`. M2M (service principal) is the CI option; inject the
secret via env, never inline. PAT works but is a long-lived bearer token — avoid.

## Read-only grant (the real guarantee)

Unity Catalog: `USE CATALOG` + `USE SCHEMA` + `SELECT` on the table, and **no**
`MODIFY`. Grant only `CAN USE` (not `CAN MANAGE`) on the SQL warehouse.

## As-of (reproducible)

Delta time travel via an `{as_of}` placeholder after the table:

```sql
SELECT * FROM main.schema.events {as_of} ORDER BY event_id
```

`--as-of v123` -> `VERSION AS OF 123` (preferred: versions are immutable, and the
`v` prefix keeps a version from being mistaken for a compact date);
`--as-of '2026-06-01T00:00:00Z'` -> `TIMESTAMP AS OF '...'`. Record the resolved
version in the manifest.

## Caveats

- **`ARRAY`/`MAP`/`STRUCT`/`VARIANT` come back as JSON strings**, not nested Arrow
  — the most likely fidelity surprise. Parse in SQL or accept stringly-typed
  columns. `DECIMAL` and `TIMESTAMP` are faithful.

## Example

```bash
python -m lib.sources pull \
  --experiment experiments/churn \
  --source databricks --as-of v412 \
  --conn server_hostname=<host> --conn http_path=/sql/1.0/warehouses/<id> \
  --conn auth_type=databricks-oauth \
  --query "SELECT * FROM main.schema.events {as_of} ORDER BY event_id"
```
