# PostgreSQL source

The creds-free default for a "real database" demo: a local Postgres with `trust`
auth needs no password at all. See `examples/warehouse/postgres/` for a
docker-compose that stands one up with a SELECT-only role.

## Install

```bash
uv sync --extra postgres   # adbc-driver-postgresql + manager + pyarrow
```

## Auth (keyless local)

- **Local demo:** Postgres with `trust`/`peer` auth — no password. Connect as the
  SELECT-only role: `postgresql://demo_ro@localhost:5432/demo`.
- **Real targets:** keep the URI out of the repo. Use `PG*` env vars, `~/.pgpass`,
  or resolve it at runtime: `--uri "$(op read 'op://vault/pg-readonly/uri')"`.

## Read-only grant (the real guarantee)

```sql
CREATE ROLE demo_ro LOGIN;
GRANT USAGE ON SCHEMA public TO demo_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO demo_ro;   -- or GRANT pg_read_all_data
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO demo_ro;
```

A SELECT-only role makes writes impossible regardless of the client.

## As-of

Postgres has no time travel. Model the as-of as a predicate
(`WHERE created_at < '2026-01-01'`); the value is recorded in the manifest.

## Caveats

- High-precision `NUMERIC` and `TIMESTAMPTZ` are preserved via Arrow/ADBC; if you
  route through pandas elsewhere, watch decimal->float and tz drift.
- `JSONB` and array columns arrive as strings/objects; flatten in SQL if needed.

## Example

```bash
python -m lib.sources pull \
  --experiment experiments/demo_warehouse \
  --source postgres \
  --uri "postgresql://demo_ro@localhost:5432/demo" \
  --query "SELECT * FROM events WHERE created_at < '2026-01-01' ORDER BY id"
```
