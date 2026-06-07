# Snowflake source

Bulk extract via ADBC (Arrow-native). There is no anonymous public Snowflake, so
this is bring-your-own-account; the repo carries no credentials.

## Install

```bash
uv sync --extra snowflake   # adbc-driver-snowflake + manager + pyarrow
```

## Auth (keyless local)

Use a **named connection** in `~/.snowflake/connections.toml` (shared with the
Snowflake CLI) with browser SSO — nothing secret in the repo:

```toml
[ml_ro]
account = "myorg-myacct"
user = "you@example.com"
authenticator = "externalbrowser"
role = "ANALYST_RO"
warehouse = "WH_XS"
database = "ANALYTICS"
schema = "MART"
```

Pass the connection via `--conn` db_kwargs (account/user/authenticator/role/...),
or wire the connector to read the named connection. Key-pair auth is the
unattended/CI option (private key outside the repo).

## Read-only grant (the real guarantee)

A dedicated **SELECT-only role** (`ANALYST_RO`): `USAGE` on warehouse/db/schema +
`SELECT` on the target tables/views, nothing else. Snowflake has **no** read-only
session flag, so the role grant is the boundary.

## As-of (reproducible)

Snowflake Time Travel. Put an `{as_of}` placeholder right after the table:

```sql
SELECT * FROM mart.events {as_of} ORDER BY event_id
```

`--as-of '2026-06-01 00:00:00'` rewrites it to
`AT(TIMESTAMP => '2026-06-01 00:00:00'::timestamp_tz)`. Bounded by the table's
Time Travel retention (1 day default; up to 90 on Enterprise).

## Caveats

- `NUMBER`/`DECIMAL` may arrive as float; a nullable integer column can become
  float. Cast large keys to string in SQL if exact values matter.
- Normalize `TIMESTAMP_TZ` to UTC in SQL so the snapshot is session-tz-independent.

## Example

```bash
python -m lib.sources pull \
  --experiment experiments/expansion \
  --source snowflake --as-of '2026-06-01 00:00:00' \
  --conn account=myorg-myacct --conn user=you@example.com \
  --conn authenticator=externalbrowser --conn role=ANALYST_RO \
  --conn warehouse=WH_XS --conn database=ANALYTICS --conn schema=MART \
  --query "SELECT * FROM mart.events {as_of} ORDER BY event_id"
```
