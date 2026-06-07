# Amazon Redshift source

Bulk extract via the native `redshift-connector`. There is no free local
Redshift; for a creds-free demo use the **Postgres** source as the stand-in (same
extract path), and treat this as the production overlay.

## Install

```bash
uv sync --extra redshift   # redshift-connector + pyarrow
```

## Auth (keyless local)

IAM via an AWS SSO **profile** — no DB password:

```bash
aws sso login --profile redshift-ro
```

Pass `--conn iam=true --conn profile=redshift-ro --conn cluster_identifier=<id>
--conn database=dev --conn db_user=ro_user --conn region=<region>`. Serverless
uses `--conn is_serverless=true --conn serverless_work_group=<wg>`. Secrets
Manager is the equivalent for CI.

## Read-only grant (the real guarantee)

A SELECT-only DB role plus a read-scoped IAM principal:

```sql
CREATE USER ro_user PASSWORD DISABLE;          -- IAM-only login
GRANT USAGE ON SCHEMA analytics TO ro_user;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO ro_user;
```

Redshift does **not** honor Postgres `default_transaction_read_only`, so the
grant is the boundary. IAM: allow only `redshift:GetClusterCredentials` +
`redshift-data:*Statement*`, no write/DDL.

## As-of

Redshift has **no time travel**. Model the as-of as a predicate
(`WHERE event_ts < '2026-06-01'`); for a consistent multi-table read, pull inside
one transaction (default SNAPSHOT isolation). The value is recorded in the manifest.

## Caveats

- `DECIMAL` arrives as `Decimal` objects and `SUPER`/JSON as strings; cast
  deliberately. Normalize `TIMESTAMPTZ` to UTC. (No Arrow-native fetch, so results
  pass through pandas — assert dtypes after extract.)
- A 0-row pull can lose column names/types (the connector returns no schema for an
  empty result) — avoid freezing empty snapshots.

## Example

```bash
python -m lib.sources pull \
  --experiment experiments/billing \
  --source redshift \
  --conn iam=true --conn profile=redshift-ro --conn cluster_identifier=prod \
  --conn database=dev --conn db_user=ro_user --conn region=eu-north-1 \
  --query "SELECT * FROM analytics.events WHERE event_ts < '2026-06-01' ORDER BY id"
```
