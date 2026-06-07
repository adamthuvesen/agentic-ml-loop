# Google BigQuery source

Bulk extract via ADBC. The ADBC BigQuery driver is **experimental** (may need
`ADBC_BIGQUERY_LIBRARY` set to the native lib path); if the install is fussy in
your environment, the native `google-cloud-bigquery[bqstorage].to_arrow()` is a
drop-in fallback behind the same extractor contract.

## Install

```bash
uv sync --extra bigquery   # adbc-driver-bigquery + manager + pyarrow
```

## Auth (keyless local)

Application Default Credentials — no key file in the repo:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <billing-project>
```

For least privilege, add `--impersonate-service-account=<sa>` (still keyless).
Avoid committing service-account key files. Pass the billing/data project with
`--conn project_id=<proj>`.

## Read-only grant (the real guarantee)

`roles/bigquery.jobUser` (run queries) + `roles/bigquery.dataViewer` (read data),
scoped to the dataset/table. **Never** grant `dataEditor`/`dataOwner`. An IAM
deny policy on write permissions makes it a hard guarantee.

## As-of (reproducible)

Time travel via an `{as_of}` placeholder after the table:

```sql
SELECT * FROM `proj.dataset.events` {as_of} ORDER BY event_id
```

`--as-of '2026-06-01 00:00:00 UTC'` rewrites it to
`FOR SYSTEM_TIME AS OF TIMESTAMP '...'`. Window is 7 days (up to 14 on
Enterprise Plus); the parquet snapshot is the durable artifact beyond that.

## Caveats

- **Cost:** BigQuery bills by bytes scanned. Select explicit columns, filter
  partitions, and pass `--max-bytes` (best-effort: set as the
  `adbc.bigquery.sql.query.max_bytes_billed` statement option — verify it fires
  on your driver version). Use `--dry-run` to preview before a billable extract.
- `NUMERIC`/`BIGNUMERIC` -> Arrow decimals; `DATETIME` is tz-naive vs `TIMESTAMP`
  UTC — decide normalization explicitly.

## Example

```bash
python -m lib.sources pull \
  --experiment experiments/usage \
  --source bigquery --conn project_id=<proj> \
  --as-of '2026-06-01 00:00:00 UTC' --max-bytes 10000000000 \
  --query "SELECT user_id, plan, ts FROM \`proj.ds.events\` {as_of} ORDER BY user_id"
```
