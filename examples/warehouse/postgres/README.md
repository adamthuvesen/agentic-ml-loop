# Postgres warehouse-ingestion demo (creds-free)

A local Postgres with `trust` auth — no password anywhere — that proves the real
"talk to a server, read-only, freeze a snapshot" path end to end. It is the
tier-1 demo: a real database server, still with zero committed secrets.

## Run

```bash
docker compose -f examples/warehouse/postgres/docker-compose.yml up -d

uv sync --extra postgres
python -m lib.sources pull \
  --experiment experiments/<your_experiment> \
  --source postgres \
  --uri "postgresql://demo_ro@localhost:5432/demo" \
  --query "SELECT * FROM events ORDER BY id"

docker compose -f examples/warehouse/postgres/docker-compose.yml down
```

This writes `experiments/<id>/data/snapshot.parquet` and a
`dataset_manifest.json` (source, query, as-of, row count, schema hash). The
`demo_ro` role is `SELECT`-only, so the pull cannot write regardless of the
query. Nothing secret is stored anywhere.

## Reading the snapshot

The experiment's `lib/<slug>/data.py` loads the frozen snapshot — no warehouse,
fully offline — and verifies it against the manifest:

```python
from pathlib import Path

from lib.sources import read_snapshot

EXPERIMENT_DIR = Path("experiments/<your_experiment>")

def load_dataset():
    # Verifies snapshot.parquet against dataset_manifest.json, returns a DataFrame.
    return read_snapshot(EXPERIMENT_DIR)
```

Everything downstream (splits, runners, the leaderboard) is unchanged — the only
difference from a CSV experiment is where `load_dataset` reads from.
