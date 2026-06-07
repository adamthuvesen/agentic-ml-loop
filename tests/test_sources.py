"""Tests for the warehouse ingestion layer (lib.sources).

Structural and offline by design: the creds-free DuckDB path is exercised for
real; the proprietary adapters are checked via fakes. No live warehouse, no
credentials. Mirrors the change's scope cut.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest

from lib.sources import (
    DatasetManifest,
    MissingSourceExtra,
    NonDeterministicQueryError,
    SnapshotIntegrityError,
    apply_as_of,
    dataset_manifest_from_table,
    freeze_snapshot,
    get_extractor,
    list_bundles,
    read_snapshot,
    require_deterministic_query,
    verify_snapshot,
)
from lib.sources.adapters import DatabricksExtractor, RedshiftExtractor, _to_adbc_db_kwargs

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

_HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None
_HAS_ADBC = importlib.util.find_spec("adbc_driver_manager") is not None


# --------------------------------------------------------------------------- #
# Determinism guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM t LIMIT 10",
        "SELECT * FROM t TABLESAMPLE SYSTEM (10)",
        "SELECT * FROM t ORDER BY random() LIMIT 5",
    ],
)
def test_nondeterministic_query_rejected(query: str) -> None:
    with pytest.raises(NonDeterministicQueryError):
        require_deterministic_query(query)


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM t ORDER BY id LIMIT 10",
        "SELECT * FROM t",
        "SELECT * FROM t TABLESAMPLE SYSTEM (10) REPEATABLE (42)",
    ],
)
def test_deterministic_query_accepted(query: str) -> None:
    require_deterministic_query(query)  # must not raise


def test_determinism_escape_hatch() -> None:
    require_deterministic_query("SELECT * FROM t LIMIT 10", allow_nondeterministic=True)


# --------------------------------------------------------------------------- #
# As-of rewriting
# --------------------------------------------------------------------------- #
def test_as_of_time_travel_rewrites_placeholder() -> None:
    rewritten, recorded = apply_as_of(
        "SELECT * FROM t {as_of} ORDER BY id", "snowflake", "2026-06-01 00:00:00"
    )
    assert "AT(TIMESTAMP =>" in rewritten
    assert "{as_of}" not in rewritten
    assert recorded == "2026-06-01 00:00:00"


def test_as_of_databricks_version_vs_timestamp() -> None:
    by_version, _ = apply_as_of("SELECT * FROM t {as_of}", "databricks", "v123")
    assert "VERSION AS OF 123" in by_version
    by_ts, _ = apply_as_of("SELECT * FROM t {as_of}", "databricks", "2026-06-01T00:00:00Z")
    assert "TIMESTAMP AS OF" in by_ts


def test_as_of_time_travel_requires_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        apply_as_of("SELECT * FROM t", "snowflake", "2026-06-01")


def test_as_of_placeholder_requires_value() -> None:
    with pytest.raises(ValueError, match="no --as-of"):
        apply_as_of("SELECT * FROM t {as_of}", "snowflake", None)


def test_as_of_non_time_travel_records_value() -> None:
    rewritten, recorded = apply_as_of(
        "SELECT * FROM t WHERE ts < '2026-06-01'", "postgres", "2026-06-01"
    )
    assert recorded == "2026-06-01"
    assert "{as_of}" not in rewritten


# --------------------------------------------------------------------------- #
# Factory + missing-extra errors
# --------------------------------------------------------------------------- #
def test_unknown_source_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown source type"):
        get_extractor("oracle")


@pytest.mark.skipif(_HAS_ADBC, reason="adbc installed; missing-extra path not exercised")
def test_missing_extra_is_actionable() -> None:
    extractor = get_extractor("snowflake")
    with pytest.raises(MissingSourceExtra) as excinfo:
        extractor.extract("SELECT 1")
    message = str(excinfo.value)
    assert "snowflake" in message
    assert "extra" in message


# --------------------------------------------------------------------------- #
# Manifest model
# --------------------------------------------------------------------------- #
def _sample_table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "label": ["a", "b", "c"]})


def test_manifest_roundtrip(tmp_path) -> None:
    manifest = dataset_manifest_from_table(
        _sample_table(),
        source_type="duckdb",
        driver="duckdb",
        query="SELECT * FROM t",
        as_of=None,
        sample_seed=None,
    )
    path = tmp_path / "dataset_manifest.json"
    manifest.write(path)
    loaded = DatasetManifest.load(path)
    assert loaded == manifest
    assert loaded.row_count == 3
    assert loaded.columns == ["id", "label"]


def test_manifest_load_rejects_unknown_keys(tmp_path) -> None:
    manifest = dataset_manifest_from_table(
        _sample_table(),
        source_type="duckdb",
        driver="duckdb",
        query="SELECT * FROM t",
        as_of=None,
        sample_seed=None,
    )
    payload = manifest.to_dict()
    payload["bogus"] = 1
    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown manifest keys"):
        DatasetManifest.load(path)


# --------------------------------------------------------------------------- #
# Snapshot integrity (verify_snapshot)
# --------------------------------------------------------------------------- #
def test_verify_snapshot_detects_row_and_schema_drift(tmp_path) -> None:
    table = _sample_table()
    snapshot = tmp_path / "snapshot.parquet"
    pq.write_table(table, snapshot)
    manifest = dataset_manifest_from_table(
        table,
        source_type="duckdb",
        driver="duckdb",
        query="SELECT * FROM t",
        as_of=None,
        sample_seed=None,
    )
    verify_snapshot(snapshot, manifest)  # matches: no raise

    pq.write_table(pa.table({"id": [1], "label": ["a"]}), snapshot)  # fewer rows
    with pytest.raises(SnapshotIntegrityError, match="row count"):
        verify_snapshot(snapshot, manifest)

    pq.write_table(pa.table({"id": [1, 2, 3]}), snapshot)  # different schema
    with pytest.raises(SnapshotIntegrityError, match="schema hash"):
        verify_snapshot(snapshot, manifest)


# --------------------------------------------------------------------------- #
# DuckDB golden round-trip (creds-free, real extraction)
# --------------------------------------------------------------------------- #
def _make_duckdb(tmp_path):
    import duckdb

    db_path = tmp_path / "demo.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE t (id INTEGER, amount DECIMAL(18,4), ts TIMESTAMPTZ, label VARCHAR)")
    con.execute(
        "INSERT INTO t VALUES "
        "(1, 12.3400, TIMESTAMPTZ '2026-01-01 00:00:00+00', 'a'), "
        "(2, 99.9900, TIMESTAMPTZ '2026-02-01 12:00:00+00', 'b')"
    )
    con.close()
    return db_path


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed (needs the models extra)")
def test_duckdb_freeze_preserves_decimal_and_timestamp_types(tmp_path) -> None:
    db_path = _make_duckdb(tmp_path)
    out_dir = tmp_path / "data"
    manifest = freeze_snapshot(
        source_type="duckdb",
        query="SELECT * FROM t ORDER BY id",
        out_dir=out_dir,
        config={"database": str(db_path), "read_only": True},
    )
    assert manifest.row_count == 2
    assert manifest.source_type == "duckdb"

    schema = pq.read_schema(str(out_dir / "snapshot.parquet"))
    assert pa.types.is_decimal(schema.field("amount").type)
    ts_type = schema.field("ts").type
    assert pa.types.is_timestamp(ts_type)
    assert ts_type.tz is not None  # timezone survived to parquet


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed (needs the models extra)")
def test_read_snapshot_roundtrips_and_verifies(tmp_path) -> None:
    db_path = _make_duckdb(tmp_path)
    experiment_dir = tmp_path / "exp"
    freeze_snapshot(
        source_type="duckdb",
        query="SELECT * FROM t ORDER BY id",
        out_dir=experiment_dir / "data",
        config={"database": str(db_path), "read_only": True},
    )
    frame = read_snapshot(experiment_dir)
    assert list(frame["id"]) == [1, 2]

    # Mutating the parquet must make read_snapshot fail the integrity check.
    pq.write_table(pa.table({"id": [1]}), experiment_dir / "data" / "snapshot.parquet")
    with pytest.raises(SnapshotIntegrityError):
        read_snapshot(experiment_dir)


@pytest.mark.skipif(not _HAS_DUCKDB, reason="duckdb not installed (needs the models extra)")
def test_experiment_validate_flags_mutated_snapshot(tmp_path) -> None:
    import experiment

    db_path = _make_duckdb(tmp_path)
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "experiment.md").write_text("# x\n", encoding="utf-8")
    (exp / "research_journal.md").write_text("# journal\n", encoding="utf-8")
    (exp / "results.json").write_text("[]\n", encoding="utf-8")
    freeze_snapshot(
        source_type="duckdb",
        query="SELECT * FROM t ORDER BY id",
        out_dir=exp / "data",
        config={"database": str(db_path), "read_only": True},
    )
    assert experiment.validate_experiment(exp) == []

    pq.write_table(pa.table({"id": [1]}), exp / "data" / "snapshot.parquet")
    errors = experiment.validate_experiment(exp)
    assert any("snapshot integrity" in error for error in errors)


# --------------------------------------------------------------------------- #
# Proprietary adapters via fakes (wiring only, no live connection)
# --------------------------------------------------------------------------- #
def test_databricks_adapter_wiring(monkeypatch) -> None:
    table = _sample_table()
    captured: dict[str, str] = {}

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query):
            captured["query"] = query

        def fetchall_arrow(self):
            return table

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self):
            return _Cursor()

    fake_sql = types.ModuleType("databricks.sql")
    fake_sql.connect = lambda **kwargs: _Conn()
    monkeypatch.setitem(sys.modules, "databricks", types.ModuleType("databricks"))
    monkeypatch.setitem(sys.modules, "databricks.sql", fake_sql)

    result = DatabricksExtractor({"server_hostname": "h"}).extract("SELECT 1")
    assert result.num_rows == 3
    assert captured["query"] == "SELECT 1"


def test_redshift_adapter_wiring(monkeypatch) -> None:
    import pandas as pd

    class _Cursor:
        def execute(self, query):
            self.query = query

        def fetch_dataframe(self):
            return pd.DataFrame({"id": [1, 2]})

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    fake = types.ModuleType("redshift_connector")
    fake.connect = lambda **kwargs: _Conn()
    monkeypatch.setitem(sys.modules, "redshift_connector", fake)

    result = RedshiftExtractor({"host": "h"}).extract("SELECT 1")
    assert result.num_rows == 2


# --------------------------------------------------------------------------- #
# Bundles
# --------------------------------------------------------------------------- #
def test_all_warehouses_have_bundles() -> None:
    types_present = {bundle.source_type for bundle in list_bundles()}
    assert {
        "snowflake",
        "bigquery",
        "redshift",
        "databricks",
        "postgres",
        "duckdb",
    } <= types_present


def test_bundles_carry_no_inline_credentials() -> None:
    import re

    deny_keys = {"password", "secret", "token", "private_key", "access_key", "api_key"}
    secret_pattern = re.compile(r"(?i)(password|secret)\s*[:=]\s*\S|://[^/\s:]+:[^/@\s]+@")
    for bundle in list_bundles():
        data = json.loads((bundle.directory / "source.json").read_text(encoding="utf-8"))
        assert not (set(map(str.lower, data)) & deny_keys), bundle.source_type
        for value in data.values():
            for text in value if isinstance(value, list) else [value]:
                assert not secret_pattern.search(str(text)), f"{bundle.source_type}: {text}"


# --------------------------------------------------------------------------- #
# Offline import guard
# --------------------------------------------------------------------------- #
def test_importing_sources_does_not_import_warehouse_clients(monkeypatch) -> None:
    for name in list(sys.modules):
        if name.startswith("lib.sources"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    import lib.sources  # noqa: F401

    for client in (
        "adbc_driver_manager",
        "adbc_driver_postgresql",
        "databricks.sql",
        "redshift_connector",
        "snowflake",
    ):
        assert client not in sys.modules, f"{client} imported at package import time"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_pull_reports_clean_error_on_nondeterministic_query(tmp_path) -> None:
    from lib.sources.cli import main

    exit_code = main(
        [
            "pull",
            "--experiment",
            str(tmp_path),
            "--source",
            "duckdb",
            "--database",
            "unused.duckdb",
            "--query",
            "SELECT * FROM t LIMIT 1",
        ]
    )
    assert exit_code == 2  # clean non-zero exit, not a traceback


# --------------------------------------------------------------------------- #
# As-of validation
# --------------------------------------------------------------------------- #
def test_as_of_rejects_injection_garbage() -> None:
    with pytest.raises(ValueError, match="Invalid --as-of"):
        apply_as_of("SELECT * FROM t {as_of}", "snowflake", "2026' OR '1'='1")


def test_as_of_rejects_ambiguous_compact_date() -> None:
    # 20260601 is neither an ISO timestamp nor a 'v<n>' version.
    with pytest.raises(ValueError, match="Invalid --as-of"):
        apply_as_of("SELECT * FROM t {as_of}", "databricks", "20260601")


def test_as_of_version_form_is_databricks_only() -> None:
    with pytest.raises(ValueError, match="only valid for databricks"):
        apply_as_of("SELECT * FROM t {as_of}", "snowflake", "v5")


# --------------------------------------------------------------------------- #
# Determinism: extra row-cap clauses + moving-time warning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM t FETCH FIRST 10 ROWS ONLY",
        "SELECT TOP 10 * FROM t",
    ],
)
def test_rowcap_clauses_rejected(query: str) -> None:
    with pytest.raises(NonDeterministicQueryError):
        require_deterministic_query(query)


def test_identifiers_named_like_clauses_are_accepted() -> None:
    # Columns containing 'limit'/'sample' must not trip the guard.
    require_deterministic_query("SELECT credit_limit, sample_id FROM t ORDER BY id")


def test_moving_time_predicate_warns_without_raising() -> None:
    with pytest.warns(UserWarning, match="moving-time"):
        require_deterministic_query("SELECT * FROM t WHERE created_at > current_date - 7")


# --------------------------------------------------------------------------- #
# ADBC friendly-key -> option-namespace mapping
# --------------------------------------------------------------------------- #
def test_adbc_key_mapping_snowflake() -> None:
    out = _to_adbc_db_kwargs(
        "snowflake",
        {
            "account": "a",
            "user": "u",
            "authenticator": "externalbrowser",
            "role": "R",
            "warehouse": "W",
            "database": "D",
            "schema": "S",
        },
    )
    assert out["adbc.snowflake.sql.account"] == "a"
    assert out["username"] == "u"
    assert out["adbc.snowflake.sql.auth_type"] == "auth_ext_browser"
    assert out["adbc.snowflake.sql.role"] == "R"
    assert out["adbc.snowflake.sql.warehouse"] == "W"
    assert out["adbc.snowflake.sql.db"] == "D"
    assert out["adbc.snowflake.sql.schema"] == "S"


def test_adbc_key_mapping_bigquery_and_passthrough() -> None:
    out = _to_adbc_db_kwargs(
        "bigquery",
        {"project_id": "p", "adbc.bigquery.sql.dataset_id": "d"},
    )
    assert out["adbc.bigquery.sql.project_id"] == "p"
    # Already-namespaced keys pass through unchanged.
    assert out["adbc.bigquery.sql.dataset_id"] == "d"
