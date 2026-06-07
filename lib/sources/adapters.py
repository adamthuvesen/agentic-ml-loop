"""Per-warehouse extractor adapters.

Every adapter exposes the same shape — ``source_type``, ``driver``, and
``extract(query) -> pyarrow.Table`` — so the rest of the package never branches
on warehouse. Warehouse client libraries are imported lazily inside ``extract``
so that ``import lib.sources`` stays cheap and works without any extra installed;
a missing client raises :class:`MissingSourceExtra` naming the extra to install.

Backend split (see the change design): ADBC for Postgres / Snowflake / BigQuery
(clean wheels, best Arrow type fidelity), native clients for Databricks and
Redshift (their ADBC drivers are not pip-clean yet).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from lib.sources.errors import MissingSourceExtra

if TYPE_CHECKING:
    import pyarrow as pa

# source_type -> (extra name in pyproject, distribution that provides the client)
SOURCE_EXTRAS: dict[str, tuple[str, str]] = {
    "postgres": ("postgres", "adbc-driver-postgresql"),
    "snowflake": ("snowflake", "adbc-driver-snowflake"),
    "bigquery": ("bigquery", "adbc-driver-bigquery"),
    "databricks": ("databricks", "databricks-sql-connector"),
    "redshift": ("redshift", "redshift-connector"),
    "duckdb": ("models", "duckdb"),
}

# ADBC-backed source types -> their per-driver DBAPI module.
ADBC_DBAPI: dict[str, str] = {
    "postgres": "adbc_driver_postgresql.dbapi",
    "snowflake": "adbc_driver_snowflake.dbapi",
    "bigquery": "adbc_driver_bigquery.dbapi",
}

# Friendly --conn keys -> ADBC option strings, per driver. ADBC expects its own
# option namespace (``adbc.<driver>.sql.*``) plus the standard
# ``username``/``password``; we translate the common friendly keys so the
# documented ``--conn account=... user=...`` UX maps to valid options. Unknown
# or already-namespaced keys pass through unchanged. Strings are per the ADBC
# driver docs; the live connection itself is not exercised by this repo's tests.
_ADBC_KEY_MAP: dict[str, dict[str, str]] = {
    "snowflake": {
        "account": "adbc.snowflake.sql.account",
        "user": "username",
        "username": "username",
        "password": "password",
        "database": "adbc.snowflake.sql.db",
        "schema": "adbc.snowflake.sql.schema",
        "warehouse": "adbc.snowflake.sql.warehouse",
        "role": "adbc.snowflake.sql.role",
        "authenticator": "adbc.snowflake.sql.auth_type",
    },
    "bigquery": {
        "project": "adbc.bigquery.sql.project_id",
        "project_id": "adbc.bigquery.sql.project_id",
        "dataset": "adbc.bigquery.sql.dataset_id",
        "dataset_id": "adbc.bigquery.sql.dataset_id",
    },
}

# Friendly values -> ADBC values, keyed by (source_type, friendly_key).
_ADBC_VALUE_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("snowflake", "authenticator"): {
        "externalbrowser": "auth_ext_browser",
        "snowflake": "auth_snowflake",
        "oauth": "auth_oauth",
    },
}

SUPPORTED_SOURCE_TYPES: list[str] = sorted(SOURCE_EXTRAS)


def _require(source_type: str, module: str, package: str | None = None):
    """Import *module* or raise an actionable :class:`MissingSourceExtra`."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        extra, default_pkg = SOURCE_EXTRAS[source_type]
        raise MissingSourceExtra(source_type, extra, package or default_pkg) from exc


def _to_adbc_db_kwargs(source_type: str, config: dict[str, Any]) -> dict[str, str]:
    """Translate friendly connection keys/values into ADBC's option namespace.

    Unknown or already-namespaced keys pass through unchanged. All values are
    coerced to ``str`` because ADBC ``db_kwargs`` is a string-to-string mapping.
    """
    key_map = _ADBC_KEY_MAP.get(source_type, {})
    result: dict[str, str] = {}
    for key, value in config.items():
        adbc_key = key_map.get(key, key)
        value_map = _ADBC_VALUE_MAP.get((source_type, key))
        adbc_value = value_map.get(str(value), value) if value_map else value
        result[adbc_key] = str(adbc_value)
    return result


class AdbcExtractor:
    """Read-only extractor for Postgres, Snowflake, and BigQuery via ADBC.

    Read-only is enforced server-side by the connection's least-privilege role
    (documented per source in its ``SETUP.md``); ADBC exposes no portable
    read-only session flag, so the role is the real guarantee.
    """

    driver = "adbc"

    def __init__(self, source_type: str, config: dict[str, Any]) -> None:
        if source_type not in ADBC_DBAPI:
            raise ValueError(f"ADBC does not back source type {source_type!r}")
        self.source_type = source_type
        config = dict(config)
        # Control keys that are not connection kwargs:
        self._max_bytes_billed = config.pop("max_bytes_billed", None)
        self._config = config

    def extract(self, query: str) -> pa.Table:
        impl = _require(self.source_type, ADBC_DBAPI[self.source_type])
        if self.source_type == "postgres":
            conn = impl.connect(self._config["uri"])
        else:
            db_kwargs = _to_adbc_db_kwargs(
                self.source_type,
                {key: value for key, value in self._config.items() if key != "uri"},
            )
            conn = impl.connect(db_kwargs=db_kwargs)
        try:
            with conn.cursor() as cur:
                self._apply_statement_options(cur)
                cur.execute(query)
                return cur.fetch_arrow_table()
        finally:
            conn.close()

    def _apply_statement_options(self, cursor: Any) -> None:
        """Set per-statement ADBC options before execute (best-effort).

        BigQuery's bytes-billed ceiling is a *statement* option, not a connection
        kwarg, so it is set on the cursor here. Not live-verified in this repo.
        """
        if self.source_type == "bigquery" and self._max_bytes_billed is not None:
            cursor.adbc_statement.set_options(
                **{"adbc.bigquery.sql.query.max_bytes_billed": str(self._max_bytes_billed)}
            )


class DatabricksExtractor:
    """Read-only extractor for Databricks SQL warehouses (native client)."""

    source_type = "databricks"
    driver = "databricks-sql-connector"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def extract(self, query: str) -> pa.Table:
        module = _require("databricks", "databricks.sql", "databricks-sql-connector")
        with module.connect(**self._config) as conn, conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall_arrow()


class RedshiftExtractor:
    """Read-only extractor for Amazon Redshift (native client).

    Redshift has no Arrow-native fetch, so results come back through a pandas
    DataFrame; see the source ``SETUP.md`` for the DECIMAL/timestamp caveats.
    """

    source_type = "redshift"
    driver = "redshift-connector"

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = dict(config)

    def extract(self, query: str) -> pa.Table:
        redshift_connector = _require("redshift", "redshift_connector")
        pa = _require("redshift", "pyarrow", "pyarrow")
        conn = redshift_connector.connect(**self._config)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            frame = cursor.fetch_dataframe()
            return pa.Table.from_pandas(frame, preserve_index=False)
        finally:
            conn.close()


class DuckDbExtractor:
    """Read-only extractor over a local DuckDB database or file (tier-0, creds-free)."""

    source_type = "duckdb"
    driver = "duckdb"

    def __init__(self, config: dict[str, Any]) -> None:
        self._database = config.get("database", ":memory:")
        # A :memory: database cannot be opened read-only; a real file can.
        self._read_only = bool(config.get("read_only", self._database != ":memory:"))

    def extract(self, query: str) -> pa.Table:
        duckdb = _require("duckdb", "duckdb")
        conn = duckdb.connect(self._database, read_only=self._read_only)
        try:
            return conn.execute(query).to_arrow_table()
        finally:
            conn.close()
