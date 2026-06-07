"""Query-level reproducibility helpers: as-of rewriting and a determinism guard.

These keep a warehouse pull reproducible — the loop's whole contract. As-of is
applied by substituting an ``{as_of}`` placeholder with the warehouse's
time-travel clause; the determinism guard rejects unseeded ``LIMIT``/sampling
and warns on moving-time predicates. Both are deliberately string-level (no SQL
parser dependency); the determinism check is a heuristic, documented as such.
"""

from __future__ import annotations

import re
import warnings

from lib.sources.errors import NonDeterministicQueryError

# Warehouses that support reading a historical version via a FROM-clause suffix.
_TIME_TRAVEL = {"snowflake", "bigquery", "databricks"}

_AS_OF_PLACEHOLDER = "{as_of}"

# Accepted --as-of forms. A Delta version must be the explicit ``v<n>`` form so a
# compact date like ``20260601`` can never be mistaken for a commit number.
_AS_OF_VERSION_RE = re.compile(r"^v\d+$", re.IGNORECASE)
_AS_OF_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?\s*(Z|UTC|[+-]\d{2}:?\d{2})?$"
)

_RANDOM = re.compile(r"\b(random|rand|newid|uuid_string)\s*\(", re.IGNORECASE)
# Row-cap clauses across dialects: LIMIT, FETCH FIRST n, TOP n.
_ROWCAP = re.compile(r"\blimit\b|\bfetch\s+first\b|\btop\s+\(?\d", re.IGNORECASE)
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_SAMPLE = re.compile(r"\b(tablesample|sample)\b", re.IGNORECASE)
_SEED = re.compile(r"\b(repeatable|seed)\b", re.IGNORECASE)
# Moving-time functions: deterministic within one pull, but a re-pull drifts.
_MOVING_TIME = re.compile(
    r"\b(now|getdate|sysdate)\s*\(|\b(current_date|current_timestamp|current_time)\b",
    re.IGNORECASE,
)


def _validate_as_of(as_of: str, source_type: str) -> None:
    """Reject an as-of that is neither a timestamp nor (Databricks) a ``v<n>``.

    Guards against garbage being string-interpolated into time-travel SQL and
    against ambiguous all-digit tokens.
    """
    value = as_of.strip()
    if _AS_OF_VERSION_RE.match(value):
        if source_type != "databricks":
            raise ValueError(
                f"as-of version form 'v<n>' is only valid for databricks, not {source_type}."
            )
        return
    if _AS_OF_TIMESTAMP_RE.match(value):
        return
    suffix = " or a Delta version 'v<n>'" if source_type == "databricks" else ""
    raise ValueError(
        f"Invalid --as-of {as_of!r}: expected an ISO timestamp "
        f"(e.g. 2026-06-01 or 2026-06-01T00:00:00Z){suffix}."
    )


def _time_travel_clause(source_type: str, as_of: str) -> str:
    """Return the dialect time-travel clause to substitute for ``{as_of}``."""
    if source_type == "snowflake":
        return f"AT(TIMESTAMP => '{as_of}'::timestamp_tz)"
    if source_type == "bigquery":
        return f"FOR SYSTEM_TIME AS OF TIMESTAMP '{as_of}'"
    if source_type == "databricks":
        version = _AS_OF_VERSION_RE.match(as_of.strip())
        if version:
            return f"VERSION AS OF {as_of.strip()[1:]}"
        return f"TIMESTAMP AS OF '{as_of}'"
    raise ValueError(f"{source_type} has no time travel")  # pragma: no cover


def apply_as_of(query: str, source_type: str, as_of: str | None) -> tuple[str, str | None]:
    """Apply an as-of point and return ``(rewritten_query, recorded_as_of)``.

    - Time-travel warehouses (Snowflake/BigQuery/Databricks): the query MUST
      contain an ``{as_of}`` placeholder right after the table reference; it is
      replaced with the dialect clause. Asking for an as-of without the
      placeholder is an error (we will not guess where the table is).
    - Other warehouses (Redshift/Postgres/DuckDB): there is no time travel, so
      the query must carry its own time predicate; the as-of value is recorded
      for provenance but the query is otherwise unchanged.
    - A placeholder with no as-of value is an error; a malformed as-of is an error.

    The recorded as-of (the caller's value) is what lands in the manifest so a
    re-pull reproduces the same rows.
    """
    has_placeholder = _AS_OF_PLACEHOLDER in query

    if as_of is None:
        if has_placeholder:
            raise ValueError(
                "Query contains an {as_of} placeholder but no --as-of value was given."
            )
        return query, None

    _validate_as_of(as_of, source_type)

    if source_type in _TIME_TRAVEL:
        if not has_placeholder:
            raise ValueError(
                f"{source_type} as-of requires an {{as_of}} placeholder after the table, "
                f"e.g. `FROM my_table {{as_of}}`."
            )
        rewritten = query.replace(_AS_OF_PLACEHOLDER, _time_travel_clause(source_type, as_of))
        return rewritten, as_of

    # Non-time-travel: the predicate lives in the query; just record the value.
    if has_placeholder:
        rewritten = query.replace(_AS_OF_PLACEHOLDER, "")
        return rewritten, as_of
    return query, as_of


def require_deterministic_query(query: str, *, allow_nondeterministic: bool = False) -> None:
    """Guard a pull query's reproducibility.

    Raises :class:`NonDeterministicQueryError` for hard non-determinism (random
    ordering/sampling, a row cap — ``LIMIT``/``FETCH FIRST``/``TOP`` — without an
    ``ORDER BY``, or unseeded sampling). Warns (without failing) on a moving-time
    predicate like ``now()``/``current_date``, which freezes fine in this pull
    but drifts on a re-pull — pin it with ``--as-of`` instead.

    Heuristic and string-level: it does not see ``LIMIT`` inside a CTE/subquery,
    ``QUALIFY``, or a moving predicate hidden in a view. ``allow_nondeterministic``
    skips the hard check for the rare intentional case.
    """
    if allow_nondeterministic:
        return
    reasons: list[str] = []
    if _RANDOM.search(query):
        reasons.append("uses a random() ordering/sampling function")
    if _SAMPLE.search(query) and not _SEED.search(query):
        reasons.append("samples without a REPEATABLE/seed clause")
    if _ROWCAP.search(query) and not _ORDER_BY.search(query):
        reasons.append("caps rows (LIMIT/FETCH FIRST/TOP) without ORDER BY")
    if reasons:
        raise NonDeterministicQueryError(
            "Query is not reproducible (" + "; ".join(reasons) + "). "
            "Add an ORDER BY / fixed seed, or pass --allow-nondeterministic to override."
        )
    if _MOVING_TIME.search(query):
        warnings.warn(
            "Query uses a moving-time function (now()/current_date/...): a re-pull will "
            "return different rows. Pin the point in time with --as-of (or a literal "
            "timestamp predicate) for a reproducible snapshot.",
            stacklevel=2,
        )
