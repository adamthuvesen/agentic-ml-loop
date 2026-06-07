"""Command line for warehouse ingestion: ``pull`` and ``sources list/enable``.

Run as ``python -m lib.sources <command>``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.paths import DATA_DIRNAME
from lib.sources.adapters import SUPPORTED_SOURCE_TYPES
from lib.sources.errors import SourceError
from lib.sources.query import apply_as_of, ensure_deterministic
from lib.sources.registry import enable_source, get_bundle, is_enabled, list_bundles
from lib.sources.snapshot import MANIFEST_FILENAME, SNAPSHOT_FILENAME, freeze_snapshot

ROOT = Path(__file__).resolve().parents[2]


def _resolve_experiment(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.exists() else (ROOT / value).resolve()


def _read_query(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _build_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {}
    if args.uri:
        config["uri"] = args.uri
    if args.database:
        config["database"] = args.database
    if args.max_bytes is not None:
        config["max_bytes_billed"] = args.max_bytes
    for pair in args.conn or []:
        if "=" not in pair:
            raise SystemExit(f"--conn expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        config[key] = value
    return config


def _cmd_pull(args: argparse.Namespace) -> int:
    experiment_dir = _resolve_experiment(args.experiment)
    if not experiment_dir.is_dir():
        raise SystemExit(f"Experiment directory does not exist: {experiment_dir}")
    query = _read_query(args.query)
    data_dir = experiment_dir / DATA_DIRNAME

    if args.dry_run:
        ensure_deterministic(query, allow_nondeterministic=args.allow_nondeterministic)
        rewritten, recorded = apply_as_of(query, args.source, args.as_of)
        print(f"[dry-run] source={args.source} experiment={experiment_dir.name} as_of={recorded}")
        print(f"[dry-run] would write {data_dir / SNAPSHOT_FILENAME}")
        print(f"[dry-run] query:\n{rewritten}")
        return 0

    manifest = freeze_snapshot(
        source_type=args.source,
        query=query,
        out_dir=data_dir,
        config=_build_config(args),
        as_of=args.as_of,
        sample_seed=args.sample_seed,
        allow_nondeterministic=args.allow_nondeterministic,
    )
    print(
        f"Wrote {manifest.row_count} rows x {manifest.column_count} cols "
        f"-> {data_dir / manifest.snapshot_filename}"
    )
    print(f"Manifest: {data_dir / MANIFEST_FILENAME}")
    return 0


def _cmd_sources_list(_: argparse.Namespace) -> int:
    bundles = list_bundles()
    if not bundles:
        print("No source bundles found.")
        return 0
    for bundle in bundles:
        mark = "*" if is_enabled(bundle.source_type) else " "
        print(f"[{mark}] {bundle.source_type:<10} {bundle.display_name} — {bundle.description}")
    print("\n* = enabled.  Enable with: python -m lib.sources sources enable <name>")
    return 0


def _cmd_sources_enable(args: argparse.Namespace) -> int:
    try:
        bundle = get_bundle(args.name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    enable_source(args.name)
    print(f"Enabled '{args.name}'. Follow setup steps in: {bundle.setup_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lib.sources",
        description="Pull a read-only warehouse query into a frozen experiment snapshot.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pull = sub.add_parser("pull", help="Materialize a snapshot into an experiment")
    pull.add_argument("--experiment", required=True, help="Experiment directory or relative path")
    pull.add_argument("--source", required=True, choices=SUPPORTED_SOURCE_TYPES)
    pull.add_argument("--query", required=True, help="SQL string, or @path to a .sql file")
    pull.add_argument("--as-of", default=None, help="As-of timestamp/version (see source SETUP.md)")
    pull.add_argument("--sample-seed", type=int, default=None, help="Seed used for sampling")
    pull.add_argument("--uri", default=None, help="Connection URI (e.g. postgres)")
    pull.add_argument("--database", default=None, help="Database path (e.g. duckdb file)")
    pull.add_argument(
        "--conn",
        action="append",
        metavar="KEY=VALUE",
        help="Connection kwarg; repeatable",
    )
    pull.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Scan ceiling in bytes (BigQuery max_bytes_billed statement option; best-effort)",
    )
    pull.add_argument("--dry-run", action="store_true", help="Show the plan without extracting")
    pull.add_argument(
        "--allow-nondeterministic",
        action="store_true",
        help="Override the reproducibility guard (unseeded LIMIT/sampling)",
    )
    pull.set_defaults(func=_cmd_pull)

    sources = sub.add_parser("sources", help="List or enable source bundles")
    sources_sub = sources.add_subparsers(dest="sources_command", required=True)
    list_parser = sources_sub.add_parser("list", help="List available source bundles")
    list_parser.set_defaults(func=_cmd_sources_list)
    enable_parser = sources_sub.add_parser("enable", help="Enable a source bundle")
    enable_parser.add_argument("name", choices=SUPPORTED_SOURCE_TYPES)
    enable_parser.set_defaults(func=_cmd_sources_enable)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (SourceError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
