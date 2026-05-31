from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "converted"
TIME_COLUMN = "event_date"

NUMERIC_COLUMNS = ["sessions_7d", "pages_viewed", "days_active"]
CATEGORICAL_COLUMNS = ["segment"]

EXCLUDE_COLUMNS = frozenset({"user_id", TARGET_COLUMN, TIME_COLUMN})

DATA_PATH = Path(__file__).parent / "data" / "demo_events.csv"


@dataclass(frozen=True)
class BootstrapSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def feature_columns() -> list[str]:
    return list(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from lib.demo_bootstrap.generate_data import generate_demo_bootstrap_dataset

        return generate_demo_bootstrap_dataset()
    df = pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN])
    missing = {TARGET_COLUMN, TIME_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def split_dataset(df: pd.DataFrame) -> BootstrapSplits:
    if len(df) < 10:
        raise ValueError(f"Dataset too small to split: {len(df)} rows (minimum 10).")
    ordered = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    n = len(ordered)
    train_cutoff = int(n * 0.60)
    val_cutoff = int(n * 0.80)
    return BootstrapSplits(
        train=ordered.iloc[:train_cutoff].reset_index(drop=True),
        validation=ordered.iloc[train_cutoff:val_cutoff].reset_index(drop=True),
        test=ordered.iloc[val_cutoff:].reset_index(drop=True),
    )
