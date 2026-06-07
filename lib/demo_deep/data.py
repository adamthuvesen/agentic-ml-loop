from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "will_convert"
TIME_COLUMN = "event_date"

NUMERIC_COLUMNS = ["signal_a", "signal_b", "signal_c", "signal_d"]
CATEGORICAL_COLUMNS = ["channel"]

DATA_PATH = Path(__file__).parent / "data" / "demo_nonlinear.csv"


@dataclass(frozen=True)
class DemoSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def feature_columns() -> list[str]:
    return NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def load_demo_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from lib.demo_deep.generate_data import generate_demo_dataset

        return generate_demo_dataset()
    return pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN])


def split_demo_dataset(df: pd.DataFrame) -> DemoSplits:
    if len(df) < 10:
        raise ValueError(f"Dataset too small to split: {len(df)} rows (minimum 10).")
    ordered = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    train_cutoff = int(len(ordered) * 0.60)
    validation_cutoff = int(len(ordered) * 0.80)
    return DemoSplits(
        train=ordered.iloc[:train_cutoff].reset_index(drop=True),
        validation=ordered.iloc[train_cutoff:validation_cutoff].reset_index(drop=True),
        test=ordered.iloc[validation_cutoff:].reset_index(drop=True),
    )
