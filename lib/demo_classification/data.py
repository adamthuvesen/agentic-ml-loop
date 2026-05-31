from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "will_expand_90d"
TIME_COLUMN = "snapshot_date"

BASE_NUMERIC_COLUMNS = [
    "account_age_days",
    "seat_count",
    "avg_weekly_active_users",
    "usage_growth_90d",
    "feature_adoption_count",
    "support_tickets_90d",
    "nps_score",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "seat_utilization",
    "support_per_active_user",
    "momentum_signal",
    "interaction_exec_growth",
]

CATEGORICAL_COLUMNS = ["region", "industry", "plan_tier"]
BOOLEAN_COLUMNS = ["has_exec_sponsor", "is_sales_touched", "previous_expansion"]

DATA_PATH = Path(__file__).parent / "data" / "demo_expansion.csv"


@dataclass(frozen=True)
class DemoSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_demo_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from lib.demo_classification.generate_data import generate_demo_dataset

        return generate_demo_dataset()
    return pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])


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


def base_feature_columns() -> list[str]:
    return BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS


def engineered_feature_columns() -> list[str]:
    return base_feature_columns() + ENGINEERED_NUMERIC_COLUMNS
