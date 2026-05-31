from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TARGET_COLUMN = "net_revenue_change_180d"
TIME_COLUMN = "snapshot_date"

BASE_NUMERIC_COLUMNS = [
    "account_age_days",
    "seat_count",
    "avg_weekly_active_users",
    "usage_growth_90d",
    "feature_adoption_count",
    "champion_engagement_score",
    "support_tickets_90d",
    "nps_score",
    "contract_value_t0",
    "renewal_window_days",
    "previous_revenue_change_365d",
    "marketing_touchpoints_90d",
    "webinar_attendance_90d",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "seat_utilization",
    "support_burden",
    "growth_efficiency",
    "champion_x_growth",
    "expansion_capacity",
]

CATEGORICAL_COLUMNS = ["region", "industry", "plan_tier", "segment"]
BOOLEAN_COLUMNS = ["has_exec_sponsor", "multi_product", "previous_expansion"]

DATA_PATH = Path(__file__).parent / "data" / "demo_revenue_change.csv"


@dataclass(frozen=True)
class DemoRegressionSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_demo_regression_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from lib.demo_regression.generate_data import generate_demo_regression_dataset

        return generate_demo_regression_dataset()
    return pd.read_csv(DATA_PATH, parse_dates=[TIME_COLUMN])


def split_demo_regression_dataset(df: pd.DataFrame) -> DemoRegressionSplits:
    if len(df) < 10:
        raise ValueError(f"Dataset too small to split: {len(df)} rows (minimum 10).")
    ordered = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    train_cutoff = int(len(ordered) * 0.60)
    validation_cutoff = int(len(ordered) * 0.80)
    return DemoRegressionSplits(
        train=ordered.iloc[:train_cutoff].reset_index(drop=True),
        validation=ordered.iloc[train_cutoff:validation_cutoff].reset_index(drop=True),
        test=ordered.iloc[validation_cutoff:].reset_index(drop=True),
    )


def base_feature_columns() -> list[str]:
    return BASE_NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS


def engineered_feature_columns() -> list[str]:
    return base_feature_columns() + ENGINEERED_NUMERIC_COLUMNS
