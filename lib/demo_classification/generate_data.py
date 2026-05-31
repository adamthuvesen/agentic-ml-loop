"""Generate the demo expansion dataset.

This is the "answer key" — it contains the exact DGP (data generating process)
used to produce the target variable. It exists for reproducibility but should
NOT be imported by anything the agent reads during a run.

Usage:
    python lib/demo_classification/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib.demo_classification.data import TARGET_COLUMN, TIME_COLUMN

SEED = 42


def generate_demo_dataset(n_rows: int = 3600, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-12-31", periods=n_rows)

    df = pd.DataFrame(
        {
            TIME_COLUMN: dates,
            "account_age_days": rng.integers(14, 1800, size=n_rows),
            "seat_count": rng.integers(1, 250, size=n_rows),
            "avg_weekly_active_users": rng.normal(18, 10, size=n_rows).clip(0, None),
            "usage_growth_90d": rng.normal(0.05, 0.24, size=n_rows),
            "feature_adoption_count": rng.integers(0, 8, size=n_rows),
            "support_tickets_90d": rng.poisson(1.6, size=n_rows),
            "nps_score": rng.normal(32, 24, size=n_rows).clip(-100, 100),
            "region": rng.choice(
                ["north_america", "emea", "apac", "latam"],
                size=n_rows,
                p=[0.34, 0.38, 0.18, 0.10],
            ),
            "industry": rng.choice(
                ["software", "education", "consulting", "consumer", "other"],
                size=n_rows,
                p=[0.28, 0.16, 0.18, 0.18, 0.20],
            ),
            "plan_tier": rng.choice(
                ["free", "basic", "pro", "business"],
                size=n_rows,
                p=[0.32, 0.24, 0.30, 0.14],
            ),
            "has_exec_sponsor": rng.binomial(1, 0.18, size=n_rows).astype(bool),
            "is_sales_touched": rng.binomial(1, 0.26, size=n_rows).astype(bool),
            "previous_expansion": rng.binomial(1, 0.14, size=n_rows).astype(bool),
        }
    )

    df["seat_utilization"] = (df["avg_weekly_active_users"] / np.maximum(df["seat_count"], 1)).clip(
        0, 2.0
    )
    df["support_per_active_user"] = (
        df["support_tickets_90d"] / np.maximum(df["avg_weekly_active_users"], 1)
    ).clip(0, 2.0)
    df["momentum_signal"] = (
        0.65 * df["usage_growth_90d"]
        + 0.08 * df["feature_adoption_count"]
        + 0.004 * df["nps_score"]
        + 0.12 * df["seat_utilization"]
        + 0.25 * df["has_exec_sponsor"].astype(int)
        + 0.18 * df["previous_expansion"].astype(int)
        + 0.10 * (df["plan_tier"].isin(["pro", "business"])).astype(int)
        - 0.22 * df["support_per_active_user"]
    )
    df["interaction_exec_growth"] = df["has_exec_sponsor"].astype(int) * df["usage_growth_90d"]

    logits = (
        -1.45
        + 2.8 * df["usage_growth_90d"]
        + 0.18 * df["feature_adoption_count"]
        + 0.006 * df["nps_score"]
        + 0.30 * df["has_exec_sponsor"].astype(int)
        + 0.24 * df["previous_expansion"].astype(int)
        + 0.0009 * df["account_age_days"]
        + 0.14 * (df["plan_tier"] == "business").astype(int)
        + 0.07 * (df["industry"] == "software").astype(int)
        - 0.28 * df["support_per_active_user"]
    )
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    df[TARGET_COLUMN] = rng.binomial(1, probabilities)

    return df


if __name__ == "__main__":
    df = generate_demo_dataset()
    output_path = Path(__file__).parent / "data" / "demo_expansion.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows -> {output_path}")
