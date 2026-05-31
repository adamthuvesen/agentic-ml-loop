"""Generate the bundled regression demo dataset.

This is the answer key for the synthetic regression demo. It exists for
reproducibility and should not be imported by the model-search loop itself.

Usage:
    python lib/demo_regression/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from lib.demo_regression.data import TIME_COLUMN, TARGET_COLUMN


SEED = 2026


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_demo_regression_dataset(
    n_rows: int = 4800, seed: int = SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", "2025-12-31", periods=n_rows)

    segment = rng.choice(
        ["smb", "mid_market", "enterprise"],
        size=n_rows,
        p=[0.42, 0.38, 0.20],
    )
    region = rng.choice(
        ["north_america", "emea", "apac", "latam"],
        size=n_rows,
        p=[0.37, 0.34, 0.17, 0.12],
    )
    industry = rng.choice(
        [
            "software",
            "financial_services",
            "healthcare",
            "education",
            "consulting",
            "other",
        ],
        size=n_rows,
        p=[0.24, 0.14, 0.15, 0.13, 0.16, 0.18],
    )

    plan_choices = {
        "smb": ["starter", "growth", "pro"],
        "mid_market": ["growth", "pro", "enterprise"],
        "enterprise": ["pro", "enterprise", "enterprise"],
    }
    plan_tier = np.array([rng.choice(plan_choices[s]) for s in segment])

    segment_scale = np.select(
        [segment == "smb", segment == "mid_market", segment == "enterprise"],
        [1.0, 2.2, 4.5],
    )
    plan_scale = np.select(
        [
            plan_tier == "starter",
            plan_tier == "growth",
            plan_tier == "pro",
            plan_tier == "enterprise",
        ],
        [0.8, 1.1, 1.5, 2.1],
    )

    account_age_days = rng.integers(30, 2200, size=n_rows)
    seat_count = np.maximum(
        5,
        (
            rng.lognormal(mean=np.log(28 * segment_scale), sigma=0.55, size=n_rows)
        ).astype(int),
    )

    latent_utilization = np.clip(
        rng.normal(0.38, 0.16, size=n_rows)
        + 0.05 * (plan_tier == "pro")
        + 0.08 * (plan_tier == "enterprise")
        + 0.04 * (industry == "software")
        - 0.03 * (industry == "education"),
        0.05,
        1.15,
    )
    avg_weekly_active_users = np.clip(
        seat_count * latent_utilization + rng.normal(0, 7.0, size=n_rows),
        1.0,
        None,
    )

    quarter_index = ((dates.year - dates.year.min()) * 4 + dates.quarter).to_numpy()
    usage_growth_90d = np.clip(
        rng.normal(0.03, 0.18, size=n_rows)
        + 0.015 * (plan_tier == "pro")
        + 0.03 * (plan_tier == "enterprise")
        + 0.006 * quarter_index
        + 0.02 * (industry == "software")
        - 0.015 * (industry == "education"),
        -0.45,
        0.75,
    )

    feature_adoption_count = np.clip(
        rng.poisson(2.8 + 1.1 * plan_scale + 1.8 * latent_utilization, size=n_rows),
        0,
        14,
    )

    has_exec_sponsor = rng.binomial(
        1,
        np.clip(
            0.14 + 0.08 * (segment != "smb") + 0.08 * (plan_tier == "enterprise"),
            0.05,
            0.65,
        ),
        size=n_rows,
    ).astype(bool)
    multi_product = rng.binomial(
        1,
        np.clip(
            0.10 + 0.09 * (plan_tier == "pro") + 0.16 * (plan_tier == "enterprise"),
            0.05,
            0.70,
        ),
        size=n_rows,
    ).astype(bool)
    previous_expansion = rng.binomial(
        1,
        np.clip(
            0.08 + 0.05 * (segment != "smb") + 0.03 * (industry == "software"),
            0.03,
            0.45,
        ),
        size=n_rows,
    ).astype(bool)

    previous_revenue_change_365d = previous_expansion.astype(float) * rng.gamma(
        shape=2.3, scale=230.0 * segment_scale * plan_scale, size=n_rows
    )

    champion_engagement_score = np.clip(
        100
        * rng.beta(
            1.8
            + 0.7 * has_exec_sponsor.astype(float)
            + 0.5 * multi_product.astype(float),
            3.4,
            size=n_rows,
        ),
        0,
        100,
    )

    support_tickets_90d = rng.poisson(
        1.2 + 0.004 * seat_count + 0.02 * np.maximum(45 - champion_engagement_score, 0),
        size=n_rows,
    )
    nps_score = np.clip(
        rng.normal(
            28
            + 10 * latent_utilization
            + 0.12 * champion_engagement_score
            - 1.9 * support_tickets_90d,
            16,
            size=n_rows,
        ),
        -100,
        100,
    )
    contract_value_t0 = np.clip(
        seat_count * plan_scale * segment_scale * rng.normal(55, 11, size=n_rows),
        500,
        None,
    )
    renewal_window_days = rng.integers(20, 360, size=n_rows)
    marketing_touchpoints_90d = rng.poisson(2.0 + 0.003 * seat_count, size=n_rows)
    webinar_attendance_90d = rng.poisson(
        0.8 + 0.0015 * seat_count + 0.3 * (industry == "software"), size=n_rows
    )

    df = pd.DataFrame(
        {
            TIME_COLUMN: dates,
            "account_age_days": account_age_days,
            "seat_count": seat_count,
            "avg_weekly_active_users": avg_weekly_active_users,
            "usage_growth_90d": usage_growth_90d,
            "feature_adoption_count": feature_adoption_count,
            "champion_engagement_score": champion_engagement_score,
            "support_tickets_90d": support_tickets_90d,
            "nps_score": nps_score,
            "contract_value_t0": contract_value_t0,
            "renewal_window_days": renewal_window_days,
            "previous_revenue_change_365d": previous_revenue_change_365d,
            "marketing_touchpoints_90d": marketing_touchpoints_90d,
            "webinar_attendance_90d": webinar_attendance_90d,
            "region": region,
            "industry": industry,
            "plan_tier": plan_tier,
            "segment": segment,
            "has_exec_sponsor": has_exec_sponsor,
            "multi_product": multi_product,
            "previous_expansion": previous_expansion,
        }
    )

    df["seat_utilization"] = (
        df["avg_weekly_active_users"] / np.maximum(df["seat_count"], 1)
    ).clip(0, 1.5)
    df["support_burden"] = (
        df["support_tickets_90d"] / np.maximum(df["avg_weekly_active_users"], 1)
    ).clip(0, 1.5)
    df["growth_efficiency"] = (
        np.maximum(df["usage_growth_90d"], -0.20) * df["seat_utilization"]
    )
    df["champion_x_growth"] = (
        df["champion_engagement_score"] / 100.0 * np.maximum(df["usage_growth_90d"], 0)
    )
    df["expansion_capacity"] = (
        np.maximum(df["contract_value_t0"], 0)
        * np.maximum(df["seat_utilization"], 0)
        * np.maximum(df["usage_growth_90d"] + 0.08, 0)
    ) / 100.0

    industry_effect = np.select(
        [
            df["industry"] == "software",
            df["industry"] == "financial_services",
            df["industry"] == "healthcare",
            df["industry"] == "education",
            df["industry"] == "consulting",
        ],
        [0.18, 0.10, 0.04, -0.10, 0.02],
        default=0.0,
    )
    plan_effect = np.select(
        [
            df["plan_tier"] == "starter",
            df["plan_tier"] == "growth",
            df["plan_tier"] == "pro",
            df["plan_tier"] == "enterprise",
        ],
        [-0.10, 0.00, 0.12, 0.22],
    )
    segment_effect = np.select(
        [
            df["segment"] == "smb",
            df["segment"] == "mid_market",
            df["segment"] == "enterprise",
        ],
        [-0.08, 0.04, 0.14],
    )

    expansion_logit = (
        -1.35
        + 2.6 * np.maximum(df["growth_efficiency"], -0.08)
        + 0.03 * df["feature_adoption_count"]
        + 0.012 * df["champion_engagement_score"]
        + 0.45 * df["has_exec_sponsor"].astype(float)
        + 0.28 * df["multi_product"].astype(float)
        + 0.24 * df["previous_expansion"].astype(float)
        + 0.000032 * df["contract_value_t0"]
        + industry_effect
        + plan_effect
        + segment_effect
        - 1.2 * df["support_burden"]
        - 0.0022 * np.maximum(df["renewal_window_days"] - 150, 0)
        + 0.015 * quarter_index
    )
    expansion_probability = _sigmoid(expansion_logit)
    expanding = rng.binomial(1, expansion_probability, size=n_rows)

    size_signal = (
        0.020 * df["contract_value_t0"]
        + 18.0 * df["feature_adoption_count"]
        + 1.4 * df["champion_engagement_score"]
        + 0.18 * df["previous_revenue_change_365d"]
        + 780.0
        * np.maximum(df["usage_growth_90d"], 0)
        * np.clip(df["seat_utilization"], 0, 1.2)
        + 180.0 * df["has_exec_sponsor"].astype(float)
        + 150.0 * df["multi_product"].astype(float)
        + 110.0 * (df["plan_tier"].isin(["pro", "enterprise"])).astype(float)
        - 24.0 * df["support_tickets_90d"]
        - 120.0 * np.maximum(df["support_burden"] - 0.05, 0)
        + 0.045 * df["account_age_days"]
    )
    interaction_boost = 420.0 * df["has_exec_sponsor"].astype(float) * df[
        "multi_product"
    ].astype(float) * np.maximum(df["usage_growth_90d"], 0) + 190.0 * (
        df["feature_adoption_count"] >= 7
    ).astype(float) * (df["segment"] != "smb").astype(float)
    noise_scale = (
        180.0
        + 0.02 * df["contract_value_t0"]
        + 130.0 * np.maximum(df["support_burden"], 0)
    )
    raw_amount = expanding * (
        size_signal + interaction_boost + rng.normal(0, noise_scale, size=n_rows)
    ) + (1 - expanding) * rng.normal(25.0, 95.0, size=n_rows)
    df[TARGET_COLUMN] = np.clip(raw_amount, 0, None)

    return df


if __name__ == "__main__":
    df = generate_demo_regression_dataset()
    output_path = Path(__file__).parent / "data" / "demo_revenue_change.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows -> {output_path}")
