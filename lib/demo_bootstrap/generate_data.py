"""Generate synthetic demo data for ``demo_bootstrap``. Not imported by runners."""

from __future__ import annotations


import numpy as np
import pandas as pd


from lib.demo_bootstrap.data import DATA_PATH, TARGET_COLUMN, TIME_COLUMN

SEED = 7
N_ROWS = 800


def generate_demo_bootstrap_dataset(
    n_rows: int = N_ROWS, seed: int = SEED
) -> pd.DataFrame:
    """Return the deterministic synthetic bootstrap demo dataset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-06-30", periods=n_rows)

    segment = rng.choice(["smb", "mid", "enterprise"], size=n_rows, p=[0.5, 0.35, 0.15])
    sessions_7d = rng.poisson(
        lam=np.where(segment == "enterprise", 6.0, 3.0), size=n_rows
    )
    pages_viewed = rng.poisson(lam=sessions_7d * 2.4 + 1.0, size=n_rows)
    days_active = rng.integers(0, 8, size=n_rows)

    t = np.linspace(0, 1, n_rows)
    logit = (
        -1.2
        + 0.35 * (segment == "enterprise").astype(float)
        + 0.12 * sessions_7d
        + 0.04 * pages_viewed
        + 0.15 * (days_active >= 4).astype(float)
        + 0.4 * t
        + rng.normal(0, 0.35, size=n_rows)
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    converted = rng.binomial(1, prob)

    return pd.DataFrame(
        {
            "user_id": [f"u{i:05d}" for i in range(n_rows)],
            TIME_COLUMN: dates,
            "sessions_7d": sessions_7d,
            "pages_viewed": pages_viewed,
            "days_active": days_active,
            "segment": segment,
            TARGET_COLUMN: converted,
        }
    )


def main() -> None:
    df = generate_demo_bootstrap_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
