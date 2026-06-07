"""Generate the synthetic nonlinear tabular dataset for the PyTorch demo.

The label depends on ring-like and interaction terms that linear models miss but
a small MLP can learn. Not imported during agent cycles — reproducibility only.

Usage:
    uv run python lib/demo_deep/generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lib.demo_deep.data import TARGET_COLUMN, TIME_COLUMN

SEED = 42


def generate_demo_dataset(n_rows: int = 3000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", periods=n_rows)

    signal_a = rng.normal(0.0, 1.0, size=n_rows)
    signal_b = rng.normal(0.0, 1.0, size=n_rows)
    signal_c = rng.normal(0.0, 1.0, size=n_rows)
    signal_d = rng.normal(0.0, 1.0, size=n_rows)
    channel = rng.choice(
        ["web", "mobile", "partner", "email"],
        size=n_rows,
        p=[0.35, 0.30, 0.20, 0.15],
    )

    radius = np.sqrt(signal_a**2 + signal_b**2)
    ring = ((radius > 0.9) & (radius < 1.8)).astype(float)
    interaction = (signal_a * signal_b > 0).astype(float)
    wave = np.sin(2.0 * np.pi * signal_c) * np.cos(signal_d)
    channel_bias = np.where(channel == "partner", 0.35, np.where(channel == "email", -0.15, 0.0))

    logits = 2.2 * ring + 1.4 * interaction + 1.1 * wave + channel_bias - 0.6
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    target = rng.binomial(1, probabilities)

    return pd.DataFrame(
        {
            TIME_COLUMN: dates,
            "signal_a": signal_a,
            "signal_b": signal_b,
            "signal_c": signal_c,
            "signal_d": signal_d,
            "channel": channel,
            TARGET_COLUMN: target,
        }
    )


def main() -> None:
    out = Path(__file__).resolve().parent / "data" / "demo_nonlinear.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_dataset().to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
