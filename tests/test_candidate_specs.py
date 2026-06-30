from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.external_targets.candidate_specs import apply_derived_features


def test_apply_derived_features_computes_supported_ops() -> None:
    frame = pd.DataFrame({"a": [1.0, 4.0, 0.0], "b": [1.0, 2.0, 0.0]})
    specs = [
        {"name": "log_a", "op": "log1p", "source": "a"},
        {"name": "sqrt_a", "op": "sqrt", "source": "a"},
        {"name": "square_a", "op": "square", "source": "a"},
        {"name": "clip_a", "op": "clip", "source": "a", "min": 2, "max": 3},
        {"name": "ratio_ab", "op": "ratio", "left": "a", "right": "b"},
        {"name": "diff_ab", "op": "difference", "left": "a", "right": "b"},
        {"name": "prod_ab", "op": "product", "left": "a", "right": "b"},
        {"name": "sum_ab", "op": "sum", "left": "a", "right": "b"},
        {"name": "gt_a", "op": "greater_than", "source": "a", "threshold": 1},
        {"name": "lt_a", "op": "less_than", "source": "a", "threshold": 1},
    ]

    out = apply_derived_features(frame, specs, forbidden_sources=set())

    np.testing.assert_allclose(out["log_a"], np.log1p(frame["a"]))
    np.testing.assert_allclose(out["sqrt_a"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(out["square_a"], [1.0, 16.0, 0.0])
    np.testing.assert_allclose(out["clip_a"], [2.0, 3.0, 2.0])
    np.testing.assert_allclose(out["ratio_ab"], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(out["diff_ab"], [0.0, 2.0, 0.0])
    np.testing.assert_allclose(out["prod_ab"], [1.0, 8.0, 0.0])
    np.testing.assert_allclose(out["sum_ab"], [2.0, 6.0, 0.0])
    np.testing.assert_allclose(out["gt_a"], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(out["lt_a"], [0.0, 0.0, 1.0])


def test_apply_derived_features_rejects_unsupported_ops() -> None:
    frame = pd.DataFrame({"a": [1.0]})

    with pytest.raises(ValueError, match="Unsupported derived feature op"):
        apply_derived_features(
            frame,
            [{"name": "bad_feature", "op": "unknown", "source": "a"}],
            forbidden_sources=set(),
        )
