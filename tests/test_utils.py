from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from lib.utils import write_json


class TestWriteJson:
    def test_writes_valid_json_atomically_visible(self, tmp_path: Path) -> None:
        path = tmp_path / "payload.json"

        write_json(
            path,
            {
                "score": np.float64(0.75),
                "counts": np.array([1, 2, 3]),
            },
        )

        assert json.loads(path.read_text()) == {"score": 0.75, "counts": [1, 2, 3]}
        assert not list(tmp_path.glob(".payload.json.*.tmp"))

    def test_rejects_non_finite_values_without_touching_existing_file(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "payload.json"
        path.write_text('{"ok": true}\n')

        with pytest.raises(ValueError, match="non-finite"):
            write_json(path, {"score": math.nan})

        assert path.read_text() == '{"ok": true}\n'
        assert not list(tmp_path.glob(".payload.json.*.tmp"))
