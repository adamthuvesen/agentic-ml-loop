"""Tests for ``lib.paths`` — canonical experiment output directories."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from lib.paths import outputs_dir, scripts_dir, work_dir


@pytest.fixture
def exp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


def test_outputs_dir_returns_canonical_path(exp_dir):
    result = outputs_dir(exp_dir)
    assert result == exp_dir / "outputs"


def test_work_dir_returns_canonical_path(exp_dir):
    result = work_dir(exp_dir)
    assert result == exp_dir / "work"


def test_scripts_dir_returns_canonical_path(exp_dir):
    result = scripts_dir(exp_dir)
    assert result == exp_dir / "scripts"


def test_outputs_dir_creates_directory(exp_dir):
    target = exp_dir / "outputs"
    assert not target.exists()
    outputs_dir(exp_dir)
    assert target.is_dir()


def test_work_dir_creates_directory(exp_dir):
    target = exp_dir / "work"
    assert not target.exists()
    work_dir(exp_dir)
    assert target.is_dir()


def test_scripts_dir_creates_directory(exp_dir):
    target = exp_dir / "scripts"
    assert not target.exists()
    scripts_dir(exp_dir)
    assert target.is_dir()


def test_outputs_dir_is_idempotent(exp_dir):
    first = outputs_dir(exp_dir)
    second = outputs_dir(exp_dir)
    assert first == second
    assert first.is_dir()


def test_work_dir_is_idempotent(exp_dir):
    first = work_dir(exp_dir)
    second = work_dir(exp_dir)
    assert first == second
    assert first.is_dir()


def test_scripts_dir_is_idempotent(exp_dir):
    first = scripts_dir(exp_dir)
    second = scripts_dir(exp_dir)
    assert first == second
    assert first.is_dir()


def test_helpers_accept_str_path(exp_dir):
    """Helpers must work when callers pass a string instead of Path."""
    result = outputs_dir(str(exp_dir))
    assert result == exp_dir / "outputs"
    assert result.is_dir()
