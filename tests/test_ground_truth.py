"""Tests for ground-truth immutability."""

from __future__ import annotations

from pathlib import Path

import pytest

from avt.storage import GroundTruth


def test_identical_ground_truth_write_is_idempotent(tmp_path: Path) -> None:
    gt = GroundTruth(tmp_path / "ground_truth.sqlite")
    with gt.connect() as c:
        c.put("cand1", "task", 1.0, {"reward": 1.0})
        c.put("cand1", "task", 1.0, {"reward": 1.0})  # same payload -> no-op
        got = c.get("cand1")
        assert got is not None
        assert got["reward"] == 1.0


def test_conflicting_ground_truth_write_raises(tmp_path: Path) -> None:
    gt = GroundTruth(tmp_path / "ground_truth.sqlite")
    with gt.connect() as c:
        c.put("cand1", "task", 1.0, {"reward": 1.0})
        with pytest.raises(ValueError):
            c.put("cand1", "task", 0.0, {"reward": 1.0})  # changed reward
        with pytest.raises(ValueError):
            c.put("cand1", "other-task", 1.0, {"reward": 1.0})  # changed task
