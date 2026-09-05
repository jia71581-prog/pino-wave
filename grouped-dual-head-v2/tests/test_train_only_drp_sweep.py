from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.sweep_train_only_drp_lwc84 import (
    _aggregate,
    _select_train_marmousi_quantiles,
)


def test_drp_sweep_selects_only_train_marmousi_quantiles(tmp_path: Path) -> None:
    path = tmp_path / "selection.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "split",
            data=np.asarray(
                [b"validation", b"train", b"train", b"test_id", b"train", b"train"]
            ),
        )
        handle.create_dataset(
            "medium_type",
            data=np.asarray(
                [b"marmousi", b"marmousi", b"uniform", b"marmousi", b"marmousi", b"marmousi"]
            ),
        )
        handle.create_dataset(
            "source_f0_hz",
            data=np.asarray([30.0, 20.0, 29.0, 25.0, 10.0, 15.0]),
        )
    with h5py.File(path, "r") as handle:
        selected = _select_train_marmousi_quantiles(handle, count=3)
        np.testing.assert_array_equal(selected, np.asarray([4, 5, 1]))
        split = handle["split"].asstr()[:]
        family = handle["medium_type"].asstr()[:]
        assert split[selected].tolist() == ["train"] * 3
        assert family[selected].tolist() == ["marmousi"] * 3


def test_drp_sweep_ranking_uses_mean_then_maximum_error() -> None:
    rows = [
        {"variant": "a", "relative_l2": 0.2, "wall_seconds": 1.0},
        {"variant": "a", "relative_l2": 0.4, "wall_seconds": 3.0},
        {"variant": "b", "relative_l2": 0.25, "wall_seconds": 2.0},
        {"variant": "b", "relative_l2": 0.30, "wall_seconds": 2.0},
    ]
    summary = _aggregate(rows)
    assert summary["ranking"] == ["b", "a"]
    assert summary["by_variant"]["a"]["mean_relative_l2"] == pytest.approx(0.3)
