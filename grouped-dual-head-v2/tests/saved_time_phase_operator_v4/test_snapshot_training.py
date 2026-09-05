"""CPU tests for snapshot-only rollout training utilities."""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from saved_time_phase_operator_v4.snapshot_training import (
    SnapshotWindowDataset,
    snapshot_rollout_loss,
)


def test_snapshot_rollout_loss_is_zero_on_exact_prediction():
    target = torch.randn(2, 4, 12, 14)
    loss, components = snapshot_rollout_loss(target.clone(), target)
    assert loss == 0.0
    assert all(value == 0.0 for value in components.values())


def test_snapshot_rollout_loss_is_finite_and_differentiable():
    prediction = torch.randn(1, 3, 10, 12, requires_grad=True)
    target = torch.randn_like(prediction)
    loss, components = snapshot_rollout_loss(prediction, target)
    assert torch.isfinite(loss)
    assert "multi_horizon" in components
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_snapshot_dataset_hard_rejects_middle_context(tmp_path):
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["dt_output_s"] = 0.0025
        h5.create_dataset("wavefield", data=np.ones((3, 401, 8, 8), dtype=np.float32))
        h5.create_dataset("split", data=np.asarray([b"validation"] * 3))
        h5.create_dataset(
            "medium_type", data=np.asarray([b"uniform", b"layered", b"marmousi"])
        )
        h5.create_dataset("source_t0_s", data=np.full(3, 0.05))
        h5.create_dataset("source_f0_hz", data=np.full(3, 10.0))
    with pytest.raises(ValueError, match="fixed validation snapshot windows are empty"):
        SnapshotWindowDataset(
            path,
            split="validation",
            history_frames=8,
            rollout_steps=4,
            seed=1,
            records_per_family=1,
            fixed_context_ends=(160,),
            minimum_context_end=48,
            maximum_context_end=240,
            maximum_context_fraction=1.0 / 3.0,
            post_peak_cycles=0.0,
        )
    dataset = SnapshotWindowDataset(
        path,
        split="validation",
        history_frames=8,
        rollout_steps=4,
        seed=1,
        records_per_family=1,
        fixed_context_ends=(80,),
        minimum_context_end=48,
        maximum_context_end=120,
        maximum_context_fraction=1.0 / 3.0,
        post_peak_cycles=0.0,
    )
    item = dataset[0]
    assert item["context_end_fraction"] < 1.0 / 3.0
    assert set(item) == {
        "wavefield_history", "target", "family", "record", "context_end",
        "context_end_fraction",
    }
    first = SnapshotWindowDataset(
        path,
        split="validation",
        history_frames=8,
        rollout_steps=4,
        seed=999,
        records_per_family=1,
        fixed_context_ends=(80,),
        minimum_context_end=48,
        maximum_context_end=120,
        maximum_context_fraction=1.0 / 3.0,
        post_peak_cycles=0.0,
        record_selection="first",
    )
    assert [first[index]["record"] for index in range(len(first))] == [0, 1, 2]
