from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scripts.train_b2_group_disjoint_ddp import (
    AnchoredDDPModule,
    ddp_loss_scale,
    split_global_batch,
)


def test_global_batch_three_splits_two_plus_one_without_overlap():
    batch = np.asarray([7, 11, 19])
    parts = split_global_batch(batch, 2)
    assert [part.tolist() for part in parts] == [[7, 11], [19]]
    assert sorted(np.concatenate(parts).tolist()) == sorted(batch.tolist())


def test_weighted_rank_means_equal_global_sample_mean_after_ddp_average():
    rank0_losses = np.asarray([1.0, 4.0])
    rank1_losses = np.asarray([9.0])
    rank0_scaled = rank0_losses.mean() * ddp_loss_scale(2, 3, 2)
    rank1_scaled = rank1_losses.mean() * ddp_loss_scale(1, 3, 2)
    ddp_average = (rank0_scaled + rank1_scaled) / 2.0
    assert np.isclose(ddp_average, np.concatenate([rank0_losses, rank1_losses]).mean())


def test_ddp_visible_forward_dispatches_to_anchored_path():
    class FakePropagator(nn.Module):
        def forward_anchored(self, base_seq, cond, initial_state=None):
            return base_seq + cond[:, None, :1] + initial_state[:, None, :1]

    wrapped = AnchoredDDPModule(FakePropagator())
    base = torch.zeros(1, 2, 1, 3, 3)
    cond = torch.ones(1, 1, 3, 3)
    initial = torch.full((1, 1, 3, 3), 2.0)
    assert torch.equal(wrapped(base, cond, initial), torch.full_like(base, 3.0))
