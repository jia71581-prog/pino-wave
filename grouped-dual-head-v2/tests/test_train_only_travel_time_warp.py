from __future__ import annotations

import torch

from scripts.probe_train_only_travel_time_warp_capacity import _travel_warp


def test_zero_travel_warp_is_identity() -> None:
    torch.manual_seed(815)
    field = torch.randn(7, 3, 4)
    travel = torch.rand(3, 4)
    time = torch.arange(7, dtype=torch.float32) * 0.25
    actual = _travel_warp(field, travel, time, scale=0.0)
    torch.testing.assert_close(actual, field, rtol=0.0, atol=0.0)


def test_positive_travel_warp_samples_later_times_and_zeros_outside() -> None:
    field = torch.arange(5, dtype=torch.float32)[:, None, None]
    travel = torch.full((1, 1), 0.5)
    time = torch.arange(5, dtype=torch.float32)
    actual = _travel_warp(field, travel, time, scale=1.0)[:, 0, 0]
    expected = torch.tensor([0.5, 1.5, 2.5, 3.5, 0.0])
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
