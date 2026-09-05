import torch

from scripts.probe_parent_correction_scale_oracle import (
    time_dilation_direction,
    travel_time_warp_direction,
)


def test_time_dilation_direction_matches_linear_time_field():
    times = torch.linspace(0.0, 1.0, 5)
    field = times[None, :, None, None].expand(2, 5, 3, 4).clone()
    onset = torch.tensor([0.2, 0.4])

    direction = time_dilation_direction(field, times, onset)

    expected = (times[None, :] - onset[:, None])[:, :, None, None].expand_as(field)
    torch.testing.assert_close(direction, expected)


def test_travel_time_warp_direction_scales_time_derivative_spatially():
    times = torch.linspace(0.0, 1.0, 5)
    field = times[None, :, None, None].expand(1, 5, 2, 3).clone()
    travel = torch.tensor([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])

    direction = travel_time_warp_direction(field, times, travel)

    torch.testing.assert_close(direction, travel[None, None].expand_as(field))
