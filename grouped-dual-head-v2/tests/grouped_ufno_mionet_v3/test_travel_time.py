from __future__ import annotations

import pytest
import torch

from grouped_ufno_mionet_v3.model.travel_time import (
    dense_query_coordinates,
    straight_ray_travel_time,
)


def test_uniform_medium_matches_distance_over_velocity():
    velocity = torch.full((1, 1, 11, 13), 2000.0)
    source = torch.tensor([[20.0, 30.0]])
    query = torch.tensor([[[20.0, 30.0], [80.0, 90.0], [120.0, 0.0]]])
    result = straight_ray_travel_time(
        velocity,
        source,
        query,
        x_extent_m=(0.0, 120.0),
        z_extent_m=(0.0, 100.0),
        samples=12,
    )
    expected = torch.linalg.vector_norm(query - source[:, None], dim=-1) / 2000.0
    torch.testing.assert_close(result.seconds, expected, rtol=2.0e-5, atol=2.0e-7)
    torch.testing.assert_close(result.path_velocity_mps, torch.full_like(expected, 2000.0))
    torch.testing.assert_close(result.endpoint_velocity_mps, torch.full_like(expected, 2000.0))


def test_travel_time_is_differentiable_through_velocity_and_coordinates():
    z = torch.linspace(1800.0, 2600.0, 9)[:, None]
    velocity = z.expand(9, 9).clone()[None, None].requires_grad_()
    source = torch.tensor([[10.0, 10.0]], requires_grad=True)
    query = torch.tensor([[[60.0, 70.0], [75.0, 20.0]]], requires_grad=True)
    result = straight_ray_travel_time(
        velocity,
        source,
        query,
        x_extent_m=(0.0, 80.0),
        z_extent_m=(0.0, 80.0),
        samples=12,
    )
    result.seconds.sum().backward()
    for gradient in (velocity.grad, source.grad, query.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_record_to_medium_reuses_velocity_for_independent_sources():
    velocity = torch.stack(
        [torch.full((1, 7, 7), 2000.0), torch.full((1, 7, 7), 3000.0)]
    )
    sources = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
    query = torch.tensor([[[60.0, 0.0]], [[60.0, 0.0]], [[60.0, 0.0]]])
    result = straight_ray_travel_time(
        velocity,
        sources,
        query,
        record_to_medium=torch.tensor([0, 0, 1]),
        x_extent_m=(0.0, 60.0),
        z_extent_m=(0.0, 60.0),
        samples=4,
    )
    torch.testing.assert_close(result.seconds[:, 0], torch.tensor([0.03, 0.025, 0.02]))


def test_dense_coordinate_helper_matches_flattened_query_order():
    x = torch.tensor([0.0, 10.0, 20.0])
    z = torch.tensor([0.0, 5.0])
    coords = dense_query_coordinates(x, z, records=2)
    assert coords.shape == (2, 6, 2)
    torch.testing.assert_close(
        coords[0],
        torch.tensor([[0, 0], [10, 0], [20, 0], [0, 5], [10, 5], [20, 5]], dtype=x.dtype),
    )
    torch.testing.assert_close(coords[0], coords[1])


def test_travel_time_rejects_out_of_domain_queries():
    velocity = torch.full((1, 1, 5, 5), 2000.0)
    with pytest.raises(ValueError, match="domain"):
        straight_ray_travel_time(
            velocity,
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[[101.0, 50.0]]]),
            x_extent_m=(0.0, 100.0),
            z_extent_m=(0.0, 100.0),
        )
