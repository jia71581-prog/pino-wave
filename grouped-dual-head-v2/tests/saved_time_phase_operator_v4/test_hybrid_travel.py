import numpy as np
import torch

from grouped_ufno_mionet_v3.model.travel_time import (
    dense_query_coordinates,
    straight_ray_travel_time,
)
from saved_time_phase_operator_v4.hybrid_travel import straight_ray_grid_numpy


def test_numpy_straight_ray_grid_matches_operator_feature():
    rng = np.random.default_rng(19)
    velocity = rng.uniform(1500.0, 4200.0, size=(9, 11)).astype(np.float32)
    x_m = np.linspace(0.0, 2000.0, velocity.shape[1], dtype=np.float32)
    z_m = np.linspace(0.0, 2000.0, velocity.shape[0], dtype=np.float32)
    source_x = np.asarray([300.0, 1700.0], dtype=np.float32)
    source_z = np.asarray([0.0, 250.0], dtype=np.float32)

    actual = straight_ray_grid_numpy(
        velocity,
        source_x_m=source_x,
        source_z_m=source_z,
        x_m=x_m,
        z_m=z_m,
        samples=12,
    )
    velocity_t = torch.from_numpy(velocity)[None, None].expand(2, -1, -1, -1)
    sources_t = torch.from_numpy(np.stack((source_x, source_z), axis=-1))
    queries = dense_query_coordinates(
        torch.from_numpy(x_m), torch.from_numpy(z_m), records=2
    )
    expected = straight_ray_travel_time(
        velocity_t,
        sources_t,
        queries,
        x_extent_m=(0.0, 2000.0),
        z_extent_m=(0.0, 2000.0),
        samples=12,
    ).seconds.reshape(2, len(z_m), len(x_m))

    np.testing.assert_allclose(actual, expected.numpy(), rtol=2.0e-6, atol=2.0e-7)


def test_uniform_straight_ray_grid_has_exact_radial_travel_time():
    velocity = np.full((5, 7), 2000.0, dtype=np.float32)
    x_m = np.linspace(0.0, 600.0, 7, dtype=np.float32)
    z_m = np.linspace(0.0, 400.0, 5, dtype=np.float32)

    travel = straight_ray_grid_numpy(
        velocity,
        source_x_m=np.asarray([300.0], dtype=np.float32),
        source_z_m=np.asarray([0.0], dtype=np.float32),
        x_m=x_m,
        z_m=z_m,
    )[0]
    zz, xx = np.meshgrid(z_m, x_m, indexing="ij")
    expected = np.sqrt((xx - 300.0) ** 2 + zz**2) / 2000.0

    np.testing.assert_allclose(travel, expected, rtol=1.0e-6, atol=1.0e-7)
