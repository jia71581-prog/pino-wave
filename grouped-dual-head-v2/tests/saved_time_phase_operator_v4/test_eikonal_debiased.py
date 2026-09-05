from __future__ import annotations

import numpy as np

from saved_time_phase_operator_v4.eikonal import debiased_grid_eikonal_travel_time


def test_debiased_grid_eikonal_is_exact_for_uniform_fractional_source():
    velocity = np.full((21, 25), 2500.0, dtype=np.float32)
    source_z, source_x = 37.25, 81.75
    travel = debiased_grid_eikonal_travel_time(
        velocity,
        source_coordinates_m=[(source_z, source_x)],
        dx_m=10.0,
        dz_m=10.0,
    )[0]
    z = np.arange(21, dtype=np.float32) * 10.0
    x = np.arange(25, dtype=np.float32) * 10.0
    zz, xx = np.meshgrid(z, x, indexing="ij")
    exact = np.sqrt((xx - source_x) ** 2 + (zz - source_z) ** 2) / 2500.0
    np.testing.assert_allclose(travel, exact, rtol=2.0e-6, atol=2.0e-7)


def test_debiased_grid_eikonal_supports_exterior_coordinate_origin():
    velocity = np.full((23, 29), 3000.0, dtype=np.float32)
    travel = debiased_grid_eikonal_travel_time(
        velocity,
        source_coordinates_m=[(50.0, 20.0)],
        dx_m=10.0,
        dz_m=10.0,
        x0_m=-20.0,
    )[0]
    assert travel.shape == velocity.shape
    assert np.isfinite(travel).all()
    assert float(travel.min()) >= 0.0
