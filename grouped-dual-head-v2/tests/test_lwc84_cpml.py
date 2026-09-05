from __future__ import annotations

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.cpml import CFSCPMLOperator, build_cfs_cpml_profiles
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.lwc84 import apply_l_operator


def test_cfs_cpml_profiles_are_only_left_right_and_bottom() -> None:
    grid = AcousticGrid(nx=41, nz=41, dx_m=5.0, dz_m=5.0, lx_m=200.0, lz_m=200.0, centering="node")
    boundaries = BoundaryConfig(npml=8)
    profiles = build_cfs_cpml_profiles(
        grid,
        boundaries,
        dt_s=2.0e-4,
        c_ref_mps=6000.0,
        target_reflection=1.0e-8,
        polynomial_order=3,
        kappa_max=3.0,
        minimum_frequency_hz=8.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    physical_z, physical_x = boundaries.physical_slices(grid)
    assert profiles.sigma_x.shape == boundaries.extended_shape(grid)
    assert torch.count_nonzero(profiles.sigma_x[physical_z, physical_x]) == 0
    assert torch.count_nonzero(profiles.sigma_z[physical_z, physical_x]) == 0
    assert torch.count_nonzero(profiles.sigma_z[0]) == 0
    assert torch.all(profiles.sigma_x[:, 0] > 0.0)
    assert torch.all(profiles.sigma_x[:, -1] > 0.0)
    assert torch.all(profiles.sigma_z[-1] > 0.0)
    assert profiles.sigma_max_s_inv > 0.0
    assert profiles.alpha_max_rad_s == pytest.approx(np.pi * 8.0)


def test_cpml_operator_matches_lwc84_core_inside_physical_region_and_updates_memory() -> None:
    grid = AcousticGrid(nx=41, nz=41, dx_m=5.0, dz_m=5.0, lx_m=200.0, lz_m=200.0, centering="node")
    boundaries = BoundaryConfig(npml=8)
    profiles = build_cfs_cpml_profiles(
        grid,
        boundaries,
        dt_s=2.0e-4,
        c_ref_mps=3000.0,
        target_reflection=1.0e-8,
        polynomial_order=3,
        kappa_max=3.0,
        minimum_frequency_hz=8.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    operator = CFSCPMLOperator(profiles, dx_m=5.0, dz_m=5.0)
    shape = boundaries.extended_shape(grid)
    pressure = torch.zeros(shape, dtype=torch.float64)
    pressure[12:30, 12:35] = torch.randn((18, 23), dtype=torch.float64)
    velocity = torch.full(shape, 2500.0, dtype=torch.float64)
    coupled = operator.apply(pressure, velocity, update_memory=True)
    core = apply_l_operator(pressure, velocity, dx_m=5.0, dz_m=5.0, boundary="free_surface")
    # Radius-eight margin keeps this comparison away from the CPML interface.
    physical_z, physical_x = boundaries.physical_slices(grid)
    assert torch.allclose(coupled[8:32, physical_x.start + 8 : physical_x.stop - 8], core[8:32, physical_x.start + 8 : physical_x.stop - 8])

    pml_pressure = torch.zeros_like(pressure)
    pml_pressure[:, 1:6] = 1.0
    operator.apply(pml_pressure, velocity, update_memory=True)
    assert operator.memory_norm() > 0.0
    operator.reset()
    assert operator.memory_norm() == 0.0


def test_vectorized_cpml_zero_order_hold_matches_repeated_updates() -> None:
    torch.manual_seed(742)
    grid = AcousticGrid(
        nx=25,
        nz=25,
        dx_m=10.0,
        dz_m=10.0,
        lx_m=240.0,
        lz_m=240.0,
        centering="node",
    )
    boundaries = BoundaryConfig(npml=8)
    profiles = build_cfs_cpml_profiles(
        grid,
        boundaries,
        dt_s=1.25e-4,
        c_ref_mps=6750.0,
        target_reflection=1.0e-8,
        polynomial_order=3,
        kappa_max=3.0,
        minimum_frequency_hz=8.0,
        device="cpu",
        dtype=torch.float64,
    )
    shape = boundaries.extended_shape(grid)
    field = torch.randn((2, *shape), dtype=torch.float64) * 1.0e-5
    velocity = torch.full_like(field, 2500.0)
    repeated = CFSCPMLOperator(profiles, dx_m=10.0, dz_m=10.0)
    vectorized = CFSCPMLOperator(profiles, dx_m=10.0, dz_m=10.0)
    expected = None
    for _ in range(20):
        expected = repeated.apply(field, velocity, update_memory=True)
    actual = vectorized.apply_repeated_zero_order_hold(
        field, velocity, substeps=20, update_memory=True
    )
    assert expected is not None
    torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-11)
    for vectorized_memory, repeated_memory in (
        (vectorized.psi_x, repeated.psi_x),
        (vectorized.psi_z, repeated.psi_z),
        (vectorized.phi_x, repeated.phi_x),
        (vectorized.phi_z, repeated.phi_z),
    ):
        assert vectorized_memory is not None and repeated_memory is not None
        torch.testing.assert_close(
            vectorized_memory, repeated_memory, rtol=2.0e-11, atol=2.0e-11
        )


def test_vectorized_cpml_linear_trajectory_matches_repeated_updates() -> None:
    torch.manual_seed(743)
    grid = AcousticGrid(
        nx=25, nz=25, dx_m=10.0, dz_m=10.0,
        lx_m=240.0, lz_m=240.0, centering="node",
    )
    boundaries = BoundaryConfig(npml=8)
    profiles = build_cfs_cpml_profiles(
        grid, boundaries, dt_s=1.25e-4, c_ref_mps=6750.0,
        target_reflection=1.0e-8, polynomial_order=3, kappa_max=3.0,
        minimum_frequency_hz=8.0, device="cpu", dtype=torch.float64,
    )
    shape = boundaries.extended_shape(grid)
    previous = torch.randn((2, *shape), dtype=torch.float64) * 1.0e-5
    current = torch.randn_like(previous) * 1.0e-5
    velocity = torch.full_like(previous, 2500.0)
    repeated = CFSCPMLOperator(profiles, dx_m=10.0, dz_m=10.0)
    vectorized = CFSCPMLOperator(profiles, dx_m=10.0, dz_m=10.0)
    expected = None
    for step in range(1, 21):
        field = previous + (current - previous) * (step / 20.0)
        expected = repeated.apply(field, velocity, update_memory=True)
    actual = vectorized.apply_linear_trajectory(
        previous, current, velocity, substeps=20, update_memory=True
    )
    assert expected is not None
    torch.testing.assert_close(actual, expected, rtol=2.0e-11, atol=2.0e-11)
    for vectorized_memory, repeated_memory in (
        (vectorized.psi_x, repeated.psi_x),
        (vectorized.psi_z, repeated.psi_z),
        (vectorized.phi_x, repeated.phi_x),
        (vectorized.phi_z, repeated.phi_z),
    ):
        assert vectorized_memory is not None and repeated_memory is not None
        torch.testing.assert_close(
            vectorized_memory, repeated_memory, rtol=2.0e-11, atol=2.0e-11
        )
