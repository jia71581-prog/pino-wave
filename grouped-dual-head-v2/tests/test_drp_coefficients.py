from __future__ import annotations

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.cfl import _symbol_dt_limit
from fno_acoustic.data_generation.cpml import (
    CFSCPMLOperator,
    build_cfs_cpml_profiles,
)
from fno_acoustic.data_generation.high_order_teacher import _drp_radius4_coefficients
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from fno_acoustic.data_generation.stencils import second_derivatives_symmetric
from fno_acoustic.numerics.drp_coefficients import (
    optimized_drp_second_derivative_coefficients,
)


def _band_objective(center: float, positive: np.ndarray, upper: float) -> float:
    theta = np.linspace(0.0, upper, 200_001, dtype=np.float64)
    offsets = np.arange(1, positive.size + 1, dtype=np.float64)
    symbol = center + 2.0 * np.sum(
        positive[None, :] * np.cos(theta[:, None] * offsets), axis=1
    )
    return float(np.trapz(np.square(symbol + theta**2), theta))


def test_radius_four_drp_coefficients_are_consistent_and_deterministic() -> None:
    first = optimized_drp_second_derivative_coefficients(
        radius=4, max_nyquist_fraction=0.65
    )
    second = _drp_radius4_coefficients()
    assert first is second
    assert first.radius == 4
    assert first.symbol(0.0) == pytest.approx(0.0, abs=2.0e-12)
    assert sum(
        value * offset**2
        for offset, value in enumerate(first.positive_offsets, start=1)
    ) == pytest.approx(1.0, abs=2.0e-12)
    assert len(first.coefficient_sha256) == 64
    assert first.coefficient_sha256 == second.coefficient_sha256


def test_radius_four_drp_reduces_registered_band_objective() -> None:
    drp = _drp_radius4_coefficients()
    standard_positive = np.asarray(
        [8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0],
        dtype=np.float64,
    )
    standard_center = -2.0 * float(standard_positive.sum())
    upper = np.pi * 0.65
    drp_objective = _band_objective(
        drp.center, np.asarray(drp.positive_offsets), upper
    )
    standard_objective = _band_objective(
        standard_center, standard_positive, upper
    )
    assert drp_objective < 0.02 * standard_objective
    theta = np.linspace(0.0, np.pi, 8193, dtype=np.float64)
    assert float(np.max(drp.symbol(theta))) <= 2.0e-12


def test_drp_symbol_cfl_limit_is_finite_and_positive() -> None:
    limit = _symbol_dt_limit(6750.0, 10.0, 10.0)
    assert np.isfinite(limit)
    assert limit > 0.0


def test_drp_stencil_matches_its_periodic_fourier_symbol() -> None:
    coefficients = _drp_radius4_coefficients()
    n = 64
    mode = 13
    theta = 2.0 * np.pi * mode / n
    coordinate = torch.arange(n, dtype=torch.float64)
    field = torch.cos(theta * coordinate).repeat(n, 1)
    dxx, dzz = second_derivatives_symmetric(
        field,
        dx_m=1.0,
        dz_m=1.0,
        coefficients=coefficients,
        boundary="periodic",
    )
    torch.testing.assert_close(
        dxx,
        field * float(coefficients.symbol(theta)),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(dzz, torch.zeros_like(dzz), atol=2.0e-12, rtol=0.0)


def test_drp_lwc84_solver_is_explicit_opt_in_and_finite() -> None:
    common = dict(
        grid=AcousticGrid(
            nx=25,
            nz=25,
            dx_m=10.0,
            dz_m=10.0,
            lx_m=240.0,
            lz_m=240.0,
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=8),
        dt_s=2.0e-4,
        output_times_s=np.asarray([0.0, 0.005, 0.010]),
        c_ref_mps=3000.0,
        device="cpu",
        dtype=torch.float64,
        output_restriction_factor=1,
    )
    standard = LWC84CPMLSolver(**common)
    drp = LWC84CPMLSolver(**common, drp_max_nyquist_fraction=0.65)
    assert standard.spatial_coefficients is None
    assert drp.spatial_coefficients is _drp_radius4_coefficients()
    velocity = np.full((25, 25), 2000.0, dtype=np.float64)
    result = drp.simulate(
        velocity,
        source_x_m=120.0,
        source_z_m=60.0,
        source_f0_hz=80.0,
    )
    assert result.wavefield.shape == (1, 3, 25, 25)
    assert np.isfinite(result.wavefield).all()
    assert float(np.max(np.abs(result.wavefield[:, 1:]))) > 0.0


def test_drp_cpml_vectorized_path_uses_registered_physical_core() -> None:
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
        dt_s=2.0e-4,
        c_ref_mps=3000.0,
        device="cpu",
        dtype=torch.float64,
    )
    shape = boundaries.extended_shape(grid)
    field = torch.randn(shape, dtype=torch.float64)
    velocity = torch.full_like(field, 2000.0)
    coefficients = _drp_radius4_coefficients()
    operator = CFSCPMLOperator(
        profiles,
        dx_m=10.0,
        dz_m=10.0,
        physical_second_derivative_coefficients=coefficients,
    )
    coupled = operator.apply_repeated_zero_order_hold(
        field,
        velocity,
        substeps=2,
        update_memory=False,
    )
    dxx, dzz = second_derivatives_symmetric(
        field,
        dx_m=10.0,
        dz_m=10.0,
        coefficients=coefficients,
        boundary="free_surface",
    )
    physical_z, physical_x = boundaries.physical_slices(grid)
    core = velocity.square() * (dxx + dzz)
    torch.testing.assert_close(
        coupled[physical_z, physical_x],
        core[physical_z, physical_x],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
