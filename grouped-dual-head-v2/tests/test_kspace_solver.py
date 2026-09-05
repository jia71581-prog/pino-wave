from __future__ import annotations

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_kspace import KSpacePSTDSolver


def _solver(*, dt_s: float = 2.0e-4, temporal_order: int = 2) -> KSpacePSTDSolver:
    return KSpacePSTDSolver(
        grid=AcousticGrid(
            nx=41,
            nz=41,
            dx_m=5.0,
            dz_m=5.0,
            lx_m=200.0,
            lz_m=200.0,
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=8),
        dt_s=dt_s,
        output_times_s=np.asarray([0.0, 0.01, 0.02]),
        c_ref_mps=3000.0,
        device="cpu",
        dtype=torch.float64,
        temporal_order=temporal_order,
    )


def test_kspace_solver_preserves_saved_grid_and_source_contract() -> None:
    result = _solver().simulate(
        np.full((41, 41), 2000.0, dtype=np.float64),
        source_x_m=103.25,
        source_z_m=52.75,
        source_f0_hz=80.0,
    )
    assert result.wavefield.shape == (1, 3, 41, 41)
    assert result.velocity_saved_mps.shape == (1, 41, 41)
    assert result.source_map_saved.shape == (1, 41, 41)
    assert float(result.source_map_saved.sum()) == pytest.approx(1.0, abs=2.0e-7)
    assert np.count_nonzero(result.source_map_saved) == 4
    assert np.max(np.abs(result.wavefield[:, :, 0, :])) == 0.0
    assert np.isfinite(result.wavefield).all()
    assert float(np.max(np.abs(result.wavefield[:, 1:]))) > 0.0
    assert result.metrics[0]["internal_steps"] == 100


def test_kspace_solver_rejects_output_times_not_aligned_to_internal_dt() -> None:
    with pytest.raises(ValueError, match="align exactly"):
        _solver(dt_s=3.0e-4)


def test_fourth_order_kspace_solver_is_finite_and_respects_contract() -> None:
    result = _solver(temporal_order=4).simulate(
        np.full((41, 41), 2000.0, dtype=np.float64),
        source_x_m=103.25,
        source_z_m=52.75,
        source_f0_hz=80.0,
    )
    assert result.wavefield.shape == (1, 3, 41, 41)
    assert np.isfinite(result.wavefield).all()
    assert float(np.max(np.abs(result.wavefield[:, 1:]))) > 0.0
    assert result.metrics[0]["temporal_order"] == 4


def test_fine_grid_restriction_match_preserves_shape_and_free_surface() -> None:
    solver = KSpacePSTDSolver(
        grid=AcousticGrid(
            nx=41, nz=41, dx_m=5.0, dz_m=5.0,
            lx_m=200.0, lz_m=200.0, centering="node",
        ),
        boundaries=BoundaryConfig(npml=8),
        dt_s=2.0e-4,
        output_times_s=np.asarray([0.0, 0.01, 0.02]),
        c_ref_mps=1.0,
        damping_c_ref_mps=3000.0,
        device="cpu",
        dtype=torch.float64,
        temporal_order=4,
        match_fine_grid_restriction=True,
    )
    result = solver.simulate(
        np.full((41, 41), 2000.0),
        source_x_m=103.25,
        source_z_m=52.75,
        source_f0_hz=80.0,
    )
    assert result.wavefield.shape == (1, 3, 41, 41)
    assert np.max(np.abs(result.wavefield[:, :, 0, :])) == 0.0
    assert result.metrics[0]["match_fine_grid_restriction"] is True


def test_kspace_laplacian_annihilates_zero_and_keeps_shape() -> None:
    solver = _solver()
    field = torch.zeros((2, 49, 57), dtype=torch.float64)
    second_x, second_z = solver._spectral_second_derivatives(field)
    assert second_x.shape == field.shape
    assert second_z.shape == field.shape
    assert torch.count_nonzero(second_x) == 0
    assert torch.count_nonzero(second_z) == 0
