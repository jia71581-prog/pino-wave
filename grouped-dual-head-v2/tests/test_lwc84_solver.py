from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver


def test_lwc84_time_loop_has_no_per_step_host_scalar_extraction() -> None:
    source = inspect.getsource(LWC84CPMLSolver.simulate)
    loop = source.split("for step in range(1, maximum_step):", maxsplit=1)[1]
    loop = loop.split("output_np =", maxsplit=1)[0]
    assert ".item()" not in loop
    assert ".cpu()" not in loop
    assert "np.isfinite" not in loop
    assert "torch.isfinite" not in loop
    assert "ricker_triplet" not in loop


def _small_solver(
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    *,
    output_restriction_factor: int = 2,
) -> LWC84CPMLSolver:
    grid = AcousticGrid(nx=41, nz=41, dx_m=5.0, dz_m=5.0, lx_m=200.0, lz_m=200.0, centering="node")
    return LWC84CPMLSolver(
        grid=grid,
        boundaries=BoundaryConfig(npml=8),
        dt_s=2.0e-4,
        output_times_s=np.asarray([0.0, 0.01, 0.02], dtype=np.float64),
        c_ref_mps=3000.0,
        device=device,
        dtype=dtype,
        output_restriction_factor=output_restriction_factor,
    )


def test_solver_outputs_anti_aliased_21_grid_with_free_surface_and_source_metadata() -> None:
    solver = _small_solver()
    velocity = np.full((41, 41), 2000.0, dtype=np.float64)
    result = solver.simulate(
        velocity,
        source_x_m=103.25,
        source_z_m=52.75,
        source_f0_hz=80.0,
        source_amplitude=1.0,
    )
    assert result.wavefield.shape == (1, 3, 21, 21)
    assert result.velocity_saved_mps.shape == (1, 21, 21)
    assert result.source_map_saved.shape == (1, 21, 21)
    assert float(result.source_map_saved.sum()) == pytest.approx(1.0, abs=2.0e-7)
    assert np.count_nonzero(result.source_map_saved) == 4
    assert np.max(np.abs(result.wavefield[:, :, 0, :])) == 0.0
    assert np.isfinite(result.wavefield).all()
    assert float(np.max(np.abs(result.wavefield[:, 1:]))) > 0.0
    assert result.metrics[0]["cfl_2d"] < 0.45


def test_solver_factor_one_returns_physical_grid_and_conserves_source() -> None:
    result = _small_solver(output_restriction_factor=1).simulate(
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
    np.testing.assert_array_equal(
        result.velocity_saved_mps,
        np.full((1, 41, 41), 2000.0, dtype=np.float32),
    )


def test_explicit_factor_two_is_bitwise_identical_to_default() -> None:
    velocity = np.full((41, 41), 2000.0, dtype=np.float64)
    kwargs = {
        "source_x_m": 103.25,
        "source_z_m": 52.75,
        "source_f0_hz": 80.0,
        "source_amplitude": 1.0,
    }
    implicit = _small_solver().simulate(velocity, **kwargs)
    explicit = _small_solver(output_restriction_factor=2).simulate(velocity, **kwargs)
    np.testing.assert_array_equal(explicit.wavefield, implicit.wavefield)
    np.testing.assert_array_equal(explicit.velocity_saved_mps, implicit.velocity_saved_mps)
    np.testing.assert_array_equal(explicit.source_map_saved, implicit.source_map_saved)


def test_optional_exterior_auxiliary_captures_three_sided_cpml_state() -> None:
    result = _small_solver().simulate(
        np.full((41, 41), 2000.0, dtype=np.float64),
        source_x_m=10.0,
        source_z_m=50.0,
        source_f0_hz=80.0,
        capture_exterior_auxiliary=True,
    )
    auxiliary = result.exterior_auxiliary
    assert auxiliary is not None
    assert auxiliary.pressure_extended_saved.shape == (1, 3, 25, 29)
    assert auxiliary.velocity_extended_saved_mps.shape == (1, 25, 29)
    assert auxiliary.physical_slice_zx == ((0, 21), (4, 25))
    for memory in (
        auxiliary.psi_x_before_step,
        auxiliary.psi_z_before_step,
        auxiliary.phi_x_before_step,
        auxiliary.phi_z_before_step,
    ):
        assert memory.shape == auxiliary.pressure_extended_saved.shape
        assert np.isfinite(memory).all()
    pressure = auxiliary.pressure_extended_saved
    assert np.max(np.abs(pressure[:, :, 0, :])) == 0.0
    assert np.max(np.abs(pressure[:, :, -1, :])) == 0.0
    assert np.max(np.abs(pressure[:, :, :, 0])) == 0.0
    assert np.max(np.abs(pressure[:, :, :, -1])) == 0.0
    assert max(
        float(np.max(np.abs(auxiliary.psi_x_before_step))),
        float(np.max(np.abs(auxiliary.psi_z_before_step))),
        float(np.max(np.abs(auxiliary.phi_x_before_step))),
        float(np.max(np.abs(auxiliary.phi_z_before_step))),
    ) > 0.0


@pytest.mark.parametrize("factor", [0, 3, -1, 1.5])
def test_solver_rejects_unsupported_output_restriction_factor(factor) -> None:
    with pytest.raises(ValueError, match="output_restriction_factor must be 1 or 2"):
        _small_solver(output_restriction_factor=factor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_float64_and_gpu_float32_solver_agree_on_short_smoke() -> None:
    velocity = np.full((41, 41), 2000.0, dtype=np.float64)
    kwargs = {
        "source_x_m": 103.25,
        "source_z_m": 52.75,
        "source_f0_hz": 80.0,
        "source_amplitude": 1.0,
    }
    cpu = _small_solver("cpu", torch.float64).simulate(velocity, **kwargs).wavefield
    gpu = _small_solver("cuda", torch.float32).simulate(velocity.astype(np.float32), **kwargs).wavefield
    relative = np.linalg.norm(cpu - gpu) / max(np.linalg.norm(cpu), 1.0e-30)
    assert relative < 3.0e-3
