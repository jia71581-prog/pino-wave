from __future__ import annotations

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from fno_acoustic.data_generation.solver_lwc84_fused import FusedLWC84CPMLSolver


@pytest.mark.parametrize("drp_fraction", [None, 0.45])
@pytest.mark.parametrize("restriction", [1, 2])
@pytest.mark.parametrize(
    ("fuse_saved_interval", "block_step_count"),
    [(False, None), (True, None), (False, 2)],
)
def test_full_fused_solver_matches_production(
    drp_fraction: float | None,
    restriction: int,
    fuse_saved_interval: bool,
    block_step_count: int | None,
) -> None:
    grid = AcousticGrid(
        nx=25,
        nz=25,
        dx_m=10.0,
        dz_m=10.0,
        lx_m=240.0,
        lz_m=240.0,
        centering="node",
    )
    common = {
        "grid": grid,
        "boundaries": BoundaryConfig(npml=8),
        "dt_s": 2.0e-4,
        "output_times_s": np.arange(4, dtype=np.float64) * 4.0e-4,
        "c_ref_mps": 3000.0,
        "device": "cpu",
        "dtype": torch.float64,
        "output_restriction_factor": restriction,
        "drp_max_nyquist_fraction": drp_fraction,
    }
    velocity = np.linspace(1800.0, 2600.0, 25 * 25).reshape(25, 25)
    source = {
        "source_x_m": 113.0,
        "source_z_m": 77.0,
        "source_f0_hz": 18.0,
        "source_t0_s": 0.03,
        "source_amplitude": 1.2,
    }
    expected = LWC84CPMLSolver(**common).simulate(velocity, **source)
    solver = FusedLWC84CPMLSolver(
        compile_graph=False,
        fuse_saved_interval=fuse_saved_interval,
        block_step_count=block_step_count,
        **common,
    )
    assert solver.warmup(batch=1) >= 0.0
    actual = solver.simulate(velocity, **source)

    np.testing.assert_allclose(actual.wavefield, expected.wavefield, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual.velocity_saved_mps, expected.velocity_saved_mps)
    np.testing.assert_array_equal(actual.source_map_saved, expected.source_map_saved)
    np.testing.assert_array_equal(actual.source_map_solver, expected.source_map_solver)
    np.testing.assert_array_equal(actual.source_wavelet, expected.source_wavelet)
    for actual_metrics, expected_metrics in zip(
        actual.metrics, expected.metrics, strict=True
    ):
        for key in expected_metrics:
            if key != "compute_elapsed_s":
                assert actual_metrics[key] == expected_metrics[key]


def test_fused_solver_rejects_nonpositive_warmup_batch() -> None:
    solver = FusedLWC84CPMLSolver(
        grid=AcousticGrid(
            nx=17,
            nz=17,
            dx_m=10.0,
            dz_m=10.0,
            lx_m=160.0,
            lz_m=160.0,
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=4),
        dt_s=2.0e-4,
        output_times_s=np.asarray([0.0, 2.0e-4]),
        c_ref_mps=2500.0,
    )
    with pytest.raises(ValueError, match="warmup batch"):
        solver.warmup(batch=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_graph_solver_matches_production() -> None:
    grid = AcousticGrid(
        nx=25,
        nz=25,
        dx_m=10.0,
        dz_m=10.0,
        lx_m=240.0,
        lz_m=240.0,
        centering="node",
    )
    common = {
        "grid": grid,
        "boundaries": BoundaryConfig(npml=8),
        "dt_s": 2.0e-4,
        "output_times_s": np.arange(5, dtype=np.float64) * 4.0e-4,
        "c_ref_mps": 3000.0,
        "device": "cuda",
        "dtype": torch.float32,
        "output_restriction_factor": 1,
    }
    velocity = np.linspace(1800.0, 2600.0, 25 * 25).reshape(25, 25)
    source = {
        "source_x_m": 113.0,
        "source_z_m": 77.0,
        "source_f0_hz": 18.0,
        "source_t0_s": 0.03,
        "source_amplitude": 1.2,
    }
    expected = LWC84CPMLSolver(**common).simulate(velocity, **source)
    solver = FusedLWC84CPMLSolver(cuda_graphs=True, **common)
    solver.warmup(batch=1)
    actual = solver.simulate(velocity, **source)
    np.testing.assert_allclose(
        actual.wavefield, expected.wavefield, rtol=2.0e-5, atol=1.0e-12
    )
