from __future__ import annotations

import pytest
import torch

from fno_acoustic.data_generation.cpml import (
    CFSCPMLOperator,
    build_cfs_cpml_profiles,
)
from fno_acoustic.data_generation.fused_lwc84 import (
    make_cfs_cpml_lwc84_block,
    make_cfs_cpml_lwc84_step,
)
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.lwc84 import apply_l_operator
from fno_acoustic.numerics.drp_coefficients import (
    optimized_drp_second_derivative_coefficients,
)


@pytest.mark.parametrize("use_drp", [False, True])
def test_functional_step_matches_production_operator(use_drp: bool) -> None:
    torch.manual_seed(811)
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
    coefficients = (
        optimized_drp_second_derivative_coefficients(
            radius=4,
            max_nyquist_fraction=0.65,
        )
        if use_drp
        else None
    )
    shape = boundaries.extended_shape(grid)
    p_nm1 = torch.randn((2, *shape), dtype=torch.float64) * 1.0e-7
    p_n = torch.randn_like(p_nm1) * 1.0e-7
    velocity = torch.full_like(p_nm1, 2200.0)
    q_n = torch.randn_like(p_nm1) * 1.0e-7
    qtt_n = torch.randn_like(p_nm1) * 1.0e-4
    memories = [torch.randn_like(p_nm1) * 1.0e-8 for _ in range(4)]

    operator = CFSCPMLOperator(
        profiles,
        dx_m=10.0,
        dz_m=10.0,
        physical_second_derivative_coefficients=coefficients,
    )
    operator.psi_x, operator.psi_z, operator.phi_x, operator.phi_z = (
        value.clone() for value in memories
    )
    acceleration = operator.apply(p_n, velocity, update_memory=True) + q_n
    fourth = apply_l_operator(
        acceleration,
        velocity,
        dx_m=10.0,
        dz_m=10.0,
        boundary="free_surface",
        spatial_coefficients=coefficients,
    ) + qtt_n
    expected = (
        2.0 * p_n
        - p_nm1
        + (2.0e-4) ** 2 * acceleration
        + (2.0e-4) ** 4 * fourth / 12.0
    )
    expected[..., 0, :] = 0.0
    expected[..., -1, :] = 0.0
    expected[..., :, 0] = 0.0
    expected[..., :, -1] = 0.0

    step = make_cfs_cpml_lwc84_step(
        profiles=profiles,
        dx_m=10.0,
        dz_m=10.0,
        dt_s=2.0e-4,
        spatial_coefficients=coefficients,
    )
    actual = step(
        p_nm1,
        p_n,
        *memories,
        q_n,
        qtt_n,
        velocity,
    )
    torch.testing.assert_close(actual[0], p_n, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected, rtol=2.0e-15, atol=2.0e-15)
    for actual_memory, expected_memory in zip(
        actual[2:],
        (operator.psi_x, operator.psi_z, operator.phi_x, operator.phi_z),
        strict=True,
    ):
        assert expected_memory is not None
        torch.testing.assert_close(
            actual_memory,
            expected_memory,
            rtol=2.0e-15,
            atol=2.0e-15,
        )


def test_functional_step_does_not_mutate_input_state() -> None:
    grid = AcousticGrid(
        nx=17,
        nz=17,
        dx_m=10.0,
        dz_m=10.0,
        lx_m=160.0,
        lz_m=160.0,
        centering="node",
    )
    profiles = build_cfs_cpml_profiles(
        grid,
        BoundaryConfig(npml=4),
        dt_s=2.0e-4,
        c_ref_mps=2500.0,
        dtype=torch.float64,
    )
    shape = (1, 21, 25)
    state = [torch.randn(shape, dtype=torch.float64) for _ in range(6)]
    snapshots = [value.clone() for value in state]
    zero = torch.zeros(shape, dtype=torch.float64)
    velocity = torch.full(shape, 2000.0, dtype=torch.float64)
    step = make_cfs_cpml_lwc84_step(
        profiles=profiles,
        dx_m=10.0,
        dz_m=10.0,
        dt_s=2.0e-4,
    )
    step(*state, zero, zero, velocity)
    for value, snapshot in zip(state, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0.0, atol=0.0)


def test_fixed_block_matches_repeated_functional_steps() -> None:
    grid = AcousticGrid(
        nx=17,
        nz=17,
        dx_m=10.0,
        dz_m=10.0,
        lx_m=160.0,
        lz_m=160.0,
        centering="node",
    )
    profiles = build_cfs_cpml_profiles(
        grid,
        BoundaryConfig(npml=4),
        dt_s=2.0e-4,
        c_ref_mps=2500.0,
        dtype=torch.float64,
    )
    torch.manual_seed(812)
    shape = (1, 21, 25)
    state = tuple(torch.randn(shape, dtype=torch.float64) * 1.0e-7 for _ in range(6))
    q = torch.randn((3, *shape), dtype=torch.float64) * 1.0e-8
    qtt = torch.randn((3, *shape), dtype=torch.float64) * 1.0e-5
    velocity = torch.full(shape, 2100.0, dtype=torch.float64)
    step = make_cfs_cpml_lwc84_step(
        profiles=profiles,
        dx_m=10.0,
        dz_m=10.0,
        dt_s=2.0e-4,
    )
    expected = state
    for index in range(3):
        expected = step(*expected, q[index], qtt[index], velocity)
    block = make_cfs_cpml_lwc84_block(
        profiles=profiles,
        dx_m=10.0,
        dz_m=10.0,
        dt_s=2.0e-4,
        step_count=3,
    )
    actual = block(*state, q, qtt, velocity)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            actual_value, expected_value, rtol=0.0, atol=0.0
        )
