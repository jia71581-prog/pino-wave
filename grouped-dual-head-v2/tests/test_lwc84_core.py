from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from fno_acoustic.data_generation.cfl import plan_lwc84_timestep
from fno_acoustic.data_generation.free_surface import pad_pressure_halo
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.lwc84 import apply_l_operator, lwc84_startup, lwc84_step
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.ricker import ricker_triplet
from fno_acoustic.data_generation.source import bilinear_point_source
from fno_acoustic.data_generation.stencils import SECOND_DERIVATIVE_8, laplacian8


def test_eighth_order_second_derivative_coefficients_satisfy_moments() -> None:
    offsets = np.arange(-4, 5, dtype=np.float64)
    coeffs = np.asarray(SECOND_DERIVATIVE_8, dtype=np.float64)
    assert coeffs.shape == (9,)
    for degree in range(9):
        moment = float(np.sum(coeffs * offsets**degree))
        expected = 2.0 if degree == 2 else 0.0
        assert moment == pytest.approx(expected, abs=2.0e-12)


def test_ricker_triplet_matches_centered_numerical_derivatives() -> None:
    f0 = 15.0
    t0 = 1.5 / f0
    h = 1.0e-6
    times = np.linspace(0.02, 0.18, 31, dtype=np.float64)
    w, wt, wtt = ricker_triplet(times, f0_hz=f0, t0_s=t0)
    wp = ricker_triplet(times + h, f0_hz=f0, t0_s=t0)[0]
    wm = ricker_triplet(times - h, f0_hz=f0, t0_s=t0)[0]
    wt_fd = (wp - wm) / (2.0 * h)
    wtt_fd = (wp - 2.0 * w + wm) / h**2
    assert float(w[np.argmax(w)]) == pytest.approx(1.0, rel=2.0e-3)
    assert np.allclose(wt, wt_fd, rtol=2.0e-7, atol=2.0e-7)
    assert np.allclose(wtt, wtt_fd, rtol=2.0e-5, atol=2.0e-3)


def test_lwc84_cfl_gate_keeps_or_uniformly_reduces_requested_dt() -> None:
    keep = plan_lwc84_timestep(
        global_vmax_mps=6000.0,
        dx_m=5.0,
        dz_m=5.0,
        dt_requested_s=2.0e-4,
        output_interval_s=5.0e-3,
    )
    assert keep.dt_used_s == pytest.approx(2.0e-4)
    assert keep.snapshot_stride == 25
    assert keep.cfl_2d == pytest.approx(6000.0 * 2.0e-4 * math.sqrt(2.0) / 5.0)
    assert keep.cfl_2d <= 0.45
    assert keep.lwc_qmax <= 9.6

    reduced = plan_lwc84_timestep(
        global_vmax_mps=12000.0,
        dx_m=5.0,
        dz_m=5.0,
        dt_requested_s=2.0e-4,
        output_interval_s=5.0e-3,
    )
    assert reduced.dt_used_s < 2.0e-4
    assert reduced.dt_used_s == pytest.approx(5.0e-3 / reduced.snapshot_stride)
    assert reduced.cfl_2d <= 0.45

    half_density_401_frame_plan = plan_lwc84_timestep(
        global_vmax_mps=6000.0,
        dx_m=5.0,
        dz_m=5.0,
        dt_requested_s=2.0e-4,
        output_interval_s=2.5e-3,
        snapshot_stride_requested=20,
    )
    assert half_density_401_frame_plan.snapshot_stride == 20
    assert half_density_401_frame_plan.dt_used_s == pytest.approx(1.25e-4)

    with pytest.raises(ValueError, match="positive and finite"):
        plan_lwc84_timestep(
            global_vmax_mps=float("nan"),
            dx_m=5.0,
            dz_m=5.0,
            dt_requested_s=2.0e-4,
            output_interval_s=5.0e-3,
        )


def test_free_surface_halo_uses_odd_pressure_extension() -> None:
    field = torch.arange(1, 1 + 6 * 7, dtype=torch.float64).reshape(6, 7)
    field[0] = 0.0
    padded = pad_pressure_halo(field, radius=4, side_mode="replicate")
    assert padded.shape == (14, 15)
    assert torch.equal(padded[:4, 4:-4], -torch.flip(field[1:5], dims=(0,)))
    assert torch.count_nonzero(padded[4, 4:-4]) == 0


def test_401_node_solver_grid_and_201_saved_grid_contract() -> None:
    grid = AcousticGrid(nx=401, nz=401, dx_m=5.0, dz_m=5.0, lx_m=2000.0, lz_m=2000.0, centering="node")
    boundaries = BoundaryConfig(npml=40)
    assert grid.x_m.shape == (401,)
    assert grid.x_m[[0, -1]].tolist() == [0.0, 2000.0]
    assert boundaries.extended_shape(grid) == (441, 481)

    zz, xx = np.meshgrid(grid.z_m, grid.x_m, indexing="ij")
    smooth = np.sin(2.0 * np.pi * xx / 2000.0) * np.cos(2.0 * np.pi * zz / 2000.0)
    restricted = restrict_nodal_2x(smooth.astype(np.float64))
    assert restricted.shape == (201, 201)
    assert np.isfinite(restricted).all()


def test_source_is_rediscretized_on_solver_and_saved_nodal_grids() -> None:
    kwargs = {"x0_m": 1003.25, "z0_m": 102.75, "centering": "node"}
    fine = bilinear_point_source(nx=401, nz=401, dx_m=5.0, dz_m=5.0, **kwargs)
    saved = bilinear_point_source(nx=201, nz=201, dx_m=10.0, dz_m=10.0, **kwargs)
    assert fine.source_map.shape == (401, 401)
    assert saved.source_map.shape == (201, 201)
    assert float(fine.source_map.sum()) == pytest.approx(1.0, abs=2.0e-7)
    assert float(saved.source_map.sum()) == pytest.approx(1.0, abs=2.0e-7)
    assert np.count_nonzero(saved.source_map) == 4


def test_variable_velocity_l_squared_is_composed_operator() -> None:
    dtype = torch.float64
    nz = nx = 40
    z = torch.linspace(0.0, 1.0, nz, dtype=dtype)
    x = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    pressure = torch.sin(2.0 * math.pi * xx) * torch.cos(2.0 * math.pi * zz)
    velocity = 1800.0 + 700.0 * xx + 300.0 * zz
    first = apply_l_operator(pressure, velocity, dx_m=1.0 / nx, dz_m=1.0 / nz, boundary="periodic")
    composed = apply_l_operator(first, velocity, dx_m=1.0 / nx, dz_m=1.0 / nz, boundary="periodic")
    wrong = velocity**4 * laplacian8(
        laplacian8(pressure, dx_m=1.0 / nx, dz_m=1.0 / nz, boundary="periodic"),
        dx_m=1.0 / nx,
        dz_m=1.0 / nz,
        boundary="periodic",
    )
    assert not torch.allclose(composed, wrong, rtol=1.0e-4, atol=1.0e-4)


def test_fourth_order_startup_contains_qt_and_lq_terms() -> None:
    dtype = torch.float64
    q0 = torch.zeros((24, 24), dtype=dtype)
    q0[12, 12] = 1.0
    qt0 = 2.0 * q0
    qtt0 = -3.0 * q0
    velocity = torch.full_like(q0, 2.0)
    dt = 0.01
    result = lwc84_startup(
        p0=torch.zeros_like(q0),
        pt0=torch.zeros_like(q0),
        q0=q0,
        qt0=qt0,
        qtt0=qtt0,
        velocity_mps=velocity,
        dx_m=1.0,
        dz_m=1.0,
        dt_s=dt,
        boundary="periodic",
    )
    lq = apply_l_operator(q0, velocity, dx_m=1.0, dz_m=1.0, boundary="periodic")
    expected = 0.5 * dt**2 * q0 + dt**3 * qt0 / 6.0 + dt**4 * (lq + qtt0) / 24.0
    assert torch.allclose(result, expected, rtol=0.0, atol=1.0e-15)


def test_laplacian8_has_eighth_order_periodic_spatial_convergence() -> None:
    errors: list[float] = []
    spacings: list[float] = []
    for n in (16, 20, 24, 28):
        h = 2.0 * math.pi / n
        coord = torch.arange(n, dtype=torch.float64) * h
        zz, xx = torch.meshgrid(coord, coord, indexing="ij")
        field = torch.sin(xx) + torch.cos(2.0 * zz)
        exact = -torch.sin(xx) - 4.0 * torch.cos(2.0 * zz)
        numeric = laplacian8(field, dx_m=h, dz_m=h, boundary="periodic")
        errors.append(float(torch.linalg.vector_norm(numeric - exact) / torch.linalg.vector_norm(exact)))
        spacings.append(h)
    orders = [
        math.log(errors[index] / errors[index + 1]) / math.log(spacings[index] / spacings[index + 1])
        for index in range(len(errors) - 1)
    ]
    assert min(orders) > 7.5, {"errors": errors, "orders": orders}


def test_lwc84_has_fourth_order_time_convergence_for_semidiscrete_mode() -> None:
    n = 32
    length = 2.0 * math.pi
    h = length / n
    coord = torch.arange(n, dtype=torch.float64) * h
    zz, xx = torch.meshgrid(coord, coord, indexing="ij")
    p0 = torch.sin(xx)
    velocity = torch.ones_like(p0)
    zero = torch.zeros_like(p0)
    lp0 = apply_l_operator(p0, velocity, dx_m=h, dz_m=h, boundary="periodic")
    eigenvalue = float((lp0 * p0).sum() / (p0 * p0).sum())
    omega = math.sqrt(-eigenvalue)
    t_end = 0.8
    errors: list[float] = []
    dts: list[float] = []
    for steps in (10, 20, 40, 80):
        dt = t_end / steps
        p_nm1 = p0
        p_n = lwc84_startup(
            p0=p0,
            pt0=zero,
            q0=zero,
            qt0=zero,
            qtt0=zero,
            velocity_mps=velocity,
            dx_m=h,
            dz_m=h,
            dt_s=dt,
            boundary="periodic",
        )
        for _ in range(1, steps):
            p_np1 = lwc84_step(
                p_nm1=p_nm1,
                p_n=p_n,
                q_n=zero,
                qtt_n=zero,
                velocity_mps=velocity,
                dx_m=h,
                dz_m=h,
                dt_s=dt,
                boundary="periodic",
            )
            p_nm1, p_n = p_n, p_np1
        exact = p0 * math.cos(omega * t_end)
        errors.append(float(torch.linalg.vector_norm(p_n - exact) / torch.linalg.vector_norm(exact)))
        dts.append(dt)
    orders = [math.log(errors[i] / errors[i + 1]) / math.log(dts[i] / dts[i + 1]) for i in range(3)]
    assert min(orders) > 3.7, {"errors": errors, "orders": orders}
