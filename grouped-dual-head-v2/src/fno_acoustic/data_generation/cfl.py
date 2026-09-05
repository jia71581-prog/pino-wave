from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .grid import AcousticGrid, OutputTimeGrid


@dataclass(frozen=True)
class LWC84TimePlan:
    global_vmax_mps: float
    dt_requested_s: float
    dt_used_s: float
    output_interval_s: float
    snapshot_stride: int
    cfl_2d: float
    lwc_qmax: float
    engineering_dt_limit_s: float
    lwc_safety_dt_limit_s: float
    reduced: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def plan_lwc84_timestep(
    *,
    global_vmax_mps: float,
    dx_m: float,
    dz_m: float,
    dt_requested_s: float,
    output_interval_s: float,
    snapshot_stride_requested: int | None = None,
    engineering_cfl_limit: float = 0.45,
    lwc_q_safety_limit: float = 9.6,
) -> LWC84TimePlan:
    values = (global_vmax_mps, dx_m, dz_m, dt_requested_s, output_interval_s)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("velocity, spacing, and time values must be positive and finite")
    vmax = float(global_vmax_mps)
    inv_h2 = 1.0 / float(dx_m) ** 2 + 1.0 / float(dz_m) ** 2
    engineering_limit = float(engineering_cfl_limit) / (vmax * math.sqrt(inv_h2))
    lwc_limit = math.sqrt(float(lwc_q_safety_limit) / (vmax**2 * (2048.0 / 315.0) * inv_h2))
    dt_cap = min(float(dt_requested_s), engineering_limit, lwc_limit)
    minimum_stride = max(1, int(math.ceil(float(output_interval_s) / dt_cap - 1.0e-12)))
    if snapshot_stride_requested is not None and int(snapshot_stride_requested) <= 0:
        raise ValueError("snapshot_stride_requested must be positive")
    stride = max(minimum_stride, int(snapshot_stride_requested or 1))
    dt_used = float(output_interval_s) / stride
    cfl_2d = vmax * dt_used * math.sqrt(inv_h2)
    lwc_qmax = dt_used**2 * vmax**2 * (2048.0 / 315.0) * inv_h2
    if dt_used > float(dt_requested_s) * (1.0 + 1.0e-12):
        raise ValueError("aligned dt_used would exceed dt_requested")
    if cfl_2d > float(engineering_cfl_limit) * (1.0 + 1.0e-12):
        raise ValueError("engineering CFL limit was not satisfied")
    if lwc_qmax > float(lwc_q_safety_limit) * (1.0 + 1.0e-12):
        raise ValueError("LWC spectral safety limit was not satisfied")
    return LWC84TimePlan(
        global_vmax_mps=vmax,
        dt_requested_s=float(dt_requested_s),
        dt_used_s=dt_used,
        output_interval_s=float(output_interval_s),
        snapshot_stride=stride,
        cfl_2d=cfl_2d,
        lwc_qmax=lwc_qmax,
        engineering_dt_limit_s=engineering_limit,
        lwc_safety_dt_limit_s=lwc_limit,
        reduced=bool(dt_used < float(dt_requested_s) * (1.0 - 1.0e-12)),
    )


@dataclass(frozen=True)
class CFLPlan:
    c_global_max_mps: float
    dt_internal_5m_s: float
    n_substeps_5m: int
    cfl_axis_5m: float
    dt_symbol_limit_5m_s: float
    dt_internal_fine_s: float
    n_substeps_fine: int
    cfl_axis_fine: float
    dt_symbol_limit_fine_s: float
    fine_dx_m: float = 2.5
    fine_dz_m: float = 2.5
    fine_npml: int = 80

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _symbol_dt_limit(cmax: float, dx: float, dz: float) -> float:
    from fno_acoustic.numerics.drp_coefficients import optimized_drp_second_derivative_coefficients

    coeffs = optimized_drp_second_derivative_coefficients(radius=4, max_nyquist_fraction=0.65)
    theta = np.linspace(0.0, math.pi, 4097, dtype=np.float64)
    min_symbol = float(np.min(coeffs.symbol(theta)))
    spectral_radius = abs(min_symbol) * (1.0 / float(dx) ** 2 + 1.0 / float(dz) ** 2)
    return float(2.0 / (float(cmax) * math.sqrt(spectral_radius)))


def _tier(cmax: float, dx: float, dz: float, time: OutputTimeGrid, cfl_axis_safety: float) -> tuple[float, int, float, float]:
    dt_axis = float(cfl_axis_safety) * min(float(dx), float(dz)) / float(cmax)
    n_sub = int(math.ceil(float(time.dt_out_s) / dt_axis))
    dt = float(time.dt_out_s) / n_sub
    cfl = float(cmax) * dt / min(float(dx), float(dz))
    return dt, n_sub, cfl, _symbol_dt_limit(cmax, dx, dz)


def compute_cfl_plan(
    *,
    c_global_max_mps: float,
    grid: AcousticGrid,
    time: OutputTimeGrid,
    cfl_axis_safety: float = 0.30,
    symbol_limit_fraction: float = 0.80,
    fine_dx_m: float = 2.5,
    fine_dz_m: float = 2.5,
) -> CFLPlan:
    cmax = float(c_global_max_mps)
    dt5, n5, cfl5, symbol5 = _tier(cmax, grid.dx_m, grid.dz_m, time, cfl_axis_safety)
    dtf, nf, cflf, symbolf = _tier(cmax, fine_dx_m, fine_dz_m, time, cfl_axis_safety)
    if cfl5 > cfl_axis_safety + 1.0e-12 or cflf > cfl_axis_safety + 1.0e-12:
        raise ValueError("CFL axis safety was not satisfied")
    if dt5 > float(symbol_limit_fraction) * symbol5 or dtf > float(symbol_limit_fraction) * symbolf:
        raise ValueError("symbol stability limit was not satisfied")
    return CFLPlan(
        c_global_max_mps=cmax,
        dt_internal_5m_s=dt5,
        n_substeps_5m=n5,
        cfl_axis_5m=cfl5,
        dt_symbol_limit_5m_s=symbol5,
        dt_internal_fine_s=dtf,
        n_substeps_fine=nf,
        cfl_axis_fine=cflf,
        dt_symbol_limit_fine_s=symbolf,
        fine_dx_m=float(fine_dx_m),
        fine_dz_m=float(fine_dz_m),
        fine_npml=80,
    )
