from __future__ import annotations

import numpy as np

from .grid import AcousticGrid, OutputTimeGrid
from .source import bilinear_point_source, source_time_function


def image_source_distances(*, x_m: np.ndarray, z_m: np.ndarray, source_xy_m: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    x0, z0 = float(source_xy_m[0]), float(source_xy_m[1])
    rd = np.sqrt((np.asarray(x_m) - x0) ** 2 + (np.asarray(z_m) - z0) ** 2)
    ri = np.sqrt((np.asarray(x_m) - x0) ** 2 + (np.asarray(z_m) + z0) ** 2)
    return rd, ri


def _causal_pulse(time_s: np.ndarray, distance_m: np.ndarray, c_mps: float, f0_hz: float) -> np.ndarray:
    tau = time_s[None, None, :] - distance_m[:, :, None] / float(c_mps)
    values = source_time_function(np.maximum(tau, 0.0), float(f0_hz))
    values = np.where(tau >= 0.0, values, 0.0)
    return values / np.sqrt(np.maximum(distance_m[:, :, None], 0.5))


def analytic_halfspace_reference(
    grid: AcousticGrid,
    time: OutputTimeGrid,
    *,
    c_mps: float,
    source_xy_m: tuple[float, float],
    f0_hz: float,
) -> np.ndarray:
    src = bilinear_point_source(source_xy_m[0], source_xy_m[1], nx=grid.nx, nz=grid.nz, dx_m=grid.dx_m, dz_m=grid.dz_m)
    x, z = np.meshgrid(grid.x_m, grid.z_m, indexing="xy")
    out = np.zeros((grid.nz, grid.nx, time.nt_out), dtype=np.float64)
    for (z_idx, x_idx), weight in zip(src.indices, src.weights):
        sub_source = (float(grid.x_m[int(x_idx)]), float(grid.z_m[int(z_idx)]))
        rd, ri = image_source_distances(x_m=x, z_m=z, source_xy_m=sub_source)
        out += float(weight) * (_causal_pulse(time.t_s, rd, c_mps, f0_hz) - _causal_pulse(time.t_s, ri, c_mps, f0_hz))
    out[0, :, :] = 0.0
    return out.astype(np.float32)
