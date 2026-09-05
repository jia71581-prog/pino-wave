from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FORMULA = "-9.6*f0*(0.6*f0*t-1)*exp(-8*(0.6*f0*t-1)^2)*delta(x-x0,z-z0)"
LATEX_FORMULA = (
    r"-9.6f_0(0.6f_0t-1)\exp[-8(0.6f_0t-1)^2]\delta(x-x_0,z-z_0)"
)


@dataclass(frozen=True)
class BilinearPointSource:
    indices: np.ndarray
    weights: np.ndarray
    source_map: np.ndarray
    delta_h: np.ndarray
    x0_m: float
    z0_m: float


def source_formula_string() -> str:
    return FORMULA


def source_formula_latex() -> str:
    return LATEX_FORMULA


def source_time_function(t_s, f0_hz):
    t = np.asarray(t_s, dtype=np.float64)
    f0 = np.asarray(f0_hz, dtype=np.float64)
    a = 0.6 * f0 * t - 1.0
    return -9.6 * f0 * a * np.exp(-8.0 * a**2)


def _bracketing_index(coord_m: float, *, n: int, d_m: float, centering: str) -> tuple[int, int, float]:
    if centering not in {"cell", "node"}:
        raise ValueError("centering must be 'cell' or 'node'")
    offset = 0.5 if centering == "cell" else 0.0
    coordinates = (np.arange(n, dtype=np.float64) + offset) * float(d_m)
    if not (coordinates[0] <= coord_m <= coordinates[-1]):
        raise ValueError(
            f"source coordinate {coord_m} is outside {centering} range "
            f"[{coordinates[0]}, {coordinates[-1]}]"
        )
    upper = int(np.searchsorted(coordinates, coord_m, side="right"))
    upper = min(max(upper, 1), n - 1)
    lower = upper - 1
    frac = float((coord_m - coordinates[lower]) / (coordinates[upper] - coordinates[lower]))
    return lower, upper, frac


def bilinear_point_source(
    x0_m: float,
    z0_m: float,
    *,
    nx: int,
    nz: int,
    dx_m: float,
    dz_m: float,
    centering: str = "cell",
) -> BilinearPointSource:
    x0 = float(x0_m)
    z0 = float(z0_m)
    ix0, ix1, fx = _bracketing_index(x0, n=int(nx), d_m=float(dx_m), centering=centering)
    iz0, iz1, fz = _bracketing_index(z0, n=int(nz), d_m=float(dz_m), centering=centering)

    indices = np.asarray([[iz0, ix0], [iz0, ix1], [iz1, ix0], [iz1, ix1]], dtype=np.int32)
    weights = np.asarray(
        [(1.0 - fz) * (1.0 - fx), (1.0 - fz) * fx, fz * (1.0 - fx), fz * fx],
        dtype=np.float64,
    )
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    source_map = np.zeros((int(nz), int(nx)), dtype=np.float32)
    delta_h = np.zeros_like(source_map, dtype=np.float64)
    for (z_idx, x_idx), weight in zip(indices, weights):
        source_map[int(z_idx), int(x_idx)] += np.float32(weight)
        delta_h[int(z_idx), int(x_idx)] += float(weight) / (float(dx_m) * float(dz_m))
    return BilinearPointSource(
        indices=indices,
        weights=weights,
        source_map=source_map,
        delta_h=delta_h,
        x0_m=x0,
        z0_m=z0,
    )
