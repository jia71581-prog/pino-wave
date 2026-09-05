"""Analytic finite-difference dispersion symbols and numerical phase velocity.

Three schemes share one interface so the manuscript can contrast them on a single
phase-velocity-vs-angle / vs-points-per-wavelength figure:

* ``fd2``   -- 2nd-order in space (3-point Laplacian), 2nd-order leapfrog in time.
* ``fd4``   -- 4th-order in space (5-point), 2nd-order leapfrog in time.
* ``lwc84`` -- 8th-order in space (9-point), 4th-order Lax--Wendroff-corrected time.

The numerical phase velocity follows the modified-equation dispersion relation of the
explicit update.  For a plane wave the leapfrog step gives
``cos(omega_num * dt) = 1 - q/2`` with ``q = -C^2 * S(theta)``, where ``C`` is the axis
Courant number ``c*dt/h``, ``S`` is the (dimensionless) spatial-stencil symbol summed over
both axes, and the LWC-84 4th-order time correction adds ``+ q^2 / 24``.  The ratio of the
numerical to the exact temporal phase ``omega_num*dt / (C * k*h)`` is 1 in the continuum
limit and departs from 1 as the grid coarsens -- this departure *is* the numerical
dispersion the paper quantifies.
"""
from __future__ import annotations

import math

import numpy as np


# (second-derivative coefficients, integer offsets, time order)
STENCILS: dict[str, tuple[np.ndarray, np.ndarray, int]] = {
    "fd2": (
        np.asarray((1.0, -2.0, 1.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 1.0), dtype=np.float64),
        2,
    ),
    "fd4": (
        np.asarray((-1.0 / 12.0, 4.0 / 3.0, -2.5, 4.0 / 3.0, -1.0 / 12.0), dtype=np.float64),
        np.arange(-2.0, 3.0, dtype=np.float64),
        2,
    ),
    "lwc84": (
        np.asarray(
            (
                -1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
                -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0,
            ),
            dtype=np.float64,
        ),
        np.arange(-4.0, 5.0, dtype=np.float64),
        4,
    ),
}


def spatial_symbol(scheme: str, theta: float) -> float:
    """Dimensionless second-derivative symbol S(theta) = sum_n c_n cos(theta*n).

    Approaches ``-theta**2`` in the continuum (small-theta) limit for every scheme.
    """
    coefficients, offsets, _ = STENCILS[str(scheme)]
    return float(np.sum(coefficients * np.cos(float(theta) * offsets)))


def phase_velocity_ratio(
    *,
    scheme: str,
    points_per_wavelength: float,
    angle_deg: float,
    courant_axis: float,
) -> float:
    """Numerical/exact temporal phase ratio for a plane wave.

    ``points_per_wavelength`` is the shortest-wavelength sampling ``lambda / h``,
    ``angle_deg`` the propagation angle relative to the x axis, and ``courant_axis`` the
    per-axis Courant number ``c*dt/h``.  Returns 1.0 in the continuum limit; values below
    1.0 indicate a lagging (slow) numerical wave, the usual grid-dispersion signature.
    """
    ppw = float(points_per_wavelength)
    if ppw <= 2.0 or float(courant_axis) <= 0.0:
        raise ValueError("points per wavelength must exceed 2 and Courant number be positive")
    angle = math.radians(float(angle_deg))
    kh = 2.0 * math.pi / ppw
    symbol = spatial_symbol(scheme, kh * math.cos(angle)) + spatial_symbol(
        scheme, kh * math.sin(angle)
    )
    q = -float(courant_axis) ** 2 * symbol
    time_order = STENCILS[str(scheme)][2]
    cosine = 1.0 - 0.5 * q
    if time_order == 4:
        cosine += q**2 / 24.0
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("unstable dispersion sample: increase points per wavelength or reduce Courant")
    numeric_omega_dt = math.acos(cosine)
    exact_omega_dt = float(courant_axis) * kh
    return numeric_omega_dt / exact_omega_dt
