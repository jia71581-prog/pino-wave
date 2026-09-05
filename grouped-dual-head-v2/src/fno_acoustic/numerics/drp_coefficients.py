from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DRPCoefficients:
    """Symmetric radius-``r`` coefficients for a centered second derivative.

    The dimensionless Fourier symbol is

    ``c0 + 2 * sum(c[j] * cos(j * theta), j=1..r)``.

    Coefficients obey ``sum(j**2 * c[j]) == 1`` so that the symbol approaches
    ``-theta**2`` at the origin.  They are not advertised as eighth-order
    Taylor coefficients: the optimization deliberately trades formal order
    for smaller integrated phase error over a declared finite wavenumber band.
    """

    center: float
    positive_offsets: tuple[float, ...]
    max_nyquist_fraction: float
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        radius = len(self.positive_offsets)
        if radius < 1:
            raise ValueError("at least one positive-offset coefficient is required")
        values = (self.center, *self.positive_offsets, self.max_nyquist_fraction)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("DRP coefficients and bandwidth must be finite")
        if not 0.0 < float(self.max_nyquist_fraction) <= 1.0:
            raise ValueError("max_nyquist_fraction must lie in (0, 1]")
        zero_symbol = float(self.center) + 2.0 * sum(float(value) for value in self.positive_offsets)
        second_moment = sum(
            float(value) * float(offset * offset)
            for offset, value in enumerate(self.positive_offsets, start=1)
        )
        if abs(zero_symbol) > 2.0e-12:
            raise ValueError("DRP coefficients must annihilate constants")
        if abs(second_moment - 1.0) > 2.0e-12:
            raise ValueError("DRP coefficients must reproduce the second derivative")

    @property
    def radius(self) -> int:
        return len(self.positive_offsets)

    @property
    def coefficients(self) -> tuple[float, ...]:
        return (
            *reversed(self.positive_offsets),
            float(self.center),
            *self.positive_offsets,
        )

    @property
    def coefficient_sha256(self) -> str:
        payload = {
            "center": float(self.center).hex(),
            "positive_offsets": [float(value).hex() for value in self.positive_offsets],
            "radius": int(self.radius),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    def symbol(self, theta: np.ndarray | float) -> np.ndarray:
        angles = np.asarray(theta, dtype=np.float64)
        offsets = np.arange(1, self.radius + 1, dtype=np.float64)
        return float(self.center) + 2.0 * np.sum(
            np.cos(angles[..., None] * offsets) * np.asarray(self.positive_offsets),
            axis=-1,
        )


def _integral_cos_product(a: int, b: int, upper: float) -> float:
    if a == b:
        return 0.5 * upper + math.sin(2.0 * a * upper) / (4.0 * a)
    return 0.5 * (
        math.sin((a - b) * upper) / (a - b)
        + math.sin((a + b) * upper) / (a + b)
    )


def _integral_theta_squared_cos(frequency: int, upper: float) -> float:
    k = float(frequency)
    return (
        upper * upper * math.sin(k * upper) / k
        + 2.0 * upper * math.cos(k * upper) / (k * k)
        - 2.0 * math.sin(k * upper) / (k * k * k)
    )


@lru_cache(maxsize=32)
def optimized_drp_second_derivative_coefficients(
    *,
    radius: int,
    max_nyquist_fraction: float,
) -> DRPCoefficients:
    """Return constrained least-squares DRP coefficients.

    This minimizes the exact integral

    ``integral_0^theta_max (symbol(theta) + theta**2)**2 dtheta``

    with ``theta_max = pi * max_nyquist_fraction`` while enforcing consistency
    at zero.  The constant-annihilation constraint is applied analytically by
    expressing the symbol as ``2*sum(c_j*(cos(j*theta)-1))``.  No validation or
    test wavefield is consulted, making the construction safe to preregister.
    """

    radius = int(radius)
    fraction = float(max_nyquist_fraction)
    if radius < 1:
        raise ValueError("radius must be positive")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("max_nyquist_fraction must lie in (0, 1]")
    upper = math.pi * fraction

    # A_j(theta) = 2 * (cos(j theta) - 1), target = -theta**2.
    gram = np.empty((radius, radius), dtype=np.float64)
    rhs = np.empty(radius, dtype=np.float64)
    for row in range(radius):
        j = row + 1
        rhs[row] = -2.0 * (
            _integral_theta_squared_cos(j, upper) - upper**3 / 3.0
        )
        for column in range(radius):
            k = column + 1
            integral = (
                _integral_cos_product(j, k, upper)
                - math.sin(j * upper) / j
                - math.sin(k * upper) / k
                + upper
            )
            gram[row, column] = 4.0 * integral

    # Equality-constrained least squares: sum(j**2*c_j) = 1.
    constraint = np.square(np.arange(1, radius + 1, dtype=np.float64))
    kkt = np.zeros((radius + 1, radius + 1), dtype=np.float64)
    kkt[:radius, :radius] = gram
    kkt[:radius, radius] = constraint
    kkt[radius, :radius] = constraint
    solution = np.linalg.solve(kkt, np.concatenate((rhs, np.asarray([1.0]))))
    positive = tuple(float(value) for value in solution[:radius])
    center = -2.0 * sum(positive)
    provenance = {
        "method": "constrained_exact_integral_drp_second_derivative_v1",
        "objective": "integral_0_theta_max_(symbol_plus_theta_squared)^2",
        "constraint": "sum_j_squared_cj_equals_one",
        "max_nyquist_fraction": fraction,
        "theta_max_rad": upper,
        "formal_taylor_order_claimed": False,
    }
    return DRPCoefficients(
        center=float(center),
        positive_offsets=positive,
        max_nyquist_fraction=fraction,
        provenance=provenance,
    )
