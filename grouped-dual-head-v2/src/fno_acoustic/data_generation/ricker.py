from __future__ import annotations

import math

import numpy as np
import torch


def _exp(value):
    return torch.exp(value) if isinstance(value, torch.Tensor) else np.exp(value)


def ricker_triplet(t_s, *, f0_hz: float, t0_s: float | None = None):
    """Return the Ricker wavelet and its first two analytic derivatives."""

    f0 = float(f0_hz)
    if not math.isfinite(f0) or f0 <= 0.0:
        raise ValueError("f0_hz must be positive and finite")
    t0 = 1.5 / f0 if t0_s is None else float(t0_s)
    if not math.isfinite(t0) or t0 < 0.0:
        raise ValueError("t0_s must be finite and nonnegative")
    if isinstance(t_s, torch.Tensor):
        time = t_s
    else:
        time = np.asarray(t_s, dtype=np.float64)
    a = math.pi**2 * f0**2
    tau = time - t0
    exponential = _exp(-a * tau**2)
    wavelet = (1.0 - 2.0 * a * tau**2) * exponential
    first = (-6.0 * a * tau + 4.0 * a**2 * tau**3) * exponential
    second = (-6.0 * a + 24.0 * a**2 * tau**2 - 8.0 * a**3 * tau**4) * exponential
    return wavelet, first, second
