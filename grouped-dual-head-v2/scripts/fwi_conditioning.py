"""Testable conditioning primitives for staged acoustic FWI updates.

The illumination map is a static ray-density approximation.  It is useful for
limiting gross source/receiver-coverage imbalance, but is not a substitute for
the diagonal of the wave-equation Hessian.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as torch_f


@dataclass(frozen=True)
class GuardDecision:
    """Result of deciding whether an ASVGD estimate may replace L2."""

    selected: str
    l2_nmse: float
    candidate_nmse: float
    allowed_nmse: float


def parse_stage_schedule(text: str, *, stages: int, option: str) -> list[float]:
    """Parse one scalar or exactly one scalar per frequency-continuation stage."""
    if int(stages) < 1:
        raise ValueError("stages must be positive")
    values = [float(item) for item in text.replace(",", " ").split()]
    if not values:
        return []
    if len(values) == 1:
        return values * int(stages)
    if len(values) != int(stages):
        raise ValueError(f"{option} must provide either one value or {int(stages)} values")
    return values


def gaussian_smooth_2d(field: torch.Tensor, *, sigma_cells: float) -> torch.Tensor:
    """Apply normalized Gaussian smoothing to a two-dimensional gradient field."""
    if field.ndim != 2:
        raise ValueError("gaussian_smooth_2d expects a two-dimensional tensor")
    if float(sigma_cells) <= 0.0:
        return field
    radius = max(1, int(math.ceil(3.0 * float(sigma_cells))))
    coordinates = torch.arange(-radius, radius + 1, device=field.device, dtype=field.dtype)
    kernel_1d = torch.exp(-0.5 * (coordinates / float(sigma_cells)).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, 2 * radius + 1, 2 * radius + 1)
    padded = torch_f.pad(field[None, None], (radius, radius, radius, radius), mode="replicate")
    return torch_f.conv2d(padded, kernel)[0, 0]


def ray_illumination(
    nz: int,
    nx: int,
    *,
    source_ij: np.ndarray,
    receiver_ij: np.ndarray,
) -> torch.Tensor:
    """Accumulate unit density along straight source--receiver ray approximations."""
    if int(nz) < 1 or int(nx) < 1:
        raise ValueError("ray illumination requires positive grid dimensions")
    sources = np.asarray(source_ij, dtype=np.int64).reshape(-1, 2)
    receivers = np.asarray(receiver_ij, dtype=np.int64).reshape(-1, 2)
    coverage = np.zeros((int(nz), int(nx)), dtype=np.float32)
    for source in sources:
        for receiver in receivers:
            samples = int(max(abs(int(receiver[0]) - int(source[0])), abs(int(receiver[1]) - int(source[1])))) + 1
            rows = np.rint(np.linspace(int(source[0]), int(receiver[0]), samples)).astype(np.int64)
            columns = np.rint(np.linspace(int(source[1]), int(receiver[1]), samples)).astype(np.int64)
            rows = np.clip(rows, 0, int(nz) - 1)
            columns = np.clip(columns, 0, int(nx) - 1)
            np.add.at(coverage, (rows, columns), 1.0)
    return torch.from_numpy(coverage)


def illumination_gain(illumination: torch.Tensor, *, max_gain: float) -> torch.Tensor:
    """Return a bounded inverse-square-root gain from a ray-density map."""
    if float(max_gain) < 1.0:
        raise ValueError("max_gain must be at least one")
    positive = illumination[illumination > 0]
    if positive.numel() == 0:
        return torch.ones_like(illumination)
    reference = positive.median().clamp_min(torch.finfo(illumination.dtype).eps)
    stabilized = illumination.clamp_min(0.25 * reference)
    gain = torch.sqrt(reference / stabilized)
    return gain.clamp(1.0 / float(max_gain), float(max_gain))


def huber_tv_slowness_squared(
    velocity: torch.Tensor,
    *,
    reference_velocity_mps: float,
    delta: float,
) -> torch.Tensor:
    """Edge-preserving Huber-TV penalty on dimensionless squared slowness."""
    if velocity.ndim != 2:
        raise ValueError("huber_tv_slowness_squared expects a two-dimensional velocity tensor")
    if float(reference_velocity_mps) <= 0.0:
        raise ValueError("reference_velocity_mps must be positive")
    if float(delta) <= 0.0:
        raise ValueError("delta must be positive")
    model = (float(reference_velocity_mps) / velocity.clamp_min(torch.finfo(velocity.dtype).eps)).square()

    def huber(value: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(value.square() + float(delta) ** 2) - float(delta)

    return huber(model[1:, :] - model[:-1, :]).mean() + huber(model[:, 1:] - model[:, :-1]).mean()


def select_l2_guard(*, l2_nmse: float, candidate_nmse: float, tolerance: float) -> GuardDecision:
    """Select ASVGD only if its receiver NMSE is within the L2 tolerance."""
    if float(l2_nmse) < 0.0 or float(candidate_nmse) < 0.0:
        raise ValueError("receiver NMSE values must be non-negative")
    if float(tolerance) < 0.0:
        raise ValueError("tolerance must be non-negative")
    allowed_nmse = float(l2_nmse) * (1.0 + float(tolerance))
    selected = "asvgd" if math.isfinite(float(candidate_nmse)) and float(candidate_nmse) <= allowed_nmse else "l2"
    return GuardDecision(
        selected=selected,
        l2_nmse=float(l2_nmse),
        candidate_nmse=float(candidate_nmse),
        allowed_nmse=allowed_nmse,
    )
