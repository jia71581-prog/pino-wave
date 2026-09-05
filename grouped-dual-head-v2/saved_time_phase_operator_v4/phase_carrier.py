"""Analytic travel-time phase carriers for complex temporal coefficients."""
from __future__ import annotations

import math

import torch


def travel_phase_carrier(
    travel_time_s: torch.Tensor,
    frequency_hz: torch.Tensor,
) -> torch.Tensor:
    """Return ``[B,2,Z,X]`` real/imaginary channels for exp(-i omega tau)."""

    travel = torch.as_tensor(travel_time_s)
    frequency = torch.as_tensor(frequency_hz, device=travel.device, dtype=travel.dtype)
    if travel.ndim != 3 or frequency.ndim != 1 or travel.shape[0] != frequency.shape[0]:
        raise ValueError("travel phase expects [B,Z,X] travel and [B] frequency")
    phase = 2.0 * math.pi * frequency[:, None, None] * travel
    return torch.stack((torch.cos(phase), -torch.sin(phase)), dim=1)


def rotate_complex_pairs(
    value: torch.Tensor,
    carrier: torch.Tensor,
) -> torch.Tensor:
    """Multiply interleaved complex channel pairs by one spatial carrier."""

    field = torch.as_tensor(value)
    phase = torch.as_tensor(carrier, device=field.device, dtype=field.dtype)
    if field.ndim != 4 or field.shape[1] % 2:
        raise ValueError("complex-pair field must be [B,2K,Z,X]")
    if phase.shape != (field.shape[0], 2, field.shape[-2], field.shape[-1]):
        raise ValueError("carrier shape does not match complex-pair field")
    pairs = field.reshape(field.shape[0], field.shape[1] // 2, 2, *field.shape[-2:])
    real, imag = pairs[:, :, 0], pairs[:, :, 1]
    carrier_real, carrier_imag = phase[:, 0:1], phase[:, 1:2]
    rotated_real = real * carrier_real - imag * carrier_imag
    rotated_imag = real * carrier_imag + imag * carrier_real
    return torch.stack((rotated_real, rotated_imag), dim=2).flatten(1, 2)


__all__ = ["rotate_complex_pairs", "travel_phase_carrier"]
