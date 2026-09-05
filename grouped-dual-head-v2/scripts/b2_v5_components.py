"""Leakage-safe data and loss components for the B2-v5 pilot."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

TEMPORAL_BANDS = ((0, 5), (5, 12), (12, 29))


def causal_window_start(
    time_s: np.ndarray,
    *,
    source_t0_s: float,
    source_f0_hz: float,
    k_frames: int,
    lead_cycles: float = 0.5,
) -> int:
    """Choose a source-parameter-only window start.

    A Ricker source is already energetic roughly half a dominant cycle before
    ``t0``.  This rule closely matches the historical 5%-of-future-max onset
    windows while requiring no wavefield or future-derived statistic.
    """
    axis = np.asarray(time_s, dtype=np.float64)
    if axis.ndim != 1 or axis.size < k_frames:
        raise ValueError("time_s must be one-dimensional with at least k_frames")
    if not np.all(np.diff(axis) > 0.0):
        raise ValueError("time_s must be strictly increasing")
    if source_f0_hz <= 0.0 or lead_cycles < 0.0:
        raise ValueError("source_f0_hz must be positive and lead_cycles nonnegative")
    start_time = float(source_t0_s) - float(lead_cycles) / float(source_f0_hz)
    tolerance = 1.0e-12 * max(1.0, abs(start_time))
    start = int(np.searchsorted(axis, start_time - tolerance, side="left"))
    return max(0, min(start, int(axis.size - k_frames)))


def append_source_conditioning(
    base_cond: np.ndarray,
    source_f0_hz,
    source_t0_s,
) -> np.ndarray:
    """Append normalized constant f0 and t0 maps to [B,C,Z,X] conditioning."""
    value = np.asarray(base_cond, dtype=np.float32)
    squeeze = False
    if value.ndim == 3:
        value = value[None]
        squeeze = True
    if value.ndim != 4:
        raise ValueError("base_cond must be [C,Z,X] or [B,C,Z,X]")
    batch, _, height, width = value.shape
    f0 = np.asarray(source_f0_hz, dtype=np.float64).reshape(-1)
    t0 = np.asarray(source_t0_s, dtype=np.float64).reshape(-1)
    if f0.size == 1:
        f0 = np.repeat(f0, batch)
    if t0.size == 1:
        t0 = np.repeat(t0, batch)
    if f0.size != batch or t0.size != batch:
        raise ValueError("source parameters must have one value per batch record")
    f0_norm = np.clip((f0 - 20.0) / 10.0, -2.0, 2.0)
    t0_norm = np.clip((t0 - 0.10) / 0.05, -2.0, 2.0)
    f0_map = np.broadcast_to(f0_norm[:, None, None, None], (batch, 1, height, width))
    t0_map = np.broadcast_to(t0_norm[:, None, None, None], (batch, 1, height, width))
    result = np.concatenate((value, f0_map, t0_map), axis=1).astype(np.float32)
    return result[0] if squeeze else result


def per_record_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same batched shape")
    dims = tuple(range(1, prediction.ndim))
    numerator = (prediction - target).square().sum(dims).clamp_min(0.0).sqrt()
    denominator = target.square().sum(dims).clamp_min(1.0e-16).sqrt()
    return numerator / denominator


def temporal_band_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor_fraction: float = 0.005,
) -> torch.Tensor:
    """Return [B,3] low/mid/high temporal-rFFT relative errors."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must be [B,T,C,Z,X]")
    if not 0.0 < energy_floor_fraction < 1.0:
        raise ValueError("energy_floor_fraction must lie in (0,1)")
    pred_fft = torch.fft.rfft(prediction.float(), dim=1)
    target_fft = torch.fft.rfft(target.float(), dim=1)
    dims = (1, 2, 3, 4)
    total_energy = target_fft.abs().square().sum(dims).clamp_min(1.0e-16)
    values = []
    for start, stop in TEMPORAL_BANDS:
        band_target = target_fft[:, start:stop]
        band_error = pred_fft[:, start:stop] - band_target
        numerator = band_error.abs().square().sum(dims).clamp_min(0.0).sqrt()
        band_energy = band_target.abs().square().sum(dims)
        denominator = torch.maximum(
            band_energy, energy_floor_fraction * total_energy
        ).sqrt()
        values.append(numerator / denominator)
    return torch.stack(values, dim=1)


def b2_v5_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
    *,
    arm: str,
    nonworse_weight: float = 0.3,
    spectral_weight: float = 0.1,
    spectral_energy_floor_fraction: float = 0.005,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-variable loss arms sharing the same base relative-L2 objective."""
    if arm not in {"control", "source_cond", "physics_cond", "nonworse_hinge", "spectral"}:
        raise ValueError(f"unknown B2-v5 arm: {arm}")
    candidate = per_record_relative_l2(prediction, target)
    base = candidate.mean()
    hinge = torch.zeros((), device=prediction.device, dtype=base.dtype)
    spectral = torch.zeros((), device=prediction.device, dtype=base.dtype)
    loss = base
    if arm == "nonworse_hinge":
        anchor_error = per_record_relative_l2(anchor, target).detach()
        hinge = F.relu(candidate - anchor_error).mean()
        loss = loss + float(nonworse_weight) * hinge
    elif arm == "spectral":
        spectral = temporal_band_relative_l2(
            prediction,
            target,
            energy_floor_fraction=spectral_energy_floor_fraction,
        ).mean()
        loss = loss + float(spectral_weight) * spectral
    return loss, {
        "relative_l2": base.detach(),
        "nonworse_hinge": hinge.detach(),
        "spectral": spectral.detach(),
    }
