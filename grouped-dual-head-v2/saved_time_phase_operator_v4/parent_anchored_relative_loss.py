"""Metric-aligned relative-energy loss for parent-anchored block training.

The deployment model is unchanged.  This module is used only during offline
train-only optimization.  A sampled multi-block window is inverse-probability
weighted so its expected squared error is the complete future-trajectory error.
All denominators use complete-record energies, preventing near-zero local wave
windows from dominating the gradient.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def unbiased_window_weights(
    *,
    loss_start: int,
    sample_length: int,
    time_count: int,
    block_size: int,
    rollout_blocks: int,
    first_future: int = 2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Horvitz-Thompson frame and temporal-difference weights.

    Training chooses one block-aligned start uniformly. Interior frames appear
    in two-block windows more often than endpoint frames. Multiplying each
    sampled contribution by ``number_of_starts / inclusion_count`` makes the
    mean over starts exactly equal to the complete future sum.
    """

    start = int(loss_start)
    length = int(sample_length)
    total = int(time_count)
    block = int(block_size)
    horizon_blocks = int(rollout_blocks)
    first = int(first_future)
    if min(length, total, block, horizon_blocks) <= 0 or not 0 <= first < total:
        raise ValueError("invalid window-weight dimensions")
    if start < first or start + length > total:
        raise ValueError("sampled window lies outside the registered future")
    starts = tuple(range(first, total, block))
    frame_counts = torch.zeros(total - first, dtype=torch.int64)
    delta_counts = torch.zeros(max(total - first - 1, 0), dtype=torch.int64)
    for candidate_start in starts:
        candidate_stop = min(candidate_start + horizon_blocks * block, total)
        frame_counts[candidate_start - first : candidate_stop - first] += 1
        if candidate_stop - candidate_start >= 2:
            delta_counts[candidate_start - first : candidate_stop - first - 1] += 1
    local = slice(start - first, start - first + length)
    selected_frame_counts = frame_counts[local]
    if bool((selected_frame_counts <= 0).any()):
        raise RuntimeError("sampled frame has zero inclusion probability")
    frame_weights = float(len(starts)) / selected_frame_counts.to(torch.float64)
    if length == 1:
        delta_weights = torch.empty(0, dtype=torch.float64)
    else:
        delta_local = slice(start - first, start - first + length - 1)
        selected_delta_counts = delta_counts[delta_local]
        if bool((selected_delta_counts <= 0).any()):
            raise RuntimeError("sampled temporal difference has zero inclusion probability")
        delta_weights = float(len(starts)) / selected_delta_counts.to(torch.float64)
    return (
        frame_weights.to(device=device, dtype=dtype),
        delta_weights.to(device=device, dtype=dtype),
    )


def _batch_energy(
    value: torch.Tensor | float,
    *,
    batch: int,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    energy = torch.as_tensor(value, device=reference.device, dtype=torch.float32)
    if energy.ndim == 0:
        energy = energy.expand(batch)
    if energy.shape != (batch,):
        raise ValueError(f"{name} must be scalar or [batch]")
    if not bool(torch.isfinite(energy).all()) or bool((energy <= 0.0).any()):
        raise ValueError(f"{name} must contain finite positive values")
    return energy


def record_energy_squared_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parent: torch.Tensor,
    *,
    frame_weights: torch.Tensor,
    delta_weights: torch.Tensor,
    full_target_energy: torch.Tensor | float,
    full_target_delta_energy: torch.Tensor | float,
    derivative_weight: float = 0.01,
    spectral_weight: float = 0.02,
    nonworse_weight: float = 0.30,
    correction_weight: float = 0.01,
    spectral_floor_fraction: float = 0.005,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unbiased complete-record squared-relative-error training objective."""

    if prediction.shape != target.shape or prediction.shape != parent.shape:
        raise ValueError("prediction, target, and parent must have identical shapes")
    if prediction.ndim != 4 or prediction.shape[1] < 2:
        raise ValueError("relative-energy loss expects [B,T,Z,X], T>=2")
    if frame_weights.shape != (prediction.shape[1],):
        raise ValueError("frame_weights must have one entry per sampled frame")
    if delta_weights.shape != (prediction.shape[1] - 1,):
        raise ValueError("delta_weights must have one entry per sampled difference")
    if min(
        derivative_weight,
        spectral_weight,
        nonworse_weight,
        correction_weight,
        spectral_floor_fraction,
    ) < 0.0:
        raise ValueError("loss weights must be nonnegative")
    if not bool(torch.isfinite(frame_weights).all()) or bool((frame_weights <= 0).any()):
        raise ValueError("frame weights must be finite and positive")
    if not bool(torch.isfinite(delta_weights).all()) or bool((delta_weights <= 0).any()):
        raise ValueError("delta weights must be finite and positive")

    batch = int(prediction.shape[0])
    total_energy = _batch_energy(
        full_target_energy,
        batch=batch,
        reference=prediction,
        name="full_target_energy",
    )
    total_delta_energy = _batch_energy(
        full_target_delta_energy,
        batch=batch,
        reference=prediction,
        name="full_target_delta_energy",
    )
    frame = frame_weights.float()[None, :, None, None]
    delta_weight = delta_weights.float()[None, :, None, None]
    error = prediction.float() - target.float()
    parent_error = parent.float() - target.float()
    correction = prediction.float() - parent.float()

    main_rows = (error.square() * frame).sum((1, 2, 3)) / total_energy
    parent_rows = (parent_error.square() * frame).sum((1, 2, 3)) / total_energy
    correction_rows = (correction.square() * frame).sum((1, 2, 3)) / total_energy

    error_delta = error[:, 1:] - error[:, :-1]
    derivative_rows = (
        error_delta.square() * delta_weight
    ).sum((1, 2, 3)) / total_delta_energy

    # Frequency balancing remains a small auxiliary term. Applying sqrt(frame)
    # before the FFT preserves the inverse-probability energy accounting.
    weighted_error = error * frame.sqrt()
    weighted_target = target.float() * frame.sqrt()
    error_spectrum = torch.fft.rfft(weighted_error, dim=1, norm="ortho")
    target_spectrum = torch.fft.rfft(weighted_target, dim=1, norm="ortho")
    frequency_count = int(error_spectrum.shape[1])
    edges = (
        0,
        max(1, frequency_count // 4),
        max(2, frequency_count // 2),
        frequency_count,
    )
    one_sided = torch.full(
        (frequency_count,), 2.0, device=prediction.device, dtype=torch.float32
    )
    one_sided[0] = 1.0
    if prediction.shape[1] % 2 == 0:
        one_sided[-1] = 1.0
    spectral_terms = []
    floor = float(spectral_floor_fraction) * total_energy
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low:
            continue
        weight = one_sided[low:high][None, :, None, None]
        numerator = (error_spectrum[:, low:high].abs().square() * weight).sum((1, 2, 3))
        band_energy = (target_spectrum[:, low:high].abs().square() * weight).sum((1, 2, 3))
        spectral_terms.append(numerator / torch.maximum(band_energy, floor))
    spectral_rows = torch.stack(spectral_terms, dim=1).mean(dim=1)

    main = main_rows.mean()
    derivative = derivative_rows.mean()
    spectral = spectral_rows.mean()
    nonworse = F.relu(main_rows - parent_rows.detach()).mean()
    correction_penalty = correction_rows.mean()
    loss = (
        main
        + float(derivative_weight) * derivative
        + float(spectral_weight) * spectral
        + float(nonworse_weight) * nonworse
        + float(correction_weight) * correction_penalty
    )
    return loss, {
        "record_relative_l2_squared": main.detach(),
        "record_relative_l2_estimate": main.detach().clamp_min(0.0).sqrt(),
        "derivative_relative_l2_squared": derivative.detach(),
        "spectral_relative_l2_squared": spectral.detach(),
        "nonworse_squared_hinge": nonworse.detach(),
        "correction_energy_ratio": correction_penalty.detach(),
        "parent_relative_l2_squared": parent_rows.mean().detach(),
    }


__all__ = ["record_energy_squared_loss", "unbiased_window_weights"]
