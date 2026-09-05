"""Stored-frame-only, energy-aware V4 validation metrics."""
from __future__ import annotations

from collections.abc import Sequence

import torch


TIME_BIN_NAMES = ("pre_onset", "early", "middle", "late")


def _spectrum_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    height, width = target.shape[-2:]
    z_frequency = torch.fft.fftfreq(height, device=target.device).abs()
    x_frequency = torch.fft.rfftfreq(width, device=target.device).abs()
    radius = torch.sqrt(z_frequency[:, None].square() + x_frequency[None, :].square())
    radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)
    bands = {
        "low": radius <= 1.0 / 3.0,
        "middle": (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
        "high": radius > 2.0 / 3.0,
    }
    result: dict[str, float] = {}
    for name, mask in bands.items():
        difference = (prediction_fft - target_fft)[..., mask]
        reference = target_fft[..., mask]
        numerator = torch.linalg.vector_norm(difference.reshape(-1))
        denominator = torch.linalg.vector_norm(reference.reshape(-1)).clamp_min(1.0e-8)
        result[name] = float(numerator / denominator)
    return result


def exact_wavefield_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    families: Sequence[str],
    energy_floor_fraction: float = 0.01,
) -> dict[str, object]:
    """Measure four exact stored frames without interpolated-time aggregation."""

    predicted = torch.as_tensor(prediction)
    reference = torch.as_tensor(target, device=predicted.device)
    if predicted.shape != reference.shape or predicted.ndim != 4:
        raise ValueError("prediction and target must have matching [record,time,z,x] shapes")
    records, times = predicted.shape[:2]
    if times != len(TIME_BIN_NAMES):
        raise ValueError("exact V4 metrics require four stratified stored frames")
    labels = tuple(str(value) for value in families)
    if len(labels) != records or not labels:
        raise ValueError("family labels must contain one value per record")
    if not 0.0 < energy_floor_fraction <= 1.0:
        raise ValueError("energy floor fraction must lie in (0,1]")

    error = (predicted.float() - reference.float()).flatten(start_dim=2)
    flat_target = reference.float().flatten(start_dim=2)
    error_norm = torch.linalg.vector_norm(error, dim=-1)
    target_norm = torch.linalg.vector_norm(flat_target, dim=-1)
    record_peak = target_norm.amax(dim=1, keepdim=True)
    floor = float(energy_floor_fraction) * record_peak
    denominator = torch.maximum(target_norm, floor).clamp_min(1.0e-8)
    floored_relative = error_norm / denominator
    unfloored_relative = error_norm / target_norm.clamp_min(1.0e-8)
    near_zero = target_norm <= floor

    record_error_norm = torch.linalg.vector_norm(
        (predicted.float() - reference.float()).flatten(start_dim=1), dim=-1
    )
    record_target_norm = torch.linalg.vector_norm(
        reference.float().flatten(start_dim=1), dim=-1
    ).clamp_min(1.0e-8)
    record_relative = record_error_norm / record_target_norm

    family_metrics: dict[str, float] = {}
    for family in dict.fromkeys(labels):
        mask = torch.tensor(
            [value == family for value in labels], dtype=torch.bool, device=predicted.device
        )
        family_metrics[family] = float(record_relative[mask].mean())

    phase_numerator = (predicted.float() * reference.float()).flatten(start_dim=2).sum(dim=-1)
    prediction_norm = predicted.float().flatten(start_dim=2).norm(dim=-1)
    phase_denominator = (prediction_norm * target_norm).clamp_min(1.0e-8)
    informative = target_norm > floor
    phase_values = phase_numerator / phase_denominator
    phase_correlation = (
        float(phase_values[informative].mean()) if bool(informative.any()) else 1.0
    )

    return {
        "record_count": records,
        "frame_count": records * times,
        "near_zero_frame_count": int(near_zero.sum()),
        "aggregate_floored_relative_l2": float(record_relative.mean()),
        "aggregate_unfloored_relative_l2": float(unfloored_relative.mean()),
        "family_floored_relative_l2": family_metrics,
        "time_bin_floored_relative_l2": {
            name: float(floored_relative[:, index].mean())
            for index, name in enumerate(TIME_BIN_NAMES)
        },
        "rmse": float(torch.sqrt(torch.mean((predicted.float() - reference.float()).square()))),
        "phase_correlation": phase_correlation,
        "spectrum_relative_l2": _spectrum_metrics(predicted, reference),
    }


__all__ = ["TIME_BIN_NAMES", "exact_wavefield_metrics"]
