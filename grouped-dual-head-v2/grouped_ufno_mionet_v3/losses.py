"""Phase-sensitive supervised objectives for V3 point and full-field heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.nn import functional as F


def _as_float(value) -> torch.Tensor:
    return torch.as_tensor(value).float()


def per_frame_relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-8,
    energy_floor_fraction: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _as_float(prediction)
    target = _as_float(target)
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("frame prediction and target shapes must match [record,time,...]")
    if not 0.0 <= energy_floor_fraction <= 1.0:
        raise ValueError("energy_floor_fraction must be in [0,1]")
    error = (prediction - target).flatten(start_dim=2).norm(dim=-1)
    target_norm = target.flatten(start_dim=2).norm(dim=-1)
    record_floor = target_norm.amax(dim=1, keepdim=True) * float(energy_floor_fraction)
    denominator = torch.maximum(target_norm, record_floor).clamp_min(eps)
    per_frame = error / denominator
    return per_frame.mean(), per_frame


def _per_frame_spectrum(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError("spectral losses require [record,time,z,x]")
    return torch.fft.rfft2(value.float(), norm="ortho")


def complex_spectrum_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-8,
    energy_floor_fraction: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= energy_floor_fraction <= 1.0:
        raise ValueError("energy_floor_fraction must be in [0,1]")
    prediction_fft = _per_frame_spectrum(prediction)
    target_fft = _per_frame_spectrum(target)
    error = (prediction_fft - target_fft).abs().flatten(start_dim=2).norm(dim=-1)
    target_norm = target_fft.abs().flatten(start_dim=2).norm(dim=-1)
    record_floor = target_norm.amax(dim=1, keepdim=True) * float(energy_floor_fraction)
    denominator = torch.maximum(target_norm, record_floor).clamp_min(eps)
    return (error / denominator).mean()


def spectral_phase_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_fraction: float = 1.0e-4,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, int]:
    if not 0.0 < energy_fraction < 1.0:
        raise ValueError("phase energy fraction must be in (0,1)")
    prediction_fft = _per_frame_spectrum(prediction)
    target_fft = _per_frame_spectrum(target)
    target_amplitude = target_fft.abs()
    threshold = target_amplitude.amax(dim=(-2, -1), keepdim=True) * energy_fraction
    mask = target_amplitude > threshold.clamp_min(eps)
    count = int(mask.sum().item())
    if count == 0:
        raise RuntimeError("empty phase mask: target has no energetic Fourier modes")
    prediction_unit = prediction_fft / prediction_fft.abs().clamp_min(eps)
    target_unit = target_fft / target_amplitude.clamp_min(eps)
    phase_distance = (prediction_unit - target_unit).abs().square()
    return phase_distance[mask].mean(), count


def spatial_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    prediction = _as_float(prediction)
    target = _as_float(target)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("spatial-gradient tensors must match [record,time,z,x]")
    prediction_grad = torch.cat(
        (
            prediction[..., 1:, :] - prediction[..., :-1, :],
            (prediction[..., :, 1:] - prediction[..., :, :-1]).transpose(-2, -1),
        ),
        dim=-1,
    ) if prediction.shape[-2] == prediction.shape[-1] else None
    if prediction_grad is not None:
        target_grad = torch.cat(
            (
                target[..., 1:, :] - target[..., :-1, :],
                (target[..., :, 1:] - target[..., :, :-1]).transpose(-2, -1),
            ),
            dim=-1,
        )
        return (prediction_grad - target_grad).norm() / target_grad.norm().clamp_min(eps)
    prediction_dz = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dz = target[..., 1:, :] - target[..., :-1, :]
    prediction_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    numerator = (prediction_dz - target_dz).square().sum() + (prediction_dx - target_dx).square().sum()
    denominator = target_dz.square().sum() + target_dx.square().sum()
    return torch.sqrt(numerator / denominator.clamp_min(eps))


def time_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    prediction = _as_float(prediction)
    target = _as_float(target)
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] < 2:
        raise ValueError("time-difference tensors require matching [record,time>=2,z,x]")
    prediction_dt = prediction[:, 1:] - prediction[:, :-1]
    target_dt = target[:, 1:] - target[:, :-1]
    return (prediction_dt - target_dt).norm() / target_dt.norm().clamp_min(eps)


@dataclass(frozen=True)
class QueryPointLoss:
    total: torch.Tensor
    huber: torch.Tensor
    relative_l2: torch.Tensor


def query_point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_probability: torch.Tensor,
    *,
    beta: float = 1.0,
    eps: float = 1.0e-8,
) -> QueryPointLoss:
    prediction = _as_float(prediction)
    target = _as_float(target)
    probability = _as_float(sample_probability)
    if prediction.shape != target.shape or probability.shape != target.shape:
        raise ValueError("query prediction, target, and probability shapes must match")
    if torch.any(probability <= 0) or not torch.isfinite(probability).all():
        raise ValueError("query sampling probabilities must be finite and positive")
    pointwise = F.smooth_l1_loss(prediction, target, reduction="none", beta=beta)
    inverse = probability.reciprocal()
    huber = (pointwise * inverse).sum() / inverse.sum()
    relative = (prediction - target).norm(dim=-1) / target.norm(dim=-1).clamp_min(eps)
    relative_l2 = relative.mean()
    return QueryPointLoss(total=huber + relative_l2, huber=huber, relative_l2=relative_l2)


def relative_head_consistency_loss(
    dense_at_query: torch.Tensor,
    prediction_query: torch.Tensor,
    target_query: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    dense = _as_float(dense_at_query)
    prediction = _as_float(prediction_query)
    target = _as_float(target_query)
    if dense.shape != prediction.shape or prediction.shape != target.shape or dense.ndim < 2:
        raise ValueError("dual-head consistency tensors must have matching record shapes")
    error = (dense - prediction).flatten(start_dim=1).norm(dim=-1)
    scale = target.flatten(start_dim=1).norm(dim=-1).clamp_min(eps)
    return (error / scale).mean()


@dataclass(frozen=True)
class V3LossWeights:
    point: float = 1.0
    frame: float = 1.0
    complex_spectrum: float = 0.15
    spectral_phase: float = 0.1
    spatial_gradient: float = 0.1
    time_difference: float = 0.1
    consistency: float = 0.05


@dataclass(frozen=True)
class V3LossResult:
    total: torch.Tensor
    unweighted: Mapping[str, torch.Tensor]
    weighted: Mapping[str, torch.Tensor]
    per_frame_relative_l2: torch.Tensor
    phase_mask_count: int


def compute_v3_losses(
    *,
    prediction_query: torch.Tensor,
    target_query: torch.Tensor,
    query_probability: torch.Tensor,
    prediction_dense: torch.Tensor,
    target_dense: torch.Tensor,
    dense_at_query: torch.Tensor,
    weights: V3LossWeights,
    phase_energy_fraction: float = 1.0e-4,
    relative_energy_floor_fraction: float = 0.0,
) -> V3LossResult:
    point = query_point_loss(prediction_query, target_query, query_probability).total
    frame, per_frame = per_frame_relative_l2(
        prediction_dense,
        target_dense,
        energy_floor_fraction=relative_energy_floor_fraction,
    )
    complex_value = complex_spectrum_loss(
        prediction_dense,
        target_dense,
        energy_floor_fraction=relative_energy_floor_fraction,
    )
    phase, phase_count = spectral_phase_loss(
        prediction_dense,
        target_dense,
        energy_fraction=phase_energy_fraction,
    )
    spatial = spatial_gradient_loss(prediction_dense, target_dense)
    temporal = time_difference_loss(prediction_dense, target_dense)
    consistency = relative_head_consistency_loss(
        dense_at_query, prediction_query, target_query
    )
    unweighted = {
        "point": point,
        "frame": frame,
        "complex_spectrum": complex_value,
        "spectral_phase": phase,
        "spatial_gradient": spatial,
        "time_difference": temporal,
        "consistency": consistency,
    }
    weighted = {
        name: value * float(getattr(weights, name)) for name, value in unweighted.items()
    }
    total = torch.stack(tuple(weighted.values())).sum()
    if not torch.isfinite(total):
        raise RuntimeError("nonfinite V3 composite loss")
    return V3LossResult(
        total=total,
        unweighted=unweighted,
        weighted=weighted,
        per_frame_relative_l2=per_frame,
        phase_mask_count=phase_count,
    )


__all__ = [
    "QueryPointLoss",
    "V3LossResult",
    "V3LossWeights",
    "complex_spectrum_loss",
    "compute_v3_losses",
    "per_frame_relative_l2",
    "query_point_loss",
    "relative_head_consistency_loss",
    "spatial_gradient_loss",
    "spectral_phase_loss",
    "time_difference_loss",
]
