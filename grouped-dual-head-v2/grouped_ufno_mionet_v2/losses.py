"""Physically coherent losses for structured V2 wave data."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _as_fp32(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).float()


def per_record_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = _as_fp32(prediction), _as_fp32(target)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("relative L2 requires equal tensors with a batch axis")
    error = (prediction - target).flatten(1).square().sum(1).sqrt()
    energy = target.flatten(1).square().sum(1).sqrt()
    active_epsilon = torch.finfo(target.dtype).eps * target.flatten(1).abs().amax(1).clamp_min(1.0)
    return (error / energy.clamp_min(active_epsilon)).mean()


def normalized_huber(prediction: torch.Tensor, target: torch.Tensor, beta: float = 0.1,
                     weight: torch.Tensor | None = None) -> torch.Tensor:
    raw = F.smooth_l1_loss(_as_fp32(prediction), _as_fp32(target), beta=beta, reduction="none")
    if weight is None:
        return raw.mean()
    weight = _as_fp32(weight)
    while weight.ndim < raw.ndim:
        weight = weight.unsqueeze(-1)
    return (raw * weight).mean()


def inverse_probability_weights(probability: torch.Tensor) -> torch.Tensor:
    probability = _as_fp32(probability)
    if probability.ndim != 2 or torch.any(probability <= 0) or not torch.isfinite(probability).all():
        raise ValueError("sample probability must be finite positive [batch,query]")
    inverse = probability.reciprocal()
    return inverse / inverse.mean(dim=1, keepdim=True)


def spatial_gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = _as_fp32(prediction), _as_fp32(target)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("spatial gradient loss requires structured [batch,time,z,x] frames")
    pred_z, target_z = prediction[:, :, 1:] - prediction[:, :, :-1], target[:, :, 1:] - target[:, :, :-1]
    pred_x, target_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1], target[:, :, :, 1:] - target[:, :, :, :-1]
    return normalized_huber(pred_z, target_z) + normalized_huber(pred_x, target_x)


def spatial_fft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = _as_fp32(prediction), _as_fp32(target)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("spatial FFT requires structured [batch,time,z,x] frames")
    pred_fft = torch.fft.rfft2(prediction, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm="ortho")
    return F.smooth_l1_loss(torch.view_as_real(pred_fft), torch.view_as_real(target_fft), beta=0.1)


def trace_fft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction, target = _as_fp32(prediction), _as_fp32(target)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("trace FFT requires structured [batch,receiver,time] traces")
    pred_fft = torch.fft.rfft(prediction, dim=-1, norm="ortho")
    target_fft = torch.fft.rfft(target, dim=-1, norm="ortho")
    return F.smooth_l1_loss(torch.view_as_real(pred_fft), torch.view_as_real(target_fft), beta=0.1)


def consistency_loss(query_prediction: torch.Tensor, dense_prediction_at_queries: torch.Tensor,
                     target: torch.Tensor | None = None) -> torch.Tensor:
    query_prediction = _as_fp32(query_prediction)
    dense_prediction_at_queries = _as_fp32(dense_prediction_at_queries)
    if query_prediction.shape != dense_prediction_at_queries.shape:
        raise ValueError("consistency tensors must have equal shapes")
    agreement = normalized_huber(query_prediction, dense_prediction_at_queries)
    if target is None:
        return agreement
    target = _as_fp32(target)
    return agreement + normalized_huber(query_prediction, target) + normalized_huber(dense_prediction_at_queries, target)


@dataclass(frozen=True)
class DualHeadLosses:
    total: torch.Tensor
    dense_data: torch.Tensor
    trace_data: torch.Tensor
    gradient: torch.Tensor
    spatial_spectrum: torch.Tensor
    trace_spectrum: torch.Tensor
    dense_relative_l2: torch.Tensor
    trace_relative_l2: torch.Tensor


def dual_head_losses(dense_prediction: torch.Tensor, dense_target: torch.Tensor,
                     trace_prediction: torch.Tensor, trace_target: torch.Tensor,
                     *, gradient_weight: float = 0.1, spatial_fft_weight: float = 0.05,
                     trace_fft_weight: float = 0.05) -> DualHeadLosses:
    dense_relative = per_record_relative_l2(dense_prediction, dense_target)
    trace_relative = per_record_relative_l2(trace_prediction, trace_target)
    dense_data = normalized_huber(dense_prediction, dense_target) + dense_relative
    trace_data = normalized_huber(trace_prediction, trace_target) + trace_relative
    gradient = spatial_gradient_loss(dense_prediction, dense_target)
    spatial_spectrum = spatial_fft_loss(dense_prediction, dense_target)
    trace_spectrum = trace_fft_loss(trace_prediction, trace_target)
    total = (dense_data + trace_data + gradient_weight * gradient
             + spatial_fft_weight * spatial_spectrum + trace_fft_weight * trace_spectrum)
    return DualHeadLosses(
        total, dense_data, trace_data, gradient, spatial_spectrum, trace_spectrum,
        dense_relative, trace_relative,
    )


def query_data_loss(prediction: torch.Tensor, target: torch.Tensor,
                    sample_probability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = inverse_probability_weights(sample_probability)
    relative = per_record_relative_l2(prediction, target)
    return normalized_huber(prediction, target, weight=weight) + relative, relative
