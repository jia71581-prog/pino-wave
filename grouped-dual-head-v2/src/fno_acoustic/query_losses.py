"""Losses for full-time traces evaluated at sampled spatial query sites."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F

from fno_acoustic.temporal_operator import trapezoid_weights


@dataclass(frozen=True)
class HHFieldLoss:
    loss: torch.Tensor
    mse: torch.Tensor
    relative_l2: torch.Tensor
    relative_kind: str = "consistent_ratio_surrogate"


@dataclass(frozen=True)
class AuxiliaryQueryLosses:
    receiver: torch.Tensor
    phase: torch.Tensor
    local_spectrum: torch.Tensor
    energy: torch.Tensor


def _positive_finite_eps(eps: float) -> None:
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError("eps must be a positive finite number")
    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError("eps must be a positive finite number")


def _validate_matching_float_tensors(
    first: torch.Tensor, second: torch.Tensor, *, name: str, ndim: int
) -> None:
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise ValueError(f"{name} must be tensors")
    if first.ndim != ndim or second.ndim != ndim or first.shape != second.shape:
        raise ValueError(f"{name} must have matching shape")
    if not first.is_floating_point() or not second.is_floating_point():
        raise ValueError(f"{name} must have a real floating dtype")
    if first.dtype != second.dtype or first.device != second.device:
        raise ValueError(f"{name} must have matching dtype and device")
    if not bool(torch.isfinite(first).all()) or not bool(torch.isfinite(second).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_full_traces(
    prediction: torch.Tensor, target: torch.Tensor, *, name: str
) -> None:
    _validate_matching_float_tensors(prediction, target, name=name, ndim=3)
    if prediction.shape[0] < 1 or prediction.shape[1] < 1:
        raise ValueError(f"{name} must have nonempty batch and site dimensions")
    if prediction.shape[-1] != 160:
        raise ValueError(f"{name} require exactly 160 saved times")


def normalized_spatial_frequency_radius(
    height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    """Return the spatial frequency radius normalized by each axis' Nyquist."""

    if isinstance(height, bool) or isinstance(width, bool):
        raise ValueError("height and width must be positive integers")
    if not isinstance(height, int) or not isinstance(width, int) or height < 1 or width < 1:
        raise ValueError("height and width must be positive integers")
    kx = torch.fft.fftfreq(height, device=device) / 0.5
    kz = torch.fft.rfftfreq(width, device=device) / 0.5
    return torch.sqrt(kx[:, None].square() + kz[None, :].square())


def hansen_hurwitz_field_loss(
    prediction_bqt: torch.Tensor,
    target_bqt: torch.Tensor,
    draw_probability_bq: torch.Tensor,
    population_size: int,
    late_time_weights: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> HHFieldLoss:
    """Estimate full-field MSE with Hansen--Hurwitz spatial reweighting.

    The relative value is the ratio of two Hansen--Hurwitz squared-total
    estimates.  It is a consistent ratio surrogate, not an unbiased estimator.
    """

    _validate_full_traces(prediction_bqt, target_bqt, name="prediction and target")
    _positive_finite_eps(eps)
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size < 1
    ):
        raise ValueError("population_size must be a positive integer")
    if (
        not isinstance(draw_probability_bq, torch.Tensor)
        or draw_probability_bq.shape != prediction_bqt.shape[:2]
    ):
        raise ValueError("draw probabilities must have shape [B,Q]")
    if not draw_probability_bq.is_floating_point():
        raise ValueError("draw probabilities must have a floating dtype")
    compute_dtype = torch.float64 if prediction_bqt.dtype == torch.float64 else torch.float32
    q = draw_probability_bq.to(device=prediction_bqt.device, dtype=compute_dtype)
    if not bool(torch.isfinite(q).all()) or bool(torch.any(q <= 0)):
        raise ValueError("draw probabilities must be positive and finite after conversion")
    if bool(torch.any(q > 1)):
        raise ValueError("draw probabilities must not exceed one")

    time_weight = torch.ones(160, device=prediction_bqt.device, dtype=compute_dtype)
    if late_time_weights is not None:
        if (
            not isinstance(late_time_weights, torch.Tensor)
            or late_time_weights.shape != (160,)
            or not late_time_weights.is_floating_point()
        ):
            raise ValueError(
                "late_time_weights must have shape [160] with positive finite values"
            )
        converted_weights = late_time_weights.to(
            device=prediction_bqt.device, dtype=compute_dtype
        )
        if not bool(torch.isfinite(converted_weights).all()) or bool(
            torch.any(converted_weights <= 0)
        ):
            raise ValueError(
                "late_time_weights must have shape [160] with positive finite values"
            )
        time_weight = converted_weights

    inverse = (1.0 / (float(population_size) * q)).unsqueeze(-1)
    squared_weight = inverse * time_weight
    if not bool(torch.isfinite(inverse).all()) or not bool(torch.isfinite(squared_weight).all()):
        raise ValueError("draw probabilities produce nonfinite Hansen-Hurwitz weights")
    prediction = prediction_bqt.to(compute_dtype)
    target = target_bqt.to(compute_dtype)
    difference = prediction - target
    error_total = (difference.square() * squared_weight).mean(dim=1).sum(dim=-1)
    target_total = (target.square() * squared_weight).mean(dim=1).sum(dim=-1)
    mse = (error_total / time_weight.sum()).mean()

    # vector_norm has a finite zero subgradient, unlike an explicit sqrt at zero.
    root_weight = torch.sqrt(squared_weight / prediction_bqt.shape[1])
    error_norm = torch.linalg.vector_norm(difference * root_weight, dim=(1, 2))
    relative = error_norm / torch.sqrt(target_total).clamp_min(eps)
    relative_l2 = relative.mean()
    return HHFieldLoss(mse + relative_l2, mse, relative_l2)


def standardized_hh_field_loss(
    prediction_hat_bqt: torch.Tensor,
    target_hat_bqt: torch.Tensor,
    draw_probability_bq: torch.Tensor,
    population_size: int,
    normalization: Any,
    late_time_weights: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> HHFieldLoss:
    """Combine standardized HH MSE with physical-unit HH relative L2."""

    normalized = hansen_hurwitz_field_loss(
        prediction_hat_bqt,
        target_hat_bqt,
        draw_probability_bq,
        population_size,
        late_time_weights=late_time_weights,
        eps=eps,
    )
    physical = hansen_hurwitz_field_loss(
        normalization.decode_wavefield(prediction_hat_bqt),
        normalization.decode_wavefield(target_hat_bqt),
        draw_probability_bq,
        population_size,
        late_time_weights=late_time_weights,
        eps=eps,
    )
    return HHFieldLoss(
        normalized.mse + physical.relative_l2,
        normalized.mse,
        physical.relative_l2,
    )


def unweighted_field_loss(
    prediction_bqt: torch.Tensor, target_bqt: torch.Tensor, eps: float = 1.0e-8
) -> HHFieldLoss:
    """Return ordinary MSE plus relative L2 for complete query traces."""

    _validate_full_traces(prediction_bqt, target_bqt, name="prediction and target")
    _positive_finite_eps(eps)
    compute_dtype = torch.float64 if prediction_bqt.dtype == torch.float64 else torch.float32
    prediction = prediction_bqt.to(compute_dtype)
    target = target_bqt.to(compute_dtype)
    difference = prediction - target
    mse = difference.square().mean()
    relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
        target
    ).clamp_min(eps)
    return HHFieldLoss(mse + relative, mse, relative)


def _validate_time_s(time_s: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(time_s, torch.Tensor)
        or time_s.ndim != 1
        or time_s.shape != (160,)
    ):
        raise ValueError("time_s must have shape [160]")
    if not time_s.is_floating_point():
        raise ValueError("time_s must have a real floating dtype")
    if not bool(torch.isfinite(time_s).all()):
        raise ValueError("time_s must contain only finite values")
    if not bool(torch.all(torch.diff(time_s) > 0)):
        raise ValueError("time_s must be strictly increasing")
    return time_s.to(device=reference.device)


def analytic_signal_on_physical_time(
    trace: torch.Tensor, time_s: torch.Tensor
) -> torch.Tensor:
    """Linearly resample onto a uniform physical grid, then apply Hilbert FFT."""

    if not isinstance(trace, torch.Tensor) or trace.ndim < 1 or trace.shape[-1] != 160:
        raise ValueError("trace must end in a 160-sample time dimension")
    if not trace.is_floating_point() or not bool(torch.isfinite(trace).all()):
        raise ValueError("trace must be finite and have a real floating dtype")
    time = _validate_time_s(time_s, trace)
    uniform_time = torch.linspace(
        time[0], time[-1], time.numel(), device=trace.device, dtype=time.dtype
    )
    right = torch.searchsorted(time.contiguous(), uniform_time).clamp(1, time.numel() - 1)
    left = right - 1
    alpha = ((uniform_time - time[left]) / (time[right] - time[left])).to(trace.dtype)
    uniform_trace = trace[..., left] * (1.0 - alpha) + trace[..., right] * alpha

    fft_dtype = torch.float64 if trace.dtype == torch.float64 else torch.float32
    # Symmetric zero padding turns the FFT Hilbert transform into a close
    # approximation of the linear (non-periodic) transform at both endpoints.
    padding = trace.shape[-1]
    padded = F.pad(uniform_trace.to(fft_dtype), (padding, padding))
    fft_length = 1 << (padded.shape[-1] - 1).bit_length()
    spectrum = torch.fft.fft(padded, n=fft_length, dim=-1)
    multiplier = torch.zeros(fft_length, device=trace.device, dtype=fft_dtype)
    multiplier[0] = 1.0
    multiplier[1 : fft_length // 2] = 2.0
    multiplier[fft_length // 2] = 1.0
    analytic_padded = torch.fft.ifft(spectrum * multiplier, dim=-1)
    return analytic_padded[..., padding : padding + trace.shape[-1]]


def auxiliary_query_losses(
    receiver_prediction: torch.Tensor,
    receiver_target: torch.Tensor,
    patch_prediction: torch.Tensor,
    patch_target: torch.Tensor,
    time_s: torch.Tensor,
    core_margin: int = 1,
    eps: float = 1.0e-8,
) -> AuxiliaryQueryLosses:
    """Compute deterministic receiver, phase, high-k patch, and energy losses."""

    _validate_full_traces(
        receiver_prediction, receiver_target, name="receiver prediction and target"
    )
    _validate_matching_float_tensors(
        patch_prediction, patch_target, name="patch prediction and target", ndim=5
    )
    if patch_prediction.shape[0] != receiver_prediction.shape[0]:
        raise ValueError("receiver and patch batch dimensions must match")
    if patch_prediction.shape[1] < 1 or patch_prediction.shape[-1] != 160:
        raise ValueError("patch tensors must have shape [B,P,H,W,160] with P > 0")
    if patch_prediction.device != receiver_prediction.device:
        raise ValueError("receiver and patch tensors must use the same device")
    if isinstance(core_margin, bool) or not isinstance(core_margin, int):
        raise ValueError("core_margin must be a nonnegative integer")
    if core_margin < 0 or 2 * core_margin >= min(patch_prediction.shape[2:4]):
        raise ValueError("core_margin must leave a nonempty patch core")
    _positive_finite_eps(eps)
    time = _validate_time_s(time_s, receiver_prediction)

    compute_dtype = (
        torch.float64 if receiver_prediction.dtype == torch.float64 else torch.float32
    )
    receiver_prediction_compute = receiver_prediction.to(compute_dtype)
    receiver_target_compute = receiver_target.to(compute_dtype)
    receiver_difference = receiver_prediction_compute - receiver_target_compute
    receiver = torch.linalg.vector_norm(receiver_difference) / torch.linalg.vector_norm(
        receiver_target_compute
    ).clamp_min(eps)

    if torch.equal(receiver_prediction, receiver_target):
        phase = receiver_prediction.sum() * 0.0
    else:
        prediction_analytic = analytic_signal_on_physical_time(receiver_prediction, time)
        target_analytic = analytic_signal_on_physical_time(receiver_target, time)
        prediction_magnitude = prediction_analytic.abs()
        target_magnitude = target_analytic.abs()
        prediction_scale = prediction_magnitude.amax(dim=-1, keepdim=True)
        target_scale = target_magnitude.amax(dim=-1, keepdim=True)
        prediction_scale_safe = torch.where(
            prediction_scale > 0, prediction_scale, torch.ones_like(prediction_scale)
        )
        target_scale_safe = torch.where(
            target_scale > 0, target_scale, torch.ones_like(target_scale)
        )
        prediction_normalized = prediction_analytic / prediction_scale_safe
        target_normalized = target_analytic / target_scale_safe
        prediction_envelope = prediction_magnitude / prediction_scale_safe
        target_envelope = target_magnitude / target_scale_safe
        prediction_unit = prediction_normalized / prediction_envelope.clamp_min(eps)
        target_unit = target_normalized / target_envelope.clamp_min(eps)
        coherence = (prediction_unit * target_unit.conj()).real
        coherence = coherence.clamp(-1.0, 1.0)
        phase = ((1.0 - coherence) * target_envelope).sum() / target_envelope.sum().clamp_min(
            eps
        )

    if core_margin:
        core_prediction = patch_prediction[
            :, :, core_margin:-core_margin, core_margin:-core_margin, :
        ]
        core_target = patch_target[
            :, :, core_margin:-core_margin, core_margin:-core_margin, :
        ]
    else:
        core_prediction = patch_prediction
        core_target = patch_target
    fft_dtype = torch.float64 if core_prediction.dtype == torch.float64 else torch.float32
    prediction_k = torch.fft.rfft2(core_prediction.to(fft_dtype), dim=(2, 3))
    target_k = torch.fft.rfft2(core_target.to(fft_dtype), dim=(2, 3))
    radius = normalized_spatial_frequency_radius(
        core_prediction.shape[2], core_prediction.shape[3], prediction_k.device
    )
    high_k = radius >= 0.5
    if bool(high_k.any()):
        spectral_error = (prediction_k - target_k).abs().square()[:, :, high_k, :].mean()
        target_spectral_energy = target_k.abs().square()[:, :, high_k, :].mean()
        if bool(target_spectral_energy <= eps**2):
            local_spectrum = spectral_error
        else:
            local_spectrum = spectral_error / target_spectral_energy
    else:
        local_spectrum = prediction_k.real.sum() * 0.0

    weights = trapezoid_weights(time).to(
        device=receiver_prediction.device, dtype=compute_dtype
    )
    prediction_energy = (receiver_prediction_compute.square() * weights).sum(dim=-1)
    target_energy = (receiver_target_compute.square() * weights).sum(dim=-1)
    relative_energy = torch.log(
        (prediction_energy + eps**2) / (target_energy + eps**2)
    ).abs()
    energy_per_trace = torch.where(
        target_energy <= eps**2, prediction_energy, relative_energy
    )
    if torch.equal(receiver_prediction, receiver_target):
        energy = receiver_prediction_compute.sum() * 0.0
    else:
        energy = energy_per_trace.mean()
    return AuxiliaryQueryLosses(receiver, phase, local_spectrum, energy)
