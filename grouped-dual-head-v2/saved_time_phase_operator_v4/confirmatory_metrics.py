"""Confirmatory complete-transient metrics for 401-frame saved-time fields."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from fno_acoustic.long_horizon_metrics import (
    komega_metrics,
    receiver_arrival_metrics,
    receiver_energy_metrics,
)


def propagation_windows(time_steps: int, onset_index: int) -> dict[str, slice]:
    """Return pre-onset and equal-count post-onset early/middle/late windows."""

    count = int(time_steps)
    onset = int(onset_index)
    if count < 4:
        raise ValueError("complete-transient metrics require at least four times")
    if onset < 0 or onset >= count:
        raise ValueError("onset index lies outside the time axis")
    active = count - onset
    first = onset + active // 3
    second = onset + 2 * active // 3
    first = max(first, onset + 1)
    second = max(second, first + 1)
    second = min(second, count - 1)
    return {
        "pre_onset": slice(0, onset),
        "early": slice(onset, first),
        "middle": slice(first, second),
        "late": slice(second, count),
    }


def _validate_fields(
    prediction_tzx: torch.Tensor,
    target_tzx: torch.Tensor,
    time_s: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = torch.as_tensor(prediction_tzx)
    target = torch.as_tensor(target_tzx, device=prediction.device)
    time = torch.as_tensor(time_s, device=prediction.device)
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("prediction and target must match [time,z,x]")
    if prediction.shape[0] < 4 or min(prediction.shape[1:]) < 2:
        raise ValueError("complete-transient fields are too small")
    if prediction.dtype != target.dtype or not prediction.is_floating_point():
        raise ValueError("prediction and target must share a real floating dtype")
    if torch.is_complex(prediction) or not bool(torch.isfinite(prediction).all()):
        raise ValueError("prediction must be finite and real")
    if not bool(torch.isfinite(target).all()):
        raise ValueError("target must be finite")
    if (
        time.ndim != 1
        or time.numel() != prediction.shape[0]
        or not time.is_floating_point()
        or not bool(torch.isfinite(time).all())
        or not bool(torch.all(torch.diff(time) > 0))
    ):
        raise ValueError("time_s must be finite, increasing, and match the field")
    return prediction, target, time


def relative_l2(
    prediction: torch.Tensor, target: torch.Tensor, *, eps: float = 1.0e-8
) -> float:
    """Joint relative L2 with the project-standard norm epsilon."""

    compute = torch.float64 if target.dtype == torch.float64 else torch.float32
    numerator = torch.linalg.vector_norm((prediction - target).to(compute))
    denominator = torch.linalg.vector_norm(target.to(compute)).clamp_min(float(eps))
    return float((numerator / denominator).detach().cpu())


def per_time_relative_l2(
    prediction_tzx: torch.Tensor,
    target_tzx: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-time relative error and a target-active mask."""

    difference_norm = torch.linalg.vector_norm(
        (prediction_tzx - target_tzx).float().flatten(1), dim=1
    )
    target_norm = torch.linalg.vector_norm(target_tzx.float().flatten(1), dim=1)
    activity_threshold = max(float(eps), float(target_norm.max().detach().cpu()) * 1.0e-6)
    active = target_norm > activity_threshold
    values = difference_norm / target_norm.clamp_min(float(eps))
    return values, active


def final_window_error_trend(
    prediction_tzx: torch.Tensor,
    target_tzx: torch.Tensor,
    time_s: torch.Tensor,
    late_window: slice,
    *,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Fit error versus physical time on active frames in the locked late window."""

    errors, active = per_time_relative_l2(prediction_tzx, target_tzx, eps=eps)
    late_indices = torch.arange(errors.numel(), device=errors.device)[late_window]
    late_active = active[late_window]
    selected = late_indices[late_active]
    if selected.numel() == 0:
        return {
            "late_active_frame_fraction": 0.0,
            "late_error_slope_per_s": 0.0,
            "late_max_per_time_relative_l2": 0.0,
        }
    selected_time = time_s[selected].double()
    selected_error = errors[selected].double()
    centered = selected_time - selected_time.mean()
    slope = 0.0
    if selected.numel() > 1:
        slope = float(
            ((centered * (selected_error - selected_error.mean())).sum()
             / centered.square().sum().clamp_min(float(eps))).detach().cpu()
        )
    return {
        "late_active_frame_fraction": float(late_active.float().mean().detach().cpu()),
        "late_error_slope_per_s": slope,
        "late_max_per_time_relative_l2": float(selected_error.max().detach().cpu()),
    }


def spatial_spectrum_relative_l2(
    prediction_tzx: torch.Tensor,
    target_tzx: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Return low/middle/high spatial Fourier-band errors over all times."""

    height, width = prediction_tzx.shape[-2:]
    prediction_fft = torch.fft.rfft2(prediction_tzx.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target_tzx.float(), norm="ortho")
    z_frequency = torch.fft.fftfreq(height, device=prediction_tzx.device).abs()
    x_frequency = torch.fft.rfftfreq(width, device=prediction_tzx.device).abs()
    radius = torch.sqrt(z_frequency[:, None].square() + x_frequency[None, :].square())
    radius = radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)
    masks = {
        "low": radius <= 1.0 / 3.0,
        "middle": (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
        "high": radius > 2.0 / 3.0,
    }
    result: dict[str, float] = {}
    for name, mask in masks.items():
        difference_energy = (prediction_fft[:, mask] - target_fft[:, mask]).abs().square().sum()
        target_energy = target_fft[:, mask].abs().square().sum()
        result[name] = float(
            (torch.sqrt(difference_energy) / torch.sqrt(target_energy).clamp_min(float(eps)))
            .detach()
            .cpu()
        )
    return result


def analytic_signal_on_saved_time(
    trace: torch.Tensor, time_s: torch.Tensor
) -> torch.Tensor:
    """Resample any saved-time trace to a uniform grid and apply a padded Hilbert FFT."""

    values = torch.as_tensor(trace)
    time = torch.as_tensor(time_s, device=values.device)
    if values.ndim < 1 or values.shape[-1] < 4:
        raise ValueError("trace must end in a time dimension of length at least four")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("trace must be finite and real floating")
    if (
        time.ndim != 1
        or time.numel() != values.shape[-1]
        or not time.is_floating_point()
        or not bool(torch.isfinite(time).all())
        or not bool(torch.all(torch.diff(time) > 0))
    ):
        raise ValueError("time_s must be finite, increasing, and match trace time")
    uniform_time = torch.linspace(
        time[0], time[-1], time.numel(), device=time.device, dtype=time.dtype
    )
    right = torch.searchsorted(time.contiguous(), uniform_time).clamp(1, time.numel() - 1)
    left = right - 1
    alpha = ((uniform_time - time[left]) / (time[right] - time[left])).to(values.dtype)
    uniform_trace = values[..., left] * (1.0 - alpha) + values[..., right] * alpha
    compute = torch.float64 if values.dtype == torch.float64 else torch.float32
    padding = values.shape[-1]
    padded = F.pad(uniform_trace.to(compute), (padding, padding))
    fft_length = 1 << (padded.shape[-1] - 1).bit_length()
    spectrum = torch.fft.fft(padded, n=fft_length, dim=-1)
    multiplier = torch.zeros(fft_length, device=values.device, dtype=compute)
    multiplier[0] = 1.0
    multiplier[1 : fft_length // 2] = 2.0
    multiplier[fft_length // 2] = 1.0
    analytic = torch.fft.ifft(spectrum * multiplier, dim=-1)
    return analytic[..., padding : padding + values.shape[-1]]


def receiver_lag_phase_metrics_saved_time(
    prediction_brt: torch.Tensor,
    target_brt: torch.Tensor,
    time_s: torch.Tensor,
    *,
    max_lag_fraction: float = 0.25,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Receiver lag, cross-correlation, and phase metrics for arbitrary saved times."""

    prediction = torch.as_tensor(prediction_brt)
    target = torch.as_tensor(target_brt, device=prediction.device)
    time = torch.as_tensor(time_s, device=prediction.device)
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("receiver inputs must match [batch,receiver,time]")
    if prediction.dtype != target.dtype or not prediction.is_floating_point():
        raise ValueError("receiver inputs must share a real floating dtype")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("receiver inputs must be finite")
    if (
        time.ndim != 1
        or time.numel() != prediction.shape[-1]
        or not bool(torch.all(torch.diff(time) > 0))
    ):
        raise ValueError("receiver time must be increasing and match traces")
    if not math.isfinite(float(max_lag_fraction)) or not 0 < float(max_lag_fraction) <= 0.5:
        raise ValueError("max_lag_fraction must lie in (0,0.5]")

    predicted_traces = prediction.reshape(-1, time.numel())
    target_traces = target.reshape(-1, time.numel())
    active = target_traces.abs().amax(-1) > float(eps)
    if not bool(active.any()):
        return {
            "receiver_lag_abs_s": 0.0,
            "receiver_xcorr_peak": 0.0,
            "receiver_phase_error": 0.0,
            "receiver_phase_coherence": 1.0,
        }
    uniform_time = torch.linspace(
        time[0], time[-1], time.numel(), device=time.device, dtype=time.dtype
    )
    right = torch.searchsorted(time.contiguous(), uniform_time).clamp(1, time.numel() - 1)
    left = right - 1
    alpha = ((uniform_time - time[left]) / (time[right] - time[left])).to(prediction.dtype)
    predicted_uniform = predicted_traces[:, left] * (1.0 - alpha) + predicted_traces[:, right] * alpha
    target_uniform = target_traces[:, left] * (1.0 - alpha) + target_traces[:, right] * alpha
    dt = float((uniform_time[-1] - uniform_time[0]) / (uniform_time.numel() - 1))
    maximum_lag = int(math.floor(float(max_lag_fraction) * (time.numel() - 1)))
    lags: list[float] = []
    peaks: list[float] = []
    for predicted_trace, target_trace in zip(
        predicted_uniform[active], target_uniform[active], strict=True
    ):
        denominator = (
            torch.linalg.vector_norm(predicted_trace)
            * torch.linalg.vector_norm(target_trace)
        ).clamp_min(float(eps))
        candidates: list[tuple[float, int, int, int]] = []
        for lag in range(-maximum_lag, maximum_lag + 1):
            if lag > 0:
                predicted_overlap, target_overlap = predicted_trace[lag:], target_trace[:-lag]
            elif lag < 0:
                predicted_overlap, target_overlap = predicted_trace[:lag], target_trace[-lag:]
            else:
                predicted_overlap, target_overlap = predicted_trace, target_trace
            correlation = float(torch.dot(predicted_overlap, target_overlap) / denominator)
            candidates.append((correlation, -abs(lag), -lag, lag))
        peak, _, _, lag = max(candidates)
        lags.append(abs(lag) * dt)
        peaks.append(peak)

    predicted_analytic = analytic_signal_on_saved_time(predicted_traces[active], time)
    target_analytic = analytic_signal_on_saved_time(target_traces[active], time)
    weight = target_analytic.abs()
    mask = weight > 1.0e-3 * weight.amax(-1, keepdim=True).clamp_min(float(eps))
    delta = torch.angle(predicted_analytic * target_analytic.conj()).abs()
    weight_sum = weight[mask].sum().clamp_min(float(eps))
    return {
        "receiver_lag_abs_s": sum(lags) / len(lags),
        "receiver_xcorr_peak": sum(peaks) / len(peaks),
        "receiver_phase_error": float((delta[mask] * weight[mask]).sum() / weight_sum),
        "receiver_phase_coherence": float((delta[mask].cos() * weight[mask]).sum() / weight_sum),
    }


def complete_transient_metrics(
    prediction_tzx: torch.Tensor,
    target_tzx: torch.Tensor,
    time_s: torch.Tensor,
    *,
    onset_index: int,
    receiver_z_index: int,
    receiver_x_indices: Sequence[int] | torch.Tensor,
    compute_komega: bool = True,
    temporal_modes: int = 32,
    komega_mode_block_size: int = 8,
    eps: float = 1.0e-8,
) -> dict[str, float]:
    """Compute exact-time long-horizon, receiver, and spectral diagnostics."""

    prediction, target, time = _validate_fields(prediction_tzx, target_tzx, time_s)
    z_index = int(receiver_z_index)
    if z_index < 0 or z_index >= prediction.shape[1]:
        raise ValueError("receiver z index lies outside the field")
    x_indices = torch.as_tensor(receiver_x_indices, dtype=torch.long, device=prediction.device)
    if (
        x_indices.ndim != 1
        or x_indices.numel() < 1
        or int(x_indices.min()) < 0
        or int(x_indices.max()) >= prediction.shape[2]
        or torch.unique(x_indices).numel() != x_indices.numel()
    ):
        raise ValueError("receiver x indices must be unique and inside the field")

    windows = propagation_windows(prediction.shape[0], onset_index)
    result: dict[str, float] = {
        "record_relative_l2": relative_l2(prediction, target, eps=eps),
        "prediction_finite": float(torch.isfinite(prediction).all()),
        "prediction_nonzero": float(torch.count_nonzero(prediction) > 0),
    }
    for name, window in windows.items():
        if window.stop is not None and window.stop <= (window.start or 0):
            continue
        result[f"{name}_relative_l2"] = relative_l2(
            prediction[window], target[window], eps=eps
        )
    result.update(final_window_error_trend(prediction, target, time, windows["late"], eps=eps))
    for name, value in spatial_spectrum_relative_l2(prediction, target, eps=eps).items():
        result[f"spectrum_{name}_relative_l2"] = value

    receiver_prediction = prediction[:, z_index, x_indices].transpose(0, 1).unsqueeze(0)
    receiver_target = target[:, z_index, x_indices].transpose(0, 1).unsqueeze(0)
    result["receiver_relative_l2"] = relative_l2(
        receiver_prediction, receiver_target, eps=eps
    )
    result.update(receiver_arrival_metrics(receiver_prediction, receiver_target, time))
    result.update(
        receiver_lag_phase_metrics_saved_time(receiver_prediction, receiver_target, time)
    )
    result.update(receiver_energy_metrics(receiver_prediction, receiver_target, time))

    if compute_komega:
        field_prediction = prediction.permute(1, 2, 0).unsqueeze(0)
        field_target = target.permute(1, 2, 0).unsqueeze(0)
        values = komega_metrics(
            field_prediction,
            field_target,
            time,
            temporal_modes=int(temporal_modes),
            mode_block_size=int(komega_mode_block_size),
        )
        result.update(values)
        late = windows["late"]
        late_values = komega_metrics(
            field_prediction[..., late],
            field_target[..., late],
            time[late],
            temporal_modes=min(int(temporal_modes), time[late].numel()),
            mode_block_size=int(komega_mode_block_size),
        )
        result["komega_relative_l2_late"] = late_values["komega_relative_l2"]
        result["komega_high_late"] = late_values["komega_high"]

    if not all(math.isfinite(float(value)) for value in result.values()):
        raise RuntimeError("confirmatory metric produced a non-finite value")
    return result


__all__ = [
    "analytic_signal_on_saved_time",
    "complete_transient_metrics",
    "final_window_error_trend",
    "per_time_relative_l2",
    "propagation_windows",
    "relative_l2",
    "receiver_lag_phase_metrics_saved_time",
    "spatial_spectrum_relative_l2",
]
