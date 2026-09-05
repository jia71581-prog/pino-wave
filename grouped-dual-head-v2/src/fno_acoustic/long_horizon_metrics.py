"""Exact full160 long-horizon metrics for native acoustic wavefields."""

from __future__ import annotations

from dataclasses import dataclass
import math
import string

import torch

from .query_losses import (
    analytic_signal_on_physical_time,
    normalized_spatial_frequency_radius,
)
from .temporal_operator import trapezoid_weights


QUARTER_SLICES = {
    "q1": slice(0, 40),
    "q2": slice(40, 80),
    "q3": slice(80, 120),
    "q4": slice(120, 160),
}


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in string.hexdigits for character in value)
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


@dataclass(frozen=True)
class ReceiverSiteManifest:
    """Receiver sites bound to a canonical row-major native grid."""

    grid_size: int | tuple[int, int]
    site_indices: torch.Tensor
    physical_xz: torch.Tensor
    sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.grid_size, tuple):
            if len(self.grid_size) != 2:
                raise ValueError("manifest grid tuple must contain height and width")
            height = _positive_integer(self.grid_size[0], "manifest height")
            width = _positive_integer(self.grid_size[1], "manifest width")
        else:
            height = width = _positive_integer(self.grid_size, "manifest grid_size")
        integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        indices = self.site_indices
        if (
            not isinstance(indices, torch.Tensor)
            or indices.ndim != 1
            or indices.dtype not in integer_dtypes
            or indices.numel() < 1
        ):
            raise ValueError("receiver site_indices must be a nonempty integer vector")
        indices = indices.detach().to(device="cpu", dtype=torch.int64).clone()
        if int(indices.min()) < 0 or int(indices.max()) >= height * width:
            raise ValueError("receiver site_indices lie outside the manifest grid")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("receiver site_indices must be unique")
        coordinates = self.physical_xz
        if (
            not isinstance(coordinates, torch.Tensor)
            or coordinates.shape != (indices.numel(), 2)
            or not coordinates.is_floating_point()
            or not bool(torch.isfinite(coordinates).all())
        ):
            raise ValueError("receiver physical_xz must be finite floating [R,2]")
        if isinstance(self.grid_size, tuple):
            object.__setattr__(self, "grid_size", (height, width))
        object.__setattr__(self, "site_indices", indices)
        object.__setattr__(self, "physical_xz", coordinates.detach().cpu().clone())
        object.__setattr__(self, "sha256", _sha256(self.sha256, "receiver manifest SHA-256"))

    @property
    def height(self) -> int:
        return self.grid_size if isinstance(self.grid_size, int) else self.grid_size[0]

    @property
    def width(self) -> int:
        return self.grid_size if isinstance(self.grid_size, int) else self.grid_size[1]


def quarter_slices(time_steps: int) -> dict[str, slice]:
    if time_steps != 160:
        raise ValueError("formal long-horizon metrics require 160 times")
    return dict(QUARTER_SLICES)


def _matching_fields(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.ndim != 4
        or prediction.shape != target.shape
        or prediction.shape[-1] != 160
        or min(prediction.shape) < 1
    ):
        raise ValueError("prediction and target must have matching [B,H,W,160] shape")
    if (
        not prediction.is_floating_point()
        or not target.is_floating_point()
        or prediction.dtype != target.dtype
        or prediction.device != target.device
    ):
        raise ValueError("prediction and target must share a real floating dtype and device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("prediction and target must contain only finite values")


def _matching_pair(
    prediction: torch.Tensor, target: torch.Tensor, ndim: int, name: str
) -> None:
    if (
        not isinstance(prediction, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or prediction.ndim != ndim
        or prediction.shape != target.shape
        or min(prediction.shape) < 1
    ):
        raise ValueError(f"{name} inputs require matching nonempty tensors")
    if (
        not prediction.is_floating_point()
        or not target.is_floating_point()
        or prediction.dtype != target.dtype
        or prediction.device != target.device
    ):
        raise ValueError(f"{name} inputs must share a real floating dtype and device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError(f"{name} inputs must contain only finite values")


def _time_vector(time_s: torch.Tensor, length: int, device: torch.device) -> torch.Tensor:
    if (
        not isinstance(time_s, torch.Tensor)
        or time_s.ndim != 1
        or time_s.numel() != length
        or not time_s.is_floating_point()
        or not bool(torch.isfinite(time_s).all())
        or not bool(torch.all(torch.diff(time_s) > 0))
    ):
        raise ValueError("time_s must be a finite strictly increasing vector matching time")
    return time_s.to(device=device)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    compute = torch.float64 if target.dtype == torch.float64 else torch.float32
    numerator = torch.linalg.vector_norm((prediction - target).to(compute))
    denominator = torch.linalg.vector_norm(target.to(compute)).clamp_min(eps)
    return float((numerator / denominator).detach().cpu())


def prediction_target_statistics(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> dict[str, float]:
    """Return physical-field norm ratio and centered Pearson correlation."""

    _matching_fields(prediction, target)
    compute = torch.float64 if target.dtype == torch.float64 else torch.float32
    prediction_flat = prediction.to(compute).reshape(-1)
    target_flat = target.to(compute).reshape(-1)
    prediction_norm = torch.linalg.vector_norm(prediction_flat)
    target_norm = torch.linalg.vector_norm(target_flat)
    norm_ratio = prediction_norm / target_norm.clamp_min(eps)
    prediction_centered = prediction_flat - prediction_flat.mean()
    target_centered = target_flat - target_flat.mean()
    denominator = (
        torch.linalg.vector_norm(prediction_centered)
        * torch.linalg.vector_norm(target_centered)
    )
    if float(denominator.detach().cpu()) <= eps:
        pearson = float(torch.equal(prediction_flat, target_flat))
    else:
        pearson = float(
            (torch.dot(prediction_centered, target_centered) / denominator)
            .clamp(-1.0, 1.0)
            .detach()
            .cpu()
        )
    return {
        "prediction_target_norm_ratio": float(norm_ratio.detach().cpu()),
        "prediction_target_pearson": pearson,
    }


def gather_native_receivers(
    field_bxzt: torch.Tensor, receivers: ReceiverSiteManifest
) -> torch.Tensor:
    if not isinstance(receivers, ReceiverSiteManifest):
        raise TypeError("receivers must be a ReceiverSiteManifest")
    if (
        not isinstance(field_bxzt, torch.Tensor)
        or field_bxzt.ndim != 4
        or field_bxzt.shape[1:3] != (receivers.height, receivers.width)
    ):
        raise ValueError("receiver gather requires canonical [B,X,Z,T] on the manifest grid")
    flat = field_bxzt.reshape(field_bxzt.shape[0], receivers.height * receivers.width, field_bxzt.shape[-1])
    return flat.index_select(1, receivers.site_indices.to(field_bxzt.device))


def target_active_time_errors(
    prediction_bxzt: torch.Tensor,
    target_bxzt: torch.Tensor,
    time_s: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    _matching_pair(prediction_bxzt, target_bxzt, 4, "active-time")
    time = _time_vector(time_s, target_bxzt.shape[-1], target_bxzt.device)
    target_norm = torch.linalg.vector_norm(target_bxzt.flatten(0, -2), dim=0)
    active = target_norm > eps * target_norm.max().clamp_min(eps)
    if not bool(active.any()):
        return {"active_time_coverage": 0.0, "active_time_error_slope_per_s": 0.0,
                "active_time_error_max": 0.0}
    error = torch.linalg.vector_norm((prediction_bxzt - target_bxzt).flatten(0, -2), dim=0)
    error = error / target_norm.clamp_min(eps)
    t, y = time[active].double(), error[active].double()
    centered_t = t - t.mean()
    slope = 0.0 if t.numel() < 2 else float(
        (centered_t * (y - y.mean())).sum() / centered_t.square().sum().clamp_min(eps)
    )
    return {"active_time_coverage": float(active.float().mean()),
            "active_time_error_slope_per_s": slope,
            "active_time_error_max": float(y.max())}


def receiver_arrival_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_s: torch.Tensor,
    threshold_fraction: float = 0.05,
    amplitude_eps: float = 1e-12,
) -> dict[str, float]:
    _matching_pair(prediction, target, 3, "arrival")
    time = _time_vector(time_s, target.shape[-1], target.device)
    if not math.isfinite(threshold_fraction) or not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must lie in (0,1]")
    peak = target.abs().amax(-1)
    target_has = peak > amplitude_eps
    threshold = (peak * threshold_fraction).clamp_min(amplitude_eps)[..., None]
    target_cross = target.abs() >= threshold
    prediction_cross = prediction.abs() >= threshold
    target_index = target_cross.to(torch.int64).argmax(-1)
    prediction_has = prediction_cross.any(-1)
    prediction_index = prediction_cross.to(torch.int64).argmax(-1)
    duration = float((time[-1] - time[0]).abs())
    errors = torch.full_like(peak, duration, dtype=torch.float64)
    both = target_has & prediction_has
    errors[both] = (time[prediction_index[both]] - time[target_index[both]]).abs().double()
    count = int(target_has.sum())
    if count == 0:
        return {"arrival_mae_s": 0.0, "arrival_miss_rate": 0.0,
                "arrival_target_coverage": 0.0}
    return {"arrival_mae_s": float(errors[target_has].mean()),
            "arrival_miss_rate": float((target_has & ~prediction_has).sum() / target_has.sum()),
            "arrival_target_coverage": float(target_has.float().mean())}


def receiver_lag_phase_metrics(
    prediction_brt: torch.Tensor,
    target_brt: torch.Tensor,
    time_s: torch.Tensor,
    *,
    temporal_modes: int = 32,
    max_lag_fraction: float = 0.25,
    eps: float = 1e-12,
) -> dict[str, float]:
    _matching_pair(prediction_brt, target_brt, 3, "receiver")
    time = _time_vector(time_s, target_brt.shape[-1], target_brt.device)
    if (
        isinstance(max_lag_fraction, bool)
        or not isinstance(max_lag_fraction, (int, float))
        or not math.isfinite(float(max_lag_fraction))
        or not 0 < float(max_lag_fraction) <= 0.5
    ):
        raise ValueError("max_lag_fraction must be finite and lie in (0,0.5]")
    if not isinstance(temporal_modes, int) or isinstance(temporal_modes, bool) or temporal_modes < 1:
        raise ValueError("temporal_modes must be a positive integer")
    pred = prediction_brt.reshape(-1, time.numel())
    target = target_brt.reshape(-1, time.numel())
    active = target.abs().amax(-1) > eps
    if not bool(active.any()):
        return {"receiver_lag_abs_s": 0.0, "receiver_xcorr_peak": 0.0,
                "receiver_phase_error": 0.0, "receiver_phase_coherence": 1.0}
    del temporal_modes
    uniform_time = torch.linspace(
        time[0], time[-1], time.numel(), device=time.device, dtype=time.dtype
    )
    right = torch.searchsorted(time.contiguous(), uniform_time).clamp(1, time.numel() - 1)
    left = right - 1
    alpha = ((uniform_time - time[left]) / (time[right] - time[left])).to(pred.dtype)
    pred_uniform = pred[:, left] * (1.0 - alpha) + pred[:, right] * alpha
    target_uniform = target[:, left] * (1.0 - alpha) + target[:, right] * alpha
    uniform_dt = float((uniform_time[-1] - uniform_time[0]) / (uniform_time.numel() - 1))
    maximum_lag = int(math.floor(float(max_lag_fraction) * (time.numel() - 1)))
    lag_seconds: list[float] = []
    peaks: list[float] = []
    for prediction_trace, target_trace in zip(
        pred_uniform[active], target_uniform[active], strict=True
    ):
        candidates: list[tuple[float, int, int, int]] = []
        denominator = (
            torch.linalg.vector_norm(prediction_trace)
            * torch.linalg.vector_norm(target_trace)
        ).clamp_min(eps)
        for lag in range(-maximum_lag, maximum_lag + 1):
            if lag > 0:
                pp, yy = prediction_trace[lag:], target_trace[:-lag]
            elif lag < 0:
                pp, yy = prediction_trace[:lag], target_trace[-lag:]
            else:
                pp, yy = prediction_trace, target_trace
            correlation = float(torch.dot(pp, yy) / denominator)
            candidates.append((correlation, -abs(lag), -lag, lag))
        peak, _, _, lag = max(candidates)
        lag_seconds.append(abs(lag) * uniform_dt)
        peaks.append(peak)
    prediction_analytic = analytic_signal_on_physical_time(pred[active], time)
    target_analytic = analytic_signal_on_physical_time(target[active], time)
    weight = target_analytic.abs()
    mask = weight > 1e-3 * weight.amax(-1, keepdim=True).clamp_min(eps)
    delta = torch.angle(prediction_analytic * target_analytic.conj()).abs()
    denominator = weight[mask].sum().clamp_min(eps)
    return {"receiver_lag_abs_s": sum(lag_seconds) / len(lag_seconds),
            "receiver_xcorr_peak": sum(peaks) / len(peaks),
            "receiver_phase_error": float((delta[mask] * weight[mask]).sum() / denominator),
            "receiver_phase_coherence": float((delta[mask].cos() * weight[mask]).sum() / denominator)}


def receiver_energy_metrics(
    prediction_brt: torch.Tensor,
    target_brt: torch.Tensor,
    time_s: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    _matching_pair(prediction_brt, target_brt, 3, "receiver")
    time = _time_vector(time_s, target_brt.shape[-1], target_brt.device)
    prediction = prediction_brt.reshape(-1, time.numel())
    target = target_brt.reshape(-1, time.numel())
    active = target.abs().amax(-1) > eps
    if not bool(active.any()):
        return {"energy_log_ratio": 0.0}
    weights = trapezoid_weights(time).to(target)
    prediction_energy = (prediction[active].square() * weights).sum(-1)
    target_energy = (target[active].square() * weights).sum(-1)
    return {"energy_log_ratio": float(torch.log((prediction_energy + eps) /
                                                  (target_energy + eps)).abs().mean())}


def spatial_high_k_mask(
    height: int, width: int, device: torch.device | str
) -> torch.Tensor:
    """Mirror Task 4's rFFT high-k mask onto a full complex FFT grid."""

    positive = normalized_spatial_frequency_radius(height, width, device) >= 0.5
    if width % 2 == 0:
        negative = positive[:, 1:-1].flip(1)
    else:
        negative = positive[:, 1:].flip(1)
    return torch.cat((positive, negative), dim=1)


def estimate_komega_workspace_bytes(
    batch: int,
    height: int,
    width: int,
    time_steps: int,
    temporal_modes: int,
    mode_block_size: int,
    dtype: torch.dtype,
) -> dict[str, int]:
    """Return an auditable upper bound for legacy and blocked temporary storage."""

    for name, value in (
        ("batch", batch), ("height", height), ("width", width),
        ("time_steps", time_steps), ("temporal_modes", temporal_modes),
        ("mode_block_size", mode_block_size),
    ):
        _positive_integer(value, name)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("workspace audit dtype must be float32 or float64")
    real_bytes = torch.empty((), dtype=dtype).element_size()
    sites = batch * height * width
    modes = min(temporal_modes, time_steps)
    block = min(mode_block_size, modes)
    legacy = 2 * sites * time_steps * modes * (2 * real_bytes)
    weighted_fields = 2 * sites * time_steps * real_bytes
    block_temporaries = 12 * sites * block * real_bytes
    bases = 2 * time_steps * block * real_bytes
    return {
        "legacy_dense_bytes": legacy,
        "blocked_peak_bytes": weighted_fields + block_temporaries + bases,
    }


def komega_metrics(
    prediction_bxzt: torch.Tensor,
    target_bxzt: torch.Tensor,
    time_s: torch.Tensor,
    *,
    temporal_modes: int = 32,
    high_k_fraction: float = 0.5,
    mode_block_size: int = 8,
    eps: float = 1e-12,
) -> dict[str, float]:
    _matching_pair(prediction_bxzt, target_bxzt, 4, "k-omega")
    time = _time_vector(time_s, target_bxzt.shape[-1], target_bxzt.device)
    if not isinstance(temporal_modes, int) or isinstance(temporal_modes, bool) or temporal_modes < 1:
        raise ValueError("temporal_modes must be a positive integer")
    _positive_integer(mode_block_size, "mode_block_size")
    modes = min(temporal_modes, time.numel())
    batch, height, width, length = prediction_bxzt.shape
    if high_k_fraction != 0.5:
        raise ValueError("Task 4 high-k threshold is fixed at 0.5")
    compute_dtype = torch.float64 if prediction_bxzt.dtype == torch.float64 else torch.float32
    normalized_time = ((time.double() - time[0].double()) /
                       (time[-1].double() - time[0].double())).to(compute_dtype)
    normalized_weights = (trapezoid_weights(time.double()) /
                          trapezoid_weights(time.double()).sum()).to(compute_dtype)
    sites = height * width
    prediction_weighted = prediction_bxzt.reshape(batch * sites, length).to(compute_dtype)
    target_weighted = target_bxzt.reshape(batch * sites, length).to(compute_dtype)
    prediction_weighted = prediction_weighted * normalized_weights
    target_weighted = target_weighted * normalized_weights
    high = spatial_high_k_mask(height, width, prediction_bxzt.device)
    total_num = torch.zeros((), dtype=compute_dtype, device=prediction_bxzt.device)
    total_den = torch.zeros_like(total_num)
    high_num = torch.zeros_like(total_num)
    high_den = torch.zeros_like(total_num)
    for start in range(0, modes, mode_block_size):
        stop = min(start + mode_block_size, modes)
        frequencies = 2.0 * torch.pi * torch.arange(
            start, stop, device=prediction_bxzt.device, dtype=compute_dtype
        )
        phase = normalized_time[:, None] * frequencies[None, :]
        cosine, negative_sine = torch.cos(phase), -torch.sin(phase)

        def coefficients(weighted: torch.Tensor) -> torch.Tensor:
            real = weighted @ cosine
            imaginary = weighted @ negative_sine
            return torch.complex(real, imaginary).reshape(
                batch, height, width, stop - start
            )

        prediction_k = torch.fft.fft2(
            coefficients(prediction_weighted), dim=(1, 2), norm="ortho"
        )
        target_k = torch.fft.fft2(
            coefficients(target_weighted), dim=(1, 2), norm="ortho"
        )
        difference = prediction_k - target_k
        total_num = total_num + difference.abs().square().sum()
        total_den = total_den + target_k.abs().square().sum()
        high_num = high_num + difference[:, high].abs().square().sum()
        high_den = high_den + target_k[:, high].abs().square().sum()
    total_relative = torch.sqrt(total_num) / torch.sqrt(total_den).clamp_min(eps)
    high_relative = torch.sqrt(high_num) / torch.sqrt(high_den).clamp_min(eps)
    return {"komega_relative_l2": float(total_relative),
            "komega_high": float(high_relative)}


def lag_phase_energy_and_komega_metrics(
    field_prediction: torch.Tensor,
    field_target: torch.Tensor,
    receiver_prediction: torch.Tensor,
    receiver_target: torch.Tensor,
    time_s: torch.Tensor,
    windows: dict[str, slice],
) -> dict[str, float]:
    result = target_active_time_errors(field_prediction, field_target, time_s)
    result.update(receiver_lag_phase_metrics(receiver_prediction, receiver_target, time_s))
    result.update(receiver_energy_metrics(receiver_prediction, receiver_target, time_s))
    result.update(komega_metrics(field_prediction, field_target, time_s))
    q4 = windows["q4"]
    q4_result = komega_metrics(field_prediction[..., q4], field_target[..., q4], time_s[q4])
    result["komega_relative_l2_q4"] = q4_result["komega_relative_l2"]
    result["komega_high_q4"] = q4_result["komega_high"]
    return result


def long_horizon_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    time_s: torch.Tensor,
    receivers: ReceiverSiteManifest,
) -> dict[str, float]:
    _matching_fields(prediction, target)
    time = _time_vector(time_s, 160, prediction.device)
    windows = quarter_slices(160)
    receiver_prediction = gather_native_receivers(prediction, receivers)
    receiver_target = gather_native_receivers(target, receivers)
    result = {"relative_l2": relative_l2(prediction, target),
              "receiver_relative_l2": relative_l2(receiver_prediction, receiver_target)}
    result.update(prediction_target_statistics(prediction, target))
    for name, window in windows.items():
        result[f"relative_l2_{name}"] = relative_l2(prediction[..., window], target[..., window])
        result[f"receiver_relative_l2_{name}"] = relative_l2(
            receiver_prediction[..., window], receiver_target[..., window]
        )
    eps = 1e-8
    result["field_q4_q1_ratio"] = (result["relative_l2_q4"] + eps) / (result["relative_l2_q1"] + eps)
    result["receiver_q4_q1_ratio"] = ((result["receiver_relative_l2_q4"] + eps) /
                                        (result["receiver_relative_l2_q1"] + eps))
    result["zero_relative_l2"] = relative_l2(torch.zeros_like(target), target)
    result["zero_relative_l2_q4"] = relative_l2(
        torch.zeros_like(target[..., windows["q4"]]), target[..., windows["q4"]]
    )
    result["zero_receiver_relative_l2"] = relative_l2(
        torch.zeros_like(receiver_target), receiver_target
    )
    result["zero_receiver_relative_l2_q4"] = relative_l2(
        torch.zeros_like(receiver_target[..., windows["q4"]]), receiver_target[..., windows["q4"]]
    )
    result.update(receiver_arrival_metrics(receiver_prediction, receiver_target, time))
    result.update(lag_phase_energy_and_komega_metrics(
        prediction, target, receiver_prediction, receiver_target, time, windows
    ))
    result["prediction_finite"] = float(torch.isfinite(prediction).all())
    result["prediction_nonzero"] = float(torch.count_nonzero(prediction) > 0)
    result["output_height"] = float(prediction.shape[1])
    result["output_width"] = float(prediction.shape[2])
    result["output_time_steps"] = float(prediction.shape[3])
    return result


__all__ = [
    "QUARTER_SLICES", "ReceiverSiteManifest", "estimate_komega_workspace_bytes",
    "gather_native_receivers",
    "komega_metrics", "lag_phase_energy_and_komega_metrics", "long_horizon_metrics",
    "prediction_target_statistics",
    "quarter_slices", "receiver_arrival_metrics", "receiver_energy_metrics",
    "receiver_lag_phase_metrics", "relative_l2", "target_active_time_errors",
    "spatial_high_k_mask",
]
