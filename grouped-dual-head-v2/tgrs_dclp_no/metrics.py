"""Method-neutral dispersion metrics on stored wavefields and receiver traces.

These operate on plain tensors so that any method (traditional solver, operator baseline,
A+1) is scored identically.  Two physically legible dispersion diagnostics are provided:

* ``receiver_phase_metrics`` -- first-arrival / phase lag and coherence at receivers,
  the direct signature of a numerically advanced or retarded (dispersed) wave;
* ``wavefront_radius``       -- radius of peak radial energy about the source, i.e. the
  wavefront *position*, which a dispersive scheme misplaces.

Receiver traces use the convention ``[receiver, time]``; wavefields use ``[z, x]``.
"""
from __future__ import annotations

import torch


def _lag_one(prediction: torch.Tensor, target: torch.Tensor) -> int:
    """Integer sample lag maximizing cross-correlation of two mean-removed traces.

    A positive lag means the prediction is delayed relative to the target.
    """
    left = prediction.double() - prediction.double().mean()
    right = target.double() - target.double().mean()
    size = int(left.numel())
    correlation = torch.fft.irfft(
        torch.fft.rfft(left, n=2 * size) * torch.conj(torch.fft.rfft(right, n=2 * size)),
        n=2 * size,
    )
    signed = int(correlation.argmax())
    if signed > size:
        signed -= 2 * size
    return signed


def receiver_phase_metrics(prediction, target, *, dt_s: float) -> dict[str, float]:
    """First-arrival lag and coherence over a receiver line ``[receiver, time]``.

    Returns median signed lag (samples and seconds), the 95th-percentile absolute lag,
    and the mean lag-aligned trace coherence.  A dispersion-free reproduction has zero
    median lag and coherence near 1.
    """
    predicted = torch.as_tensor(prediction).double()
    reference = torch.as_tensor(target).double()
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("receiver traces must match and be 2-D [receiver, time]")
    if not torch.isfinite(predicted).all() or not torch.isfinite(reference).all():
        raise ValueError("receiver traces must be finite")
    lags = torch.tensor(
        [_lag_one(predicted[index], reference[index]) for index in range(predicted.shape[0])],
        dtype=torch.int64,
    )
    coherence = []
    for index, lag in enumerate(lags.tolist()):
        aligned = torch.roll(predicted[index], shifts=-lag)
        numerator = torch.dot(aligned, reference[index])
        denominator = aligned.norm() * reference[index].norm()
        coherence.append(float(numerator / denominator.clamp_min(1.0e-12)))
    median_lag = int(torch.median(lags))
    return {
        "median_lag_samples": float(median_lag),
        "median_lag_s": float(median_lag * float(dt_s)),
        "p95_absolute_lag_samples": float(torch.quantile(lags.abs().double(), 0.95)),
        "mean_coherence": float(torch.tensor(coherence).mean()),
    }


def wavefront_radius(field, *, source_x_m: float, source_z_m: float, x_m, z_m) -> float:
    """Radius (m) of peak radial energy about the source for a single ``[z, x]`` frame.

    Bins squared amplitude into radial shells about the source and returns the shell of
    maximum mean energy -- an estimate of the wavefront position that a dispersive scheme
    advances or lags.
    """
    value = torch.as_tensor(field).double()
    x_axis = torch.as_tensor(x_m).double()
    z_axis = torch.as_tensor(z_m).double()
    if value.ndim != 2 or value.shape != (z_axis.numel(), x_axis.numel()):
        raise ValueError("field must be 2-D [z, x] matching the supplied axes")
    zz, xx = torch.meshgrid(z_axis, x_axis, indexing="ij")
    radius = torch.sqrt((xx - float(source_x_m)) ** 2 + (zz - float(source_z_m)) ** 2)
    spacing = float(min(x_axis[1] - x_axis[0], z_axis[1] - z_axis[0]))
    bins = torch.round(radius / spacing).long()
    energy = torch.zeros(int(bins.max()) + 1, dtype=torch.float64)
    counts = torch.zeros_like(energy)
    energy.scatter_add_(0, bins.flatten(), value.square().flatten())
    counts.scatter_add_(0, bins.flatten(), torch.ones_like(value).flatten())
    profile = energy / counts.clamp_min(1.0)
    return float(profile.argmax() * spacing)
