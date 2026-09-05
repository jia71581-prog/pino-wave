"""Local travel-time and source-phase fields for V4 dense decoding."""
from __future__ import annotations

import math
from typing import Sequence

import torch

from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime


def dense_propagation_features(
    travel: RayTravelTime,
    time_s: torch.Tensor,
    source_parameters: torch.Tensor,
    *,
    height: int,
    width: int,
    domain_t_s: float,
    domain_diagonal_m: float,
    causal_width_s: float = 0.005,
    gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
    include_travel_progress: bool = False,
) -> torch.Tensor:
    """Return `[record,time,C,z,x]` propagation-aligned features.

    ``C`` is 12 by default (the legacy channel set).  When
    ``include_travel_progress`` is True an extra 13th channel is appended: the
    normalized elapsed propagation distance ``c0·(t - t0)/diagonal`` broadcast
    over space, giving the renderer an explicit "how far the wavefront has
    travelled" signal that the first-arrival retarded-time channels do not carry.
    This directly targets the long-time (late-frame) regime where multiples and
    long-travel coda dominate.
    """

    seconds = torch.as_tensor(travel.seconds)
    if seconds.ndim != 2:
        raise ValueError("dense travel time must have shape [record,point]")
    records, points = seconds.shape
    if height <= 0 or width <= 0 or points != int(height) * int(width):
        raise ValueError("dense travel grid point count does not match height and width")
    if domain_t_s <= 0 or domain_diagonal_m <= 0 or causal_width_s <= 0:
        raise ValueError("propagation feature scales must be positive")
    if len(tuple(gabor_scales_s)) != 4 or any(float(scale) <= 0 for scale in gabor_scales_s):
        raise ValueError("V4 requires four positive Gabor time scales")

    device = seconds.device
    dtype = torch.float32
    times = torch.as_tensor(time_s, dtype=dtype, device=device)
    source = torch.as_tensor(source_parameters, dtype=dtype, device=device)
    if times.ndim != 2 or times.shape[0] != records or times.shape[1] == 0:
        raise ValueError("dense requested times must have shape [record,time]")
    if source.shape != (records, 5):
        raise ValueError("source parameters must have shape [record,5]")

    travel_fields = (
        travel.seconds,
        travel.distance_m,
        travel.path_velocity_mps,
        travel.endpoint_velocity_mps,
        travel.mean_slowness_s_per_m,
    )
    if any(torch.as_tensor(field).shape != (records, points) for field in travel_fields):
        raise ValueError("dense travel feature fields must share shape [record,point]")

    seconds = seconds.to(dtype=dtype)
    tau = times[:, :, None] - source[:, None, 3:4] - seconds[:, None]
    frequency = source[:, None, 2:3]
    phase = 2.0 * math.pi * frequency * tau
    causal = torch.sigmoid(tau / float(causal_width_s))

    def repeated(field: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(field, dtype=dtype, device=device)[:, None].expand_as(tau)

    values = [
        tau / float(domain_t_s),
        repeated(travel.seconds) / float(domain_t_s),
        repeated(travel.distance_m) / float(domain_diagonal_m),
        repeated(travel.path_velocity_mps) / 5000.0,
        repeated(travel.endpoint_velocity_mps) / 5000.0,
        causal,
        torch.sin(phase),
        torch.cos(phase),
    ]
    values.extend(
        torch.exp(-0.5 * (tau / float(scale)).square())
        for scale in gabor_scales_s
    )
    if include_travel_progress:
        # Normalized elapsed propagation distance c0*(t - t0)/diagonal, broadcast
        # over space.  Uses the source-endpoint velocity as a scalar reference c0
        # so the channel is a monotone "long-travel" clock independent of arrival.
        elapsed = (times - source[:, 3:4])[:, :, None]                       # (rec,time,1)
        reference_c = travel.endpoint_velocity_mps.to(dtype=dtype, device=device)
        c0 = reference_c.mean(dim=1, keepdim=True)[:, None]                  # (rec,1,1)
        progress = (c0 * elapsed) / float(domain_diagonal_m)
        values.append(progress.expand_as(tau))
    channels = len(values)
    features = torch.stack(values, dim=2)
    return features.reshape(records, times.shape[1], channels, int(height), int(width))


__all__ = ["dense_propagation_features"]
