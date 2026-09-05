"""Strict causal boundary for two onset-aligned wavefield snapshots."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OnsetWindow:
    """Saved-frame indices visible to deployment and the strictly future domain."""

    observed_indices: tuple[int, int]
    future_indices: torch.Tensor


def onset_window(time_s: torch.Tensor, source_t0_s: float) -> OnsetWindow:
    """Return the first two saved frames at or after an instance's source delay."""
    time_s = torch.as_tensor(time_s, dtype=torch.float32)
    if time_s.ndim != 1 or time_s.numel() < 2 or not torch.isfinite(time_s).all():
        raise ValueError("time_s must be a finite one-dimensional saved-time axis")
    starts = torch.nonzero(time_s >= float(source_t0_s), as_tuple=False).flatten()
    if starts.numel() == 0 or int(starts[0]) + 1 >= time_s.numel():
        raise ValueError("two onset-aligned frames are unavailable")
    k0 = int(starts[0])
    return OnsetWindow((k0, k0 + 1), torch.arange(k0 + 2, time_s.numel()))


def hard_project_observations(
    prediction: torch.Tensor,
    observed_wavefield: torch.Tensor,
    observed_indices: tuple[int, int],
) -> torch.Tensor:
    """Replace exactly the two visible time frames while retaining all predictions elsewhere."""
    if prediction.ndim != 4 or observed_wavefield.ndim != 4:
        raise ValueError("prediction and observed_wavefield must use [batch,time,z,x] axes")
    if observed_wavefield.shape[1] != 2 or prediction.shape[0] != observed_wavefield.shape[0]:
        raise ValueError("exactly two observed frames with matching batch size are required")
    if prediction.shape[2:] != observed_wavefield.shape[2:]:
        raise ValueError("observation spatial shape must match prediction")
    if len(observed_indices) != 2 or min(observed_indices) < 0 or max(observed_indices) >= prediction.shape[1]:
        raise ValueError("observed indices lie outside prediction time axis")
    out = prediction.clone()
    out[:, list(observed_indices)] = observed_wavefield.to(dtype=out.dtype, device=out.device)
    return out


class WavefieldAccessAudit:
    """Reject deployment attempts to read a reference frame outside the two observations."""

    def __init__(self, observed_indices: tuple[int, int]):
        self.observed_indices = tuple(int(index) for index in observed_indices)
        self.accessed_indices: tuple[int, ...] = ()

    def record(self, indices) -> None:
        requested = tuple(int(index) for index in indices)
        illegal = set(requested) - set(self.observed_indices)
        if illegal:
            raise RuntimeError(f"future wavefield access is forbidden: {sorted(illegal)}")
        self.accessed_indices = requested
