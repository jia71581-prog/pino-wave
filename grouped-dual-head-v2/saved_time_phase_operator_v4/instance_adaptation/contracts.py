"""Hard data-access contracts for causal instance adaptation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import torch


def onset_indices(
    time_s: torch.Tensor,
    t0_s: float,
    f0_hz: float,
    *,
    lead_cycles: float = 1.0,
) -> tuple[int, int]:
    """Return two frames after the significant onset preceding the Ricker peak."""
    axis = torch.as_tensor(time_s, dtype=torch.float64).flatten()
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError("time axis cannot provide two onset frames")
    if not torch.isfinite(axis).all() or bool((axis[1:] <= axis[:-1]).any()):
        raise ValueError("time axis must be finite and strictly increasing")
    onset = float(t0_s)
    if not torch.isfinite(torch.tensor(onset)):
        raise ValueError("source onset must be finite")
    frequency = float(f0_hz)
    cycles = float(lead_cycles)
    if not torch.isfinite(torch.tensor(frequency)) or frequency <= 0.0:
        raise ValueError("source frequency must be finite and positive")
    if not torch.isfinite(torch.tensor(cycles)) or cycles < 0.0:
        raise ValueError("onset lead cycles must be finite and nonnegative")
    onset -= cycles / frequency
    k0 = int(torch.searchsorted(axis, torch.tensor(onset, dtype=axis.dtype), right=False))
    if k0 < 0 or k0 + 1 >= int(axis.numel()):
        raise ValueError("two onset frames are outside the saved time axis")
    return k0, k0 + 1


def future_indices(time_count: int, observed_indices: tuple[int, int]) -> torch.Tensor:
    """Return saved indices strictly after the two observed onset frames."""
    count = int(time_count)
    if count <= 0 or len(observed_indices) != 2:
        raise ValueError("time count and two observed indices are required")
    k0, k1 = (int(value) for value in observed_indices)
    if not 0 <= k0 < k1 < count:
        raise ValueError("observed indices must be ordered and inside the saved axis")
    return torch.arange(k1 + 1, count, dtype=torch.long)


class SnapshotAccessAudit:
    """Audit that a deployment adapter reads only the two true onset frames."""

    def __init__(self, allowed_indices: Iterable[int]):
        allowed = tuple(int(value) for value in allowed_indices)
        if len(allowed) != 2 or allowed[0] < 0 or allowed[1] != allowed[0] + 1:
            raise ValueError("audit requires two adjacent nonnegative onset indices")
        self.allowed_indices = allowed
        self.requested_indices: list[int] = []

    def read(self, indices: Iterable[int]) -> tuple[int, ...]:
        requested = tuple(int(value) for value in indices)
        for value in requested:
            if value not in self.requested_indices:
                self.requested_indices.append(value)
        if any(value not in self.allowed_indices for value in requested):
            raise PermissionError("future truth access is forbidden during adaptation")
        return requested

    def payload(self) -> dict[str, object]:
        return {
            "allowed_indices": self.allowed_indices,
            "requested_indices": tuple(self.requested_indices),
            "future_truth_used": any(
                value not in self.allowed_indices for value in self.requested_indices
            ),
        }


@dataclass(frozen=True)
class SyntheticBridgeProvenance:
    """Provenance for a bridge tensor generated without later true snapshots."""

    synthetic: bool
    true_indices: tuple[int, int]
    tensor_sha256: str

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, true_indices: tuple[int, int]):
        value = torch.as_tensor(tensor).detach().cpu().contiguous().numpy().tobytes()
        return cls(
            synthetic=True,
            true_indices=tuple(int(index) for index in true_indices),
            tensor_sha256=hashlib.sha256(value).hexdigest(),
        )

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


__all__ = [
    "SnapshotAccessAudit",
    "SyntheticBridgeProvenance",
    "future_indices",
    "onset_indices",
]
