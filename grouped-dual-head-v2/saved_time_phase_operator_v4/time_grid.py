"""Strict indexing for the immutable HDF5 output-time grid."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SavedTimeGrid:
    """Uniform saved time values and their exact-query tolerance."""

    values_s: torch.Tensor
    tolerance_s: float

    @classmethod
    def from_values(
        cls,
        values: torch.Tensor | tuple[float, ...] | list[float],
        *,
        tolerance_fraction: float = 1.0e-4,
    ) -> "SavedTimeGrid":
        axis = torch.as_tensor(values, dtype=torch.float64).detach().clone().cpu()
        if axis.ndim != 1 or len(axis) < 2 or not bool(torch.isfinite(axis).all()):
            raise ValueError("stored time axis must be a finite one-dimensional sequence")
        spacing = torch.diff(axis)
        if bool(torch.any(spacing <= 0)):
            raise ValueError("stored time axis must be strictly increasing")
        if not torch.allclose(spacing, spacing[0], rtol=0.0, atol=1.0e-12):
            raise ValueError("stored time axis must be uniform")
        if not 0.0 < tolerance_fraction < 0.5:
            raise ValueError("stored time tolerance fraction must lie in (0, 0.5)")
        return cls(
            values_s=axis,
            tolerance_s=float(spacing[0]) * float(tolerance_fraction),
        )

    @property
    def count(self) -> int:
        return len(self.values_s)

    @property
    def spacing_s(self) -> float:
        return float(self.values_s[1] - self.values_s[0])

    def indices(self, requested_s: torch.Tensor) -> torch.Tensor:
        """Map stored physical times to indices and reject interpolation times."""

        values = torch.as_tensor(requested_s)
        if not values.dtype.is_floating_point:
            values = values.float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("requested stored times must be finite")
        axis = self.values_s.to(device=values.device)
        values64 = values.to(dtype=torch.float64)
        right = torch.searchsorted(axis, values64).clamp(0, len(axis) - 1)
        left = (right - 1).clamp(0, len(axis) - 1)
        choose_left = (values64 - axis[left]).abs() <= (values64 - axis[right]).abs()
        index = torch.where(choose_left, left, right)
        if bool(torch.any((values64 - axis[index]).abs() > self.tolerance_s)):
            raise ValueError("requested time is not a stored HDF5 time")
        return index.long()

    def values_at(self, indices: torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
        selected = torch.as_tensor(indices, dtype=torch.long)
        if bool(torch.any(selected < 0)) or bool(torch.any(selected >= self.count)):
            raise ValueError("stored time index lies outside the HDF5 time axis")
        target_device = selected.device if device is None else device
        return self.values_s.to(target_device)[selected.to(target_device)]


__all__ = ["SavedTimeGrid"]
