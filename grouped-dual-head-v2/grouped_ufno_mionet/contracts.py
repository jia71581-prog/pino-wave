"""Runtime-checked records used by the grouped data and model interfaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch


def _tensor(value: Any, *, name: str, ndim: int | None = None) -> torch.Tensor:
    out = torch.as_tensor(value)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {tuple(out.shape)}")
    if not torch.is_floating_point(out):
        out = out.float()
    if not torch.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


@dataclass
class SourceParameters:
    """One source: ``(x_m, z_m, f0_hz, t0_s, amplitude)``."""
    x_m: float
    z_m: float
    f0_hz: float
    t0_s: float
    amplitude: float = 1.0

    def as_tensor(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        result = torch.tensor([self.x_m, self.z_m, self.f0_hz, self.t0_s, self.amplitude], device=device, dtype=dtype)
        self.validate()
        return result

    def validate(self):
        vals = [self.x_m, self.z_m, self.f0_hz, self.t0_s, self.amplitude]
        if not all(torch.isfinite(torch.tensor(v)) for v in vals):
            raise ValueError("source parameters must be finite")
        if self.f0_hz <= 0:
            raise ValueError("source frequency must be positive")
        if self.t0_s < 0:
            raise ValueError("source delay must be non-negative")
        if self.amplitude == 0:
            raise ValueError("source amplitude must be non-zero")


@dataclass
class SingleSourceRecord:
    velocity_mps: torch.Tensor
    source_parameters: SourceParameters | torch.Tensor
    pressure_tzx: torch.Tensor | None = None
    sample_id: str | int | None = None
    group_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.velocity_mps = _tensor(self.velocity_mps, name="velocity_mps", ndim=3)
        if self.velocity_mps.shape[0] != 1:
            raise ValueError("velocity_mps must have shape [1,z,x]")
        if isinstance(self.source_parameters, SourceParameters):
            self.source_parameters.validate()
        else:
            src = _tensor(self.source_parameters, name="source_parameters")
            if src.ndim != 1 or src.shape != (5,):
                raise ValueError("each record must contain exactly one source with five parameters")
            if src[2] <= 0 or src[4] == 0:
                raise ValueError("source frequency must be positive and amplitude must be non-zero")
            self.source_parameters = src
        if self.pressure_tzx is not None:
            self.pressure_tzx = _tensor(self.pressure_tzx, name="pressure_tzx", ndim=3)


@dataclass
class GroupedMacroBatch:
    """A grouped batch; targets remain record-specific and are never summed."""
    velocity_mps: torch.Tensor  # [M,1,z,x]
    record_to_medium: torch.Tensor  # [S]
    source_parameters: torch.Tensor  # [S,5]
    targets: torch.Tensor | None = None
    sample_id: tuple[str | int, ...] = ()
    query_blocks: tuple[Any, ...] = ()

    def __post_init__(self):
        self.velocity_mps = _tensor(self.velocity_mps, name="velocity_mps", ndim=4)
        self.record_to_medium = torch.as_tensor(self.record_to_medium, dtype=torch.long)
        self.source_parameters = _tensor(self.source_parameters, name="source_parameters", ndim=2)
        if self.source_parameters.shape[1] != 5:
            raise ValueError("source_parameters must have shape [records,5]")
        if self.record_to_medium.shape != (self.source_parameters.shape[0],):
            raise ValueError("record_to_medium must have one entry per source record")
        if self.record_to_medium.numel() and (self.record_to_medium.min() < 0 or self.record_to_medium.max() >= self.velocity_mps.shape[0]):
            raise ValueError("record_to_medium contains an invalid medium index")
        if self.targets is not None:
            self.targets = _tensor(self.targets, name="targets")

    @property
    def num_records(self) -> int:
        return int(self.source_parameters.shape[0])

    def pin_memory(self):
        """DataLoader hook: pin nested tensors for asynchronous H2D copies."""
        self.velocity_mps = self.velocity_mps.pin_memory()
        self.record_to_medium = self.record_to_medium.pin_memory()
        self.source_parameters = self.source_parameters.pin_memory()
        if self.targets is not None:
            self.targets = self.targets.pin_memory()
        self.query_blocks = tuple(
            type(block)(block.coords.pin_memory(), block.targets.pin_memory())
            for block in self.query_blocks
        )
        return self
