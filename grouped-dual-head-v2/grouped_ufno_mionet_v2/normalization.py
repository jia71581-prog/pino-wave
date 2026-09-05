"""Train-split-bound transforms between physical and order-one model units."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class ScaleMetadata:
    velocity_center_mps: float
    velocity_scale_mps: float
    pressure_scale_pa: float
    source_scales: tuple[float, float, float, float, float]
    train_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "ScaleMetadata":
        required = set(cls.__dataclass_fields__)
        data = {key: value for key, value in dict(values).items() if key in required}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"normalization metadata missing fields: {missing}")
        data["source_scales"] = tuple(float(value) for value in data["source_scales"])
        return cls(**data)


class PhysicalNormalizer:
    def __init__(self, metadata: ScaleMetadata):
        scales = (metadata.velocity_scale_mps, metadata.pressure_scale_pa, *metadata.source_scales)
        if any(float(value) <= 0 for value in scales):
            raise ValueError("normalization scales must be positive")
        if len(metadata.train_manifest_sha256) == 0:
            raise ValueError("train manifest digest is required")
        self.metadata = metadata

    @classmethod
    def from_dict(cls, values: Mapping[str, object], *, expected_manifest: str | None = None):
        metadata = ScaleMetadata.from_dict(values)
        if expected_manifest is not None and metadata.train_manifest_sha256 != expected_manifest:
            raise ValueError("normalization manifest does not match expected train manifest")
        return cls(metadata)

    def _source_scales(self, value) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        return torch.tensor(self.metadata.source_scales, dtype=torch.float32, device=tensor.device)

    def encode_velocity(self, value) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32)
        return (value - self.metadata.velocity_center_mps) / self.metadata.velocity_scale_mps

    def decode_velocity(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32) * self.metadata.velocity_scale_mps + self.metadata.velocity_center_mps

    def encode_source(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32) / self._source_scales(value)

    def decode_source(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32) * self._source_scales(value)

    def encode_pressure(self, value, amplitude) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32) / (
            self.metadata.pressure_scale_pa * torch.as_tensor(amplitude, dtype=torch.float32)
        )

    def decode_pressure(self, value, amplitude) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32) * (
            self.metadata.pressure_scale_pa * torch.as_tensor(amplitude, dtype=torch.float32)
        )
