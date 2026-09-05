"""Train-manifest-bound transforms between physical and model units."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Mapping

import numpy as np
import torch

from .config import ALLOWED_MEDIUM_TYPES


@dataclass(frozen=True)
class ScaleMetadata:
    velocity_center_mps: float
    velocity_scale_mps: float
    pressure_scale_pa: float
    source_scales: tuple[float, float, float, float, float]
    train_manifest_sha256: str
    allowed_medium_types: tuple[str, ...]
    record_count: int
    algorithm: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "ScaleMetadata":
        payload = dict(values)
        required = {item.name for item in fields(cls)}
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"normalization metadata missing fields: {missing}")
        payload = {key: payload[key] for key in required}
        payload["source_scales"] = tuple(float(value) for value in payload["source_scales"])
        payload["allowed_medium_types"] = tuple(str(value) for value in payload["allowed_medium_types"])
        return cls(**payload)


def fit_scale_metadata(
    train_velocity_mps,
    train_pressure_pa,
    train_source_parameters,
    *,
    train_manifest_sha256: str,
    record_count: int,
    pressure_percentile: float = 99.9,
) -> ScaleMetadata:
    velocity = np.asarray(torch.as_tensor(train_velocity_mps).detach().cpu(), dtype=np.float64).reshape(-1)
    pressure = np.abs(
        np.asarray(torch.as_tensor(train_pressure_pa).detach().cpu(), dtype=np.float64).reshape(-1)
    )
    pressure = pressure[pressure > 0]
    source = np.asarray(
        torch.as_tensor(train_source_parameters).detach().cpu(), dtype=np.float64
    )
    if velocity.size == 0 or pressure.size == 0 or source.size == 0:
        raise ValueError("nonempty train velocity, pressure, and source samples are required")
    if source.shape[-1] != 5 or not 0.0 < pressure_percentile <= 100.0:
        raise ValueError("source shape or pressure percentile is invalid")
    if not train_manifest_sha256 or record_count <= 0:
        raise ValueError("train manifest digest and record count are required")
    center = float(np.median(velocity))
    q25, q75 = np.percentile(velocity, [25.0, 75.0])
    velocity_scale = float(max(q75 - q25, np.std(velocity), 1.0))
    pressure_scale = float(max(np.percentile(pressure, pressure_percentile), np.finfo(np.float32).tiny))
    return ScaleMetadata(
        velocity_center_mps=center,
        velocity_scale_mps=velocity_scale,
        pressure_scale_pa=pressure_scale,
        source_scales=(2000.0, 2000.0, 50.0, 1.2, 1.0),
        train_manifest_sha256=str(train_manifest_sha256),
        allowed_medium_types=ALLOWED_MEDIUM_TYPES,
        record_count=int(record_count),
        algorithm="filtered_train_robust_percentile_v3",
    )


class PhysicalNormalizer:
    def __init__(self, metadata: ScaleMetadata):
        scales = (metadata.velocity_scale_mps, metadata.pressure_scale_pa, *metadata.source_scales)
        if any(not np.isfinite(value) or float(value) <= 0 for value in scales):
            raise ValueError("normalization scales must be finite and positive")
        if not metadata.train_manifest_sha256 or metadata.record_count <= 0:
            raise ValueError("normalization train manifest binding is required")
        if metadata.allowed_medium_types != ALLOWED_MEDIUM_TYPES:
            raise ValueError("normalization family contract does not match V3")
        self.metadata = metadata

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, object],
        *,
        expected_manifest: str | None = None,
    ) -> "PhysicalNormalizer":
        metadata = ScaleMetadata.from_dict(values)
        if expected_manifest is not None and metadata.train_manifest_sha256 != expected_manifest:
            raise ValueError("normalization manifest does not match expected train manifest")
        return cls(metadata)

    @staticmethod
    def _tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32)

    def encode_velocity(self, value) -> torch.Tensor:
        tensor = self._tensor(value)
        return (tensor - self.metadata.velocity_center_mps) / self.metadata.velocity_scale_mps

    def decode_velocity(self, value) -> torch.Tensor:
        return self._tensor(value) * self.metadata.velocity_scale_mps + self.metadata.velocity_center_mps

    def _source_scales(self, value) -> torch.Tensor:
        tensor = self._tensor(value)
        return torch.tensor(self.metadata.source_scales, dtype=torch.float32, device=tensor.device)

    def encode_source(self, value) -> torch.Tensor:
        return self._tensor(value) / self._source_scales(value)

    def decode_source(self, value) -> torch.Tensor:
        return self._tensor(value) * self._source_scales(value)

    def _amplitude_for(self, value: torch.Tensor, amplitude) -> torch.Tensor:
        scale = self._tensor(amplitude).to(value.device)
        while scale.ndim < value.ndim:
            scale = scale.unsqueeze(-1)
        return scale

    def encode_pressure(self, value, amplitude) -> torch.Tensor:
        tensor = self._tensor(value)
        return tensor / (self.metadata.pressure_scale_pa * self._amplitude_for(tensor, amplitude))

    def decode_pressure(self, value, amplitude) -> torch.Tensor:
        tensor = self._tensor(value)
        return tensor * (self.metadata.pressure_scale_pa * self._amplitude_for(tensor, amplitude))


__all__ = ["PhysicalNormalizer", "ScaleMetadata", "fit_scale_metadata"]
