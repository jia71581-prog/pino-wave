from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .medium_encoder import MediumCache


@dataclass(frozen=True)
class SourceCache:
    latent: torch.Tensor
    source_mass: torch.Tensor
    amplitude: torch.Tensor
    parameters: torch.Tensor


class SourceEncoder(nn.Module):
    """Pool medium features conservatively around one or more point sources."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.medium_projection = nn.Linear(4 * config.width, config.width)
        self.parameter_projection = nn.Sequential(
            nn.Linear(36, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.activation = nn.SiLU()
        self.register_buffer("parameter_scale", torch.tensor([2000.0, 2000.0, 20.0, 1.0]))
        self.register_buffer("fourier_bands", torch.tensor([1.0, 2.0, 4.0, 8.0]))

    def _parameter_features(self, parameters: torch.Tensor) -> torch.Tensor:
        normalized = parameters[..., :4] / self.parameter_scale
        phase = torch.pi * normalized.unsqueeze(-1) * self.fourier_bands
        encoded = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2)
        return torch.cat((normalized, encoded), dim=-1)

    def forward(
        self,
        medium: MediumCache,
        source_map: torch.Tensor,
        source_parameters: torch.Tensor,
    ) -> SourceCache:
        if source_map.ndim != 5 or source_map.shape[2] != 1:
            raise ValueError("source_map must have shape [B,S,1,H,W]")
        batch, shots, _, height, width = source_map.shape
        if source_parameters.shape != (batch, shots, 5):
            raise ValueError("source_parameters must have shape [B,S,5]")
        if medium.local_features[0].shape[0] != batch:
            raise ValueError("medium and source batches differ")
        if medium.local_features[0].shape[-2:] != (height, width):
            raise ValueError("source_map and medium spatial shapes differ")
        if not torch.isfinite(source_map).all() or torch.any(source_map < 0.0):
            raise ValueError("source_map must be finite and nonnegative")
        if not torch.isfinite(source_parameters).all():
            raise ValueError("source_parameters must be finite")

        source_mass = source_map.sum(dim=(-3, -2, -1))
        if not torch.allclose(source_mass, torch.ones_like(source_mass), atol=1.0e-5, rtol=1.0e-5):
            raise ValueError("each source map must have unit mass")

        flat_maps = source_map.reshape(batch * shots, 1, height, width)
        pooled_scales: list[torch.Tensor] = []
        for feature in medium.local_features:
            resized = F.interpolate(flat_maps, size=feature.shape[-2:], mode="area")
            resized = resized / resized.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0e-12)
            resized = resized.reshape(batch, shots, 1, *feature.shape[-2:])
            pooled = (feature[:, None] * resized).sum(dim=(-2, -1))
            pooled_scales.append(pooled)
        medium_latent = self.medium_projection(torch.cat(pooled_scales, dim=-1))
        parameter_latent = self.parameter_projection(self._parameter_features(source_parameters))
        latent = self.activation(medium_latent + parameter_latent)
        return SourceCache(
            latent=latent,
            source_mass=source_mass,
            amplitude=source_parameters[..., 4],
            parameters=source_parameters,
        )
