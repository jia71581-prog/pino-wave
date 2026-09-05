from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DomainConfig, ModelConfig
from .coordinates import normalize_query_coordinates, physical_output_gate
from .medium_encoder import MediumCache
from .source_encoder import SourceCache


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int, source_width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))
        self.film = nn.Linear(source_width, 2 * width)

    def forward(self, inputs: torch.Tensor, source_latent: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(source_latent).chunk(2, dim=-1)
        update = self.mlp(self.norm(inputs))
        return inputs + update * (1.0 + gamma[:, :, None]) + beta[:, :, None]


class QueryDecoder(nn.Module):
    """Decode scalar pressure at continuous physical ``(x,z,t)`` queries."""

    def __init__(self, config: ModelConfig, domain: DomainConfig) -> None:
        super().__init__()
        self.config = config
        self.domain = domain
        coordinate_features = 3 + 6 * config.query_fourier_bands
        self.coordinate_projection = nn.Linear(coordinate_features, config.width)
        self.attention = nn.MultiheadAttention(
            config.width, config.attention_heads, batch_first=True
        )
        self.input_projection = nn.Linear(7 * config.width, config.decoder_width)
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(config.decoder_width, config.width)
            for _ in range(config.decoder_layers)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.decoder_width),
            nn.Linear(config.decoder_width, config.decoder_width),
            nn.SiLU(),
            nn.Linear(config.decoder_width, 1),
        )
        self.register_buffer(
            "fourier_bands",
            torch.pow(2.0, torch.arange(config.query_fourier_bands, dtype=torch.float32)),
        )

    def _coordinate_features(self, normalized: torch.Tensor) -> torch.Tensor:
        phase = torch.pi * normalized.unsqueeze(-1) * self.fourier_bands
        fourier = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1).flatten(-2)
        return torch.cat((normalized, fourier), dim=-1)

    @staticmethod
    def _sample_local_features(
        local_features: tuple[torch.Tensor, ...], normalized: torch.Tensor
    ) -> torch.Tensor:
        batch, shots, queries, _ = normalized.shape
        grid = normalized[..., :2].reshape(batch, shots * queries, 1, 2)
        sampled_scales: list[torch.Tensor] = []
        for feature in local_features:
            sampled = F.grid_sample(
                feature,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled = sampled.squeeze(-1).reshape(batch, feature.shape[1], shots, queries)
            sampled_scales.append(sampled.permute(0, 2, 3, 1))
        return torch.cat(sampled_scales, dim=-1)

    def forward(
        self,
        medium: MediumCache,
        sources: SourceCache,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        if query_coords.ndim != 4 or query_coords.shape[-1] != 3:
            raise ValueError("query_coords must have shape [B,S,Q,3]")
        batch, shots, queries, _ = query_coords.shape
        if sources.latent.shape[:2] != (batch, shots):
            raise ValueError("query and source batch/shot dimensions differ")
        if medium.global_tokens.shape[0] != batch:
            raise ValueError("query and medium batches differ")
        if not torch.isfinite(query_coords).all():
            raise ValueError("query_coords must be finite")
        normalized = normalize_query_coordinates(query_coords, self.domain)
        if torch.any(normalized < -1.0) or torch.any(normalized > 1.0):
            raise ValueError("query_coords are outside the configured domain")

        coordinate_latent = self.coordinate_projection(self._coordinate_features(normalized))
        attention_query = coordinate_latent + sources.latent[:, :, None]
        flat_query = attention_query.reshape(batch * shots, queries, self.config.width)
        tokens = medium.global_tokens[:, None].expand(-1, shots, -1, -1)
        tokens = tokens.reshape(batch * shots, tokens.shape[2], self.config.width)
        attended, _ = self.attention(flat_query, tokens, tokens, need_weights=False)
        attended = attended.reshape(batch, shots, queries, self.config.width)
        local = self._sample_local_features(medium.local_features, normalized)
        source = sources.latent[:, :, None].expand(-1, -1, queries, -1)
        hidden = self.input_projection(torch.cat((local, coordinate_latent, attended, source), dim=-1))
        for block in self.blocks:
            hidden = block(hidden, sources.latent)
        raw_pressure = self.output(hidden).squeeze(-1)
        return (
            self.config.output_pressure_scale
            * raw_pressure
            * physical_output_gate(query_coords, self.domain)
            * sources.amplitude[:, :, None]
        )
