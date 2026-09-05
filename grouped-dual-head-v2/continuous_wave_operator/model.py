from __future__ import annotations

import torch
from torch import nn

from .config import DomainConfig, ModelConfig
from .medium_encoder import MediumCache, MediumEncoder
from .query_decoder import QueryDecoder
from .source_encoder import SourceCache, SourceEncoder


class ContinuousWaveOperator(nn.Module):
    """Medium/source-decoupled continuous acoustic wave operator."""

    def __init__(self, config: ModelConfig, domain: DomainConfig) -> None:
        super().__init__()
        self.config = config
        self.domain = domain
        self.medium_encoder = MediumEncoder(config)
        self.source_encoder = SourceEncoder(config)
        self.query_decoder = QueryDecoder(config, domain)

    def encode_medium(self, velocity_mps: torch.Tensor) -> MediumCache:
        return self.medium_encoder(velocity_mps)

    def encode_sources(
        self,
        medium: MediumCache,
        source_map: torch.Tensor,
        source_params: torch.Tensor,
    ) -> SourceCache:
        return self.source_encoder(medium, source_map, source_params)

    def query(
        self,
        medium: MediumCache,
        sources: SourceCache,
        query_coords: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        query_count = query_coords.shape[-2]
        if chunk_size is None or chunk_size >= query_count:
            return self.query_decoder(medium, sources, query_coords)
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        chunks = [
            self.query_decoder(medium, sources, query_coords[:, :, start : start + chunk_size])
            for start in range(0, query_count, chunk_size)
        ]
        return torch.cat(chunks, dim=-1)

    def forward(
        self,
        velocity_mps: torch.Tensor,
        source_map: torch.Tensor,
        source_params: torch.Tensor,
        query_coords: torch.Tensor,
        *,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        medium = self.encode_medium(velocity_mps)
        sources = self.encode_sources(medium, source_map, source_params)
        return self.query(medium, sources, query_coords, chunk_size=chunk_size)
