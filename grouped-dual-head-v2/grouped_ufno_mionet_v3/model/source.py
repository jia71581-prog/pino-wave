"""Independent physical-source, source-map, and local-medium encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .medium import MediumEncoding


@dataclass(frozen=True)
class SourceEncoding:
    hidden: torch.Tensor
    rank: torch.Tensor
    map_field: torch.Tensor
    local_medium: torch.Tensor


class SourceEncoderV3(nn.Module):
    def __init__(self, *, width: int = 64, rank: int = 128) -> None:
        super().__init__()
        self.parameter_mlp = nn.Sequential(
            nn.Linear(5, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.local_projection = nn.Linear(width, width)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.map_projection = nn.Linear(width, width)
        self.hidden_norm = nn.LayerNorm(width)
        self.rank_projection = nn.Linear(width, rank)

    def forward(
        self,
        source_normalized: torch.Tensor,
        source_map: torch.Tensor,
        medium: MediumEncoding,
        record_to_medium: torch.Tensor,
    ) -> SourceEncoding:
        source = torch.as_tensor(source_normalized)
        source_map = torch.as_tensor(source_map, dtype=source.dtype, device=source.device)
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=source.device)
        if source.ndim != 2 or source.shape[-1] != 5:
            raise ValueError("source_normalized must be [record,5]")
        if source_map.ndim != 4 or source_map.shape[:2] != (source.shape[0], 1):
            raise ValueError("source_map must be [record,1,z,x]")
        if mapping.shape != (source.shape[0],):
            raise ValueError("record_to_medium must contain one index per source")
        if torch.any(source_map < 0):
            raise ValueError("source map must be nonnegative")
        mass = source_map.sum(dim=(-2, -1))
        if not torch.allclose(mass, torch.ones_like(mass), atol=2.0e-4, rtol=2.0e-4):
            raise ValueError("source map must have unit mass")
        if torch.any(mapping < 0) or torch.any(mapping >= medium.pyramid[0].shape[0]):
            raise ValueError("record_to_medium contains invalid indices")

        grid = source[:, :2].mul(2.0).sub(1.0).view(-1, 1, 1, 2)
        local_medium = F.grid_sample(
            medium.pyramid[0][mapping],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).flatten(1)
        map_field = self.map_encoder(source_map)
        local_map = F.grid_sample(
            map_field,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).flatten(1)
        hidden = self.hidden_norm(
            self.parameter_mlp(source)
            + self.local_projection(local_medium)
            + self.map_projection(local_map)
        )
        hidden = F.gelu(hidden)
        return SourceEncoding(
            hidden=hidden,
            rank=self.rank_projection(hidden),
            map_field=map_field,
            local_medium=local_medium,
        )

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        named = tuple(self.named_parameters())
        return {
            "source_parameters": tuple(p for n, p in named if n.startswith("parameter_mlp")),
            "source_map": tuple(
                p for n, p in named if n.startswith("map_encoder") or n.startswith("map_projection")
            ),
            "source_local_medium": tuple(
                p for n, p in named if n.startswith("local_projection")
            ),
            "source_rank": tuple(p for n, p in named if n.startswith("rank_projection")),
        }


__all__ = ["SourceEncoderV3", "SourceEncoding"]
