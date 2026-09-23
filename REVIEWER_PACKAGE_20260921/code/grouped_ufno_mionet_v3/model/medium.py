"""Position-aware multiscale learned-complex velocity encoder."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .spectral import ComplexSpectralResidualBlock


@dataclass(frozen=True)
class MediumEncoding:
    pyramid: tuple[torch.Tensor, ...]
    tokens: torch.Tensor
    token_positions: torch.Tensor
    rank: torch.Tensor


def _position_features(
    height: int,
    width: int,
    *,
    bands: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.linspace(0.0, 1.0, height, dtype=dtype, device=device)
    x = torch.linspace(0.0, 1.0, width, dtype=dtype, device=device)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    features = [xx, zz]
    for index in range(bands):
        frequency = (2.0**index) * math.pi
        features.extend(
            (
                torch.sin(frequency * xx),
                torch.cos(frequency * xx),
                torch.sin(frequency * zz),
                torch.cos(frequency * zz),
            )
        )
    channels = torch.stack(features, dim=0)
    positions = torch.stack((xx, zz), dim=-1)
    return channels, positions


class ComplexMediumEncoder(nn.Module):
    def __init__(
        self,
        *,
        width: int = 64,
        rank: int = 128,
        spectral_rank: int = 40,
        modes: tuple[int, ...] = (20, 16, 12, 8),
        token_grid: int = 8,
        position_bands: int = 4,
    ) -> None:
        super().__init__()
        if not modes or token_grid <= 0 or position_bands <= 0:
            raise ValueError("modes, token_grid, and position_bands must be positive")
        self.width = int(width)
        self.rank_size = int(rank)
        self.token_grid = int(token_grid)
        self.position_bands = int(position_bands)
        position_channels = 2 + 4 * self.position_bands
        self.lift = nn.Conv2d(1 + position_channels, self.width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                ComplexSpectralResidualBlock(
                    width=self.width,
                    spectral_rank=spectral_rank,
                    modes_y=mode,
                    modes_x=mode,
                )
                for mode in modes
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                )
                for _ in modes[:-1]
            ]
        )
        token_position_channels = 2 + 4 * self.position_bands
        self.token_position_projection = nn.Linear(token_position_channels, self.width)
        self.token_norm = nn.LayerNorm(self.width)
        self.rank_projection = nn.Linear(self.width, self.rank_size)

    def forward(self, velocity_normalized: torch.Tensor) -> MediumEncoding:
        velocity = torch.as_tensor(velocity_normalized)
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity input must be [medium,1,z,x]")
        if not velocity.dtype.is_floating_point:
            velocity = velocity.float()
        position, _ = _position_features(
            velocity.shape[-2],
            velocity.shape[-1],
            bands=self.position_bands,
            dtype=velocity.dtype,
            device=velocity.device,
        )
        position = position[None].expand(velocity.shape[0], -1, -1, -1)
        hidden = self.lift(torch.cat((velocity, position), dim=1))
        pyramid: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            pyramid.append(hidden)
            if index < len(self.down):
                hidden = self.down[index](hidden)

        pooled = F.adaptive_avg_pool2d(hidden, (self.token_grid, self.token_grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        token_features, token_positions_grid = _position_features(
            self.token_grid,
            self.token_grid,
            bands=self.position_bands,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        token_position_features = token_features.flatten(1).transpose(0, 1)
        tokens = self.token_norm(tokens + self.token_position_projection(token_position_features)[None])
        token_positions = token_positions_grid.reshape(1, -1, 2).expand(tokens.shape[0], -1, -1)
        return MediumEncoding(
            pyramid=tuple(pyramid),
            tokens=tokens,
            token_positions=token_positions,
            rank=self.rank_projection(tokens.mean(dim=1)),
        )

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        named = tuple(self.named_parameters())
        return {
            "medium_spectral": tuple(p for n, p in named if "spectral.weight_" in n),
            "medium_local": tuple(
                p
                for n, p in named
                if n.startswith("lift") or n.startswith("down") or "local_" in n
            ),
            "medium_tokens": tuple(
                p for n, p in named if n.startswith("token_position") or n.startswith("token_norm")
            ),
            "medium_rank": tuple(p for n, p in named if n.startswith("rank_projection")),
        }


def sample_medium_pyramid(
    medium: MediumEncoding,
    coords_xy_normalized: torch.Tensor,
    record_to_medium: torch.Tensor,
) -> torch.Tensor:
    coords = torch.as_tensor(coords_xy_normalized)
    mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=coords.device)
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("normalized coordinates must be [record,query,2]")
    if mapping.shape != (coords.shape[0],):
        raise ValueError("record_to_medium shape does not match coordinates")
    medium_count = medium.pyramid[0].shape[0]
    if torch.any(mapping < 0) or torch.any(mapping >= medium_count):
        raise ValueError("record_to_medium contains invalid indices")
    grid = coords.mul(2.0).sub(1.0)[:, None, :, :]
    sampled: list[torch.Tensor] = []
    for level in medium.pyramid:
        values = F.grid_sample(
            level[mapping],
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled.append(values.squeeze(2).transpose(1, 2))
    return torch.cat(sampled, dim=-1)


__all__ = ["ComplexMediumEncoder", "MediumEncoding", "sample_medium_pyramid"]

