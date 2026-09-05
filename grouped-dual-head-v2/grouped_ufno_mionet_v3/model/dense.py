"""V2-informed FiLM plus learned-complex dense wavefield correction."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .medium import MediumEncoding
from .source import SourceEncoding
from .spectral import ComplexSpectralResidualBlock
from .travel_time import RayTravelTime


@dataclass(frozen=True)
class DenseGridCache:
    x_m: torch.Tensor
    z_m: torch.Tensor
    coords_xy_m: torch.Tensor
    coords_xy_normalized: torch.Tensor
    travel: RayTravelTime
    height: int
    width: int


class TimeConditionedComplexDenseDecoder(nn.Module):
    """Correct shared MIONet frames with retained V2 FiLM and complex FNO blocks."""

    def __init__(
        self,
        *,
        width: int,
        spectral_rank: int,
        pyramid_levels: int,
        dense_modes: tuple[int, ...],
        time_block: int,
        domain_t_s: float,
    ) -> None:
        super().__init__()
        if not dense_modes or time_block <= 0 or domain_t_s <= 0:
            raise ValueError("dense modes, time block, and time scale must be positive")
        self.time_block = int(time_block)
        self.domain_t_s = float(domain_t_s)
        self.multiscale_fuse = nn.Conv2d(width * pyramid_levels, width, kernel_size=1)
        self.source_projection = nn.Linear(width, width)
        self.map_projection = nn.Conv2d(width, width, kernel_size=1)
        self.coarse_lift = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(5, width),
            nn.GELU(),
            nn.Linear(width, 2 * width),
        )
        self.blocks = nn.ModuleList(
            [
                ComplexSpectralResidualBlock(
                    width=width,
                    spectral_rank=spectral_rank,
                    modes_y=mode,
                    modes_x=mode,
                )
                for mode in dense_modes
            ]
        )
        self.output = nn.Conv2d(width, 1, kernel_size=1)
        self.correction_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(
        self,
        medium: MediumEncoding,
        source: SourceEncoding,
        source_parameters: torch.Tensor,
        record_to_medium: torch.Tensor,
        time_s: torch.Tensor,
        coarse: torch.Tensor,
    ) -> torch.Tensor:
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=coarse.device)
        if coarse.ndim != 4 or time_s.shape != coarse.shape[:2]:
            raise ValueError("coarse frames must be [record,time,z,x] with matching times")
        records, count, height, width = coarse.shape
        if source_parameters.shape != (records, 5) or mapping.shape != (records,):
            raise ValueError("source or medium mapping shape does not match dense frames")
        full_size = medium.pyramid[0].shape[-2:]
        if full_size != (height, width):
            raise ValueError("dense grid must match the encoded full-resolution medium grid")
        levels = [medium.pyramid[0][mapping]]
        levels.extend(
            F.interpolate(
                level[mapping],
                size=full_size,
                mode="bilinear",
                align_corners=True,
            )
            for level in medium.pyramid[1:]
        )
        spatial = (
            self.multiscale_fuse(torch.cat(levels, dim=1))
            + self.source_projection(source.hidden)[:, :, None, None]
            + self.map_projection(source.map_field)
        )
        relative_time = time_s - source_parameters[:, None, 3]
        frequency = source_parameters[:, None, 2].expand_as(time_s)
        phase = 2.0 * math.pi * frequency * relative_time
        time_features = torch.stack(
            (
                time_s / self.domain_t_s,
                relative_time / self.domain_t_s,
                frequency / 50.0,
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=-1,
        )
        scale, bias = self.time_mlp(time_features).chunk(2, dim=-1)
        conditioned = (
            spatial[:, None]
            * (1.0 + scale[:, :, :, None, None])
            + bias[:, :, :, None, None]
        )
        flat_coarse = coarse.reshape(records * count, 1, height, width)
        hidden = conditioned.reshape(records * count, -1, height, width) + self.coarse_lift(flat_coarse)
        for block in self.blocks:
            hidden = block(hidden)
        correction = self.output(hidden.float()).reshape(records, count, height, width)
        return coarse + self.correction_scale * correction

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        named = tuple(self.named_parameters())
        return {
            "dense_spectral": tuple(p for n, p in named if "spectral.weight_" in n),
            "dense_film": tuple(p for n, p in named if n.startswith("time_mlp")),
            "dense_local": tuple(
                p
                for n, p in named
                if n.startswith("multiscale_fuse")
                or n.startswith("source_projection")
                or n.startswith("map_projection")
                or n.startswith("coarse_lift")
                or n.startswith("output")
                or n == "correction_scale"
            ),
        }


__all__ = ["DenseGridCache", "TimeConditionedComplexDenseDecoder"]
