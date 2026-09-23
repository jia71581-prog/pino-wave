"""Explicit four-input DeepONet/MIONet fusion with a V2 local residual."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .features import PhaseFeatureBundle
from .medium import MediumEncoding, sample_medium_pyramid
from .source import SourceEncoding
from .travel_time import RayTravelTime


@dataclass(frozen=True)
class FusionOutput:
    normalized_pressure: torch.Tensor
    coarse: torch.Tensor
    local_residual: torch.Tensor


class TravelTimeBranch(nn.Module):
    """Independent propagation branch Btau of the MIONet product."""

    def __init__(self, *, width: int, rank: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, rank),
        )

    def forward(
        self,
        travel: RayTravelTime,
        bundle: PhaseFeatureBundle,
        *,
        domain_t_s: float,
        domain_diagonal_m: float,
    ) -> torch.Tensor:
        features = torch.stack(
            (
                travel.seconds / domain_t_s,
                bundle.tau_s / domain_t_s,
                travel.distance_m / domain_diagonal_m,
                travel.path_velocity_mps / 6000.0,
                travel.endpoint_velocity_mps / 6000.0,
                travel.mean_slowness_s_per_m * 6000.0,
                torch.sin(bundle.phase_rad),
                torch.cos(bundle.phase_rad),
                bundle.causal_feature,
            ),
            dim=-1,
        )
        return self.network(features)


class PhaseAlignedMIONetFusion(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        rank: int,
        pyramid_levels: int,
        heads: int,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        local_width = width * pyramid_levels
        self.rank = int(rank)
        self.velocity_local_to_rank = nn.Linear(local_width, rank)
        self.local_to_hidden = nn.Linear(local_width, width)
        self.trunk_to_hidden = nn.Linear(rank, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.local_output = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.coarse_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.local_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # Default-off research extension: no new legacy state_dict keys.
        self.ic_to_query = None

    def enable_query_local_ic(self) -> nn.Linear:
        """Enable one zero-initialized additive IC projection before DDP setup."""
        if self.ic_to_query is None:
            reference = self.trunk_to_hidden.weight
            projection = nn.Linear(self.attention.embed_dim, self.attention.embed_dim,
                                   device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.ic_to_query = projection
        return self.ic_to_query

    def forward(
        self,
        medium: MediumEncoding,
        source: SourceEncoding,
        coords_xy_normalized: torch.Tensor,
        record_to_medium: torch.Tensor,
        travel_rank: torch.Tensor,
        trunk_rank: torch.Tensor,
    ) -> FusionOutput:
        mapping = torch.as_tensor(
            record_to_medium,
            dtype=torch.long,
            device=coords_xy_normalized.device,
        )
        local = sample_medium_pyramid(medium, coords_xy_normalized, mapping)
        velocity_rank = (
            medium.rank[mapping, None, :] + self.velocity_local_to_rank(local)
        )
        source_rank = source.rank[:, None, :]
        if travel_rank.shape != trunk_rank.shape or travel_rank.shape != velocity_rank.shape:
            raise ValueError("MIONet branch rank shapes disagree")
        product = velocity_rank * source_rank * travel_rank * trunk_rank
        coarse = self.coarse_scale * product.sum(dim=-1) / math.sqrt(self.rank)

        query = (
            self.trunk_to_hidden(trunk_rank)
            + source.hidden[:, None, :]
            + self.local_to_hidden(local)
        )
        if self.ic_to_query is not None:
            if source.ic_field is None or source.ic_field.ndim != 4 or source.ic_field.shape[:2] != (coords_xy_normalized.shape[0], self.attention.embed_dim):
                raise ValueError("query-local IC requires per-record encoded IC field")
            grid = coords_xy_normalized.mul(2.0).sub(1.0)[:, None, :, :]
            local_ic = F.grid_sample(source.ic_field, grid, mode="bilinear",
                                     padding_mode="border", align_corners=True).squeeze(2).transpose(1, 2)
            query = query + self.ic_to_query(local_ic)
        tokens = medium.tokens[mapping]
        attended, _ = self.attention(query, tokens, tokens, need_weights=False)
        local_residual = self.local_scale * self.local_output(attended + query).squeeze(-1)
        return FusionOutput(
            normalized_pressure=coarse + local_residual,
            coarse=coarse,
            local_residual=local_residual,
        )

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        named = tuple(self.named_parameters())
        return {
            "query_local_residual": tuple(
                p
                for n, p in named
                if n.startswith("local_to_hidden")
                or n.startswith("trunk_to_hidden")
                or n.startswith("attention")
                or n.startswith("local_output")
                or n == "local_scale"
            ),
            "mionet_product": tuple(
                p for n, p in named if n.startswith("velocity_local_to_rank") or n == "coarse_scale"
            ),
        }


__all__ = ["FusionOutput", "PhaseAlignedMIONetFusion", "TravelTimeBranch"]
