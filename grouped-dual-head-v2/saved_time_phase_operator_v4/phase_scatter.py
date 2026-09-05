"""True-interface phase and scattering correction for the 64-bin WFP parent."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from saved_time_phase_operator_v4.wfp import _WFPBlock, _groups


class PhaseScatterCorrectionOperator(nn.Module):
    """Predict carrier-aligned and raw complex residual coefficients."""

    def __init__(
        self,
        *,
        medium_channels: int = 16,
        source_channels: int = 5,
        width: int = 32,
        rank: int = 16,
        depth: int = 4,
        radii: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()
        if len(radii) < depth:
            raise ValueError("one WFP radius is required per correction block")
        self.medium_stem = nn.Sequential(
            nn.Conv2d(medium_channels, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.source_stem = nn.Sequential(
            nn.Conv2d(source_channels, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.fuse = nn.Conv2d(3 * width, width, 1)
        self.blocks = nn.ModuleList(
            [_WFPBlock(width, rank=rank, radius=int(radii[index])) for index in range(depth)]
        )
        self.physical_head = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 4, 1),
        )
        nn.init.zeros_(self.physical_head[-1].weight)
        nn.init.zeros_(self.physical_head[-1].bias)

    def forward(
        self,
        medium: torch.Tensor,
        source: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        medium_latent = self.medium_stem(medium)
        source_latent = self.source_stem(source)
        value = self.fuse(
            torch.cat(
                (medium_latent, source_latent, medium_latent * source_latent), dim=1
            )
        )
        for block in self.blocks:
            value = block(value, scalars)
        residual = self.physical_head(value[..., :201, 20:221]).clone()
        residual[..., 0, :] = 0.0
        return residual[:, :2], residual[:, 2:]


def combine_phase_scatter(
    parent: torch.Tensor,
    phase_residual: torch.Tensor,
    scatter_residual: torch.Tensor,
    carrier: torch.Tensor,
) -> torch.Tensor:
    """Add an Eikonal-aligned residual and an unphased scattering residual."""
    if phase_residual.shape[1] != 2 or scatter_residual.shape[1] != 2:
        raise ValueError("phase/scatter residuals must be complex channel pairs")
    real, imag = phase_residual[:, 0], phase_residual[:, 1]
    c_real, c_imag = carrier[:, 0], carrier[:, 1]
    aligned = torch.stack(
        (real * c_real - imag * c_imag, real * c_imag + imag * c_real), dim=1
    )
    return parent + aligned + scatter_residual


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = ["PhaseScatterCorrectionOperator", "combine_phase_scatter", "parameter_count"]
