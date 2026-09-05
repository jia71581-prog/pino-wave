"""DG-inspired conservative latent fluxes across acoustic material interfaces."""
from __future__ import annotations

import torch
from torch import nn


def acoustic_interface_reflection_maps(
    conditioning: torch.Tensor,
    *,
    velocity_channel: int = 0,
    cpml_margin: int = 0,
    velocity_center_mps: float = 4500.0,
    velocity_scale_mps: float = 2500.0,
) -> torch.Tensor:
    """Return signed x/z face reflection maps aligned to the left/top cell.

    The current seven-channel conditioning stores
    ``(c-velocity_center_mps)/velocity_scale_mps`` in channel zero.  Output is
    ``[B,2,Z,X]``: x-face coefficients occupy ``[..., :-1]`` in channel zero,
    and z-face coefficients occupy ``[..., :-1, :]`` in channel one.  The
    registered 201 x 201 field is wholly physical, because its 20-layer CPML is
    outside that domain.  Therefore the correct saved-field default is no
    margin.  A nonzero margin remains available only to replay legacy artifacts
    that were trained with the old in-domain-CPML assumption.
    """
    value = torch.as_tensor(conditioning)
    if value.ndim != 4:
        raise ValueError("conditioning must be [B,C,Z,X]")
    channel = int(velocity_channel)
    if not 0 <= channel < value.shape[1]:
        raise ValueError("velocity_channel is outside conditioning")
    margin = int(cpml_margin)
    if margin < 0:
        raise ValueError("cpml_margin must be nonnegative")
    height, width = value.shape[-2:]
    if margin > 0 and (width <= 2 * margin + 1 or height <= margin + 1):
        raise ValueError("grid is too small for the requested CPML margin")

    velocity = (
        value[:, channel : channel + 1].float() * float(velocity_scale_mps)
        + float(velocity_center_mps)
    ).clamp_min(1.0)
    x_left, x_right = velocity[..., :-1], velocity[..., 1:]
    z_top, z_bottom = velocity[..., :-1, :], velocity[..., 1:, :]
    reflection_x = (x_right - x_left) / (x_right + x_left).clamp_min(1.0)
    reflection_z = (z_bottom - z_top) / (z_bottom + z_top).clamp_min(1.0)

    output = torch.zeros(
        value.shape[0], 2, height, width, device=value.device, dtype=torch.float32
    )
    if margin == 0:
        output[:, 0:1, :, :-1] = reflection_x
        output[:, 1:2, :-1, :] = reflection_z
    else:
        # Both cells adjoining a retained face must lie outside CPML.
        output[:, 0:1, :, margin : width - margin - 1] = reflection_x[
            ..., margin : width - margin - 1
        ]
        output[:, 1:2, : height - margin - 1, :] = reflection_z[
            ..., : height - margin - 1, :
        ]
        output[:, 1:2, :, :margin] = 0.0
        output[:, 1:2, :, width - margin :] = 0.0
    return output


class DGInterfaceFluxResidual2d(nn.Module):
    """Zero-gated conservative face update driven by velocity jumps.

    This is the Transfer DG pretraining bridge.  It treats the recurrent latent
    channels as element-local states, forms a material-weighted jump flux on
    each Cartesian face, and scatters equal/opposite contributions to adjacent
    cells.  The learned projections determine which latent combinations carry
    reflected/transmitted corrections; the fixed scatter preserves a discrete
    conservation identity before the final channel projection.
    """

    def __init__(self, width: int, *, rank: int = 16) -> None:
        super().__init__()
        channels, latent_rank = int(width), int(rank)
        if channels <= 0 or latent_rank <= 0:
            raise ValueError("DG interface width and rank must be positive")
        self.width = channels
        self.rank = latent_rank
        self.channel_in = nn.Conv2d(channels, latent_rank, kernel_size=1, bias=False)
        self.channel_out = nn.Conv2d(latent_rank, channels, kernel_size=1, bias=False)
        self.direction_scale = nn.Parameter(torch.ones(2, latent_rank))
        self.scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(
        self, value: torch.Tensor, reflection_maps: torch.Tensor
    ) -> torch.Tensor:
        if value.ndim != 4 or value.shape[1] != self.width:
            raise ValueError(f"value must be [B,{self.width},Z,X]")
        expected = (value.shape[0], 2, value.shape[-2], value.shape[-1])
        if tuple(reflection_maps.shape) != expected:
            raise ValueError(f"reflection_maps must have shape {expected}")
        with torch.autocast(device_type=value.device.type, enabled=False):
            latent = self.channel_in(value.float())
            reflected = reflection_maps.to(device=value.device, dtype=latent.dtype)
            jump_x = latent[..., 1:] - latent[..., :-1]
            flux_x = reflected[:, 0:1, :, :-1] * jump_x
            flux_x = flux_x * self.direction_scale[0][None, :, None, None]
            jump_z = latent[..., 1:, :] - latent[..., :-1, :]
            flux_z = reflected[:, 1:2, :-1, :] * jump_z
            flux_z = flux_z * self.direction_scale[1][None, :, None, None]
            divergence = torch.zeros_like(latent)
            divergence[..., :-1] += flux_x
            divergence[..., 1:] -= flux_x
            divergence[..., :-1, :] += flux_z
            divergence[..., 1:, :] -= flux_z
            return torch.tanh(self.scale) * self.channel_out(divergence)


__all__ = ["DGInterfaceFluxResidual2d", "acoustic_interface_reflection_maps"]
