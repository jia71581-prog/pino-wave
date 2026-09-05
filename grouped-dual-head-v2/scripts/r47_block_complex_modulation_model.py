#!/usr/bin/env python3
"""Low-dimensional spatial complex-modulation model for R47."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


PHYSICAL_INPUT_CHANNELS = 18
DERIVED_INPUT_CHANNELS = 3


def nearest_aligned_partition_pool2d(
    value: torch.Tensor,
    output_size: tuple[int, int],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Pool nonoverlapping cells aligned with ``interpolate(..., nearest)``.

    Adaptive average pooling overlaps source cells when an input dimension is
    not divisible by the output dimension.  The 201-to-8 geometry used here is
    such a case.  Integral-image pooling keeps every source cell in exactly one
    block and uses the same ceil-spaced boundaries as PyTorch nearest-neighbor
    upsampling in the reverse direction.
    """

    if value.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(value.shape)}")
    output_height, output_width = (int(output_size[0]), int(output_size[1]))
    height, width = int(value.shape[-2]), int(value.shape[-1])
    if min(output_height, output_width) < 1:
        raise ValueError("partition output dimensions must be positive")
    if output_height > height or output_width > width:
        raise ValueError("partition output cannot exceed input geometry")
    device = value.device
    y_edges = torch.div(
        torch.arange(output_height + 1, device=device) * height
        + output_height
        - 1,
        output_height,
        rounding_mode="floor",
    )
    x_edges = torch.div(
        torch.arange(output_width + 1, device=device) * width
        + output_width
        - 1,
        output_width,
        rounding_mode="floor",
    )
    integral = F.pad(value.cumsum(dim=-2).cumsum(dim=-1), (1, 0, 1, 0))
    y0, y1 = y_edges[:-1], y_edges[1:]
    x0, x1 = x_edges[:-1], x_edges[1:]
    pooled = (
        integral[..., y1[:, None], x1[None, :]]
        - integral[..., y0[:, None], x1[None, :]]
        - integral[..., y1[:, None], x0[None, :]]
        + integral[..., y0[:, None], x0[None, :]]
    )
    if reduction == "sum":
        return pooled
    if reduction != "mean":
        raise ValueError(f"unsupported partition reduction: {reduction}")
    areas = (y1 - y0)[:, None] * (x1 - x0)[None, :]
    return pooled / areas.to(dtype=value.dtype)[None, None]


class AdaptiveMish(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.mish(10.0 * self.scale * value)


class ResidualLocalBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        groups = 8 if width % 8 == 0 else 4
        self.depthwise = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.mix = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(groups, width)
        self.activation = AdaptiveMish()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.mix(self.depthwise(value))
        return value + self.activation(self.norm(update))


class DownsampleStage(nn.Module):
    def __init__(self, input_width: int, output_width: int) -> None:
        super().__init__()
        groups = 8 if output_width % 8 == 0 else 4
        self.stage = nn.Sequential(
            nn.Conv2d(input_width, output_width, 3, stride=2, padding=1),
            nn.GroupNorm(groups, output_width),
            AdaptiveMish(),
            ResidualLocalBlock(output_width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.stage(value)


class BlockComplexGainRefiner(nn.Module):
    """Predict a small block grid of complex gains and modulate the base field.

    The output is an additive complex correction so it remains compatible with
    the R44/R46 evaluation code.  Internally it is constrained to
    ``delta_gain * base_wavefield``, matching the 8x8 oracle diagnostic.
    """

    def __init__(
        self,
        *,
        block_grid: int = 8,
        encoder_width: int = 96,
        gain_cap: float = 0.5,
        low_resolution_blocks: int = 4,
    ) -> None:
        super().__init__()
        if block_grid < 2 or encoder_width < 16 or low_resolution_blocks < 1:
            raise ValueError("invalid R47 block model configuration")
        if encoder_width % 8 != 0:
            raise ValueError("R47 encoder width must be divisible by 8")
        self.block_grid = int(block_grid)
        self.encoder_width = int(encoder_width)
        self.gain_cap = float(gain_cap)
        self.low_resolution_blocks = int(low_resolution_blocks)
        expanded = PHYSICAL_INPUT_CHANNELS + DERIVED_INPUT_CHANNELS
        self.stem = nn.Sequential(
            nn.Conv2d(expanded, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            AdaptiveMish(),
            ResidualLocalBlock(32),
        )
        self.down1 = DownsampleStage(32, 48)
        self.down2 = DownsampleStage(48, 64)
        self.down3 = DownsampleStage(64, self.encoder_width)
        self.raw_projection = nn.Sequential(
            nn.Conv2d(expanded, self.encoder_width, 1),
            nn.GroupNorm(8, self.encoder_width),
            AdaptiveMish(),
        )
        self.low_blocks = nn.Sequential(
            *[
                ResidualLocalBlock(self.encoder_width)
                for _ in range(self.low_resolution_blocks)
            ]
        )
        self.global_context = nn.Sequential(
            nn.Linear(self.encoder_width, self.encoder_width),
            nn.GELU(),
            nn.Linear(self.encoder_width, self.encoder_width),
        )
        self.head = nn.Sequential(
            nn.Conv2d(self.encoder_width, self.encoder_width // 2, 1),
            nn.GELU(),
            nn.Conv2d(self.encoder_width // 2, 2, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def expanded_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or features.shape[1] != PHYSICAL_INPUT_CHANNELS:
            raise ValueError(f"unexpected R47 feature shape: {tuple(features.shape)}")
        base = features[:, :2]
        magnitude = torch.sqrt(base.square().sum(dim=1, keepdim=True) + 1.0e-8)
        phase_unit = base / (magnitude + 0.05)
        return torch.cat((features, magnitude, phase_unit), dim=1)

    def forward(
        self, features: torch.Tensor, *, return_gain: bool = False
    ):
        expanded = self.expanded_features(features)
        encoded = self.down3(self.down2(self.down1(self.stem(expanded))))
        encoded = nearest_aligned_partition_pool2d(
            encoded, (self.block_grid, self.block_grid)
        )
        raw = self.raw_projection(
            nearest_aligned_partition_pool2d(
                expanded, (self.block_grid, self.block_grid)
            )
        )
        value = self.low_blocks(encoded + raw)
        context = self.global_context(value.mean(dim=(2, 3)))
        value = value + context[:, :, None, None]
        gain = self.gain_cap * torch.tanh(self.head(value))
        gain_full = F.interpolate(
            gain,
            size=features.shape[-2:],
            mode="nearest",
        )
        base_real = features[:, 0]
        base_imag = features[:, 1]
        gain_real = gain_full[:, 0]
        gain_imag = gain_full[:, 1]
        correction = torch.stack(
            (
                gain_real * base_real - gain_imag * base_imag,
                gain_real * base_imag + gain_imag * base_real,
            ),
            dim=1,
        )
        if return_gain:
            return correction, gain
        return correction


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
