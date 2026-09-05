"""Frequency-coupled multiscale acoustic operator with optional mHC routing."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from saved_time_phase_operator_v4.wfp import _WFPBlock, _groups


def sinkhorn(logits: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    value = torch.exp(logits - logits.max())
    for _ in range(iterations):
        value = value / value.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        value = value / value.sum(dim=-2, keepdim=True).clamp_min(1e-8)
    return value


class _UFNOConvBlock(nn.Module):
    """Dense local residual block used inside the restored U-shaped path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.body(value) + self.residual(value))


class RestoredUFNOPath(nn.Module):
    """Odd-size-safe two-level U-Net branch with full encoder/skip/decoder flow.

    The global/high-frequency path remains the learned WFP block.  This branch is
    deliberately additive: it restores the complete U-shaped local path without
    replacing any existing spectral, cross-frequency, or shallow-local path.
    """

    def __init__(self, width: int, base_channels: int | None = None) -> None:
        super().__init__()
        base = max(16, width // 8) if base_channels is None else int(base_channels)
        if base <= 0:
            raise ValueError("U-FNO base channels must be positive")
        self.input_projection = nn.Conv2d(width, base, 1)
        self.encoder_full = _UFNOConvBlock(base, base)
        self.encoder_half = _UFNOConvBlock(base, 2 * base)
        self.bottleneck = _UFNOConvBlock(2 * base, 4 * base)
        self.reduce_half = nn.Conv2d(4 * base, 2 * base, 1)
        self.decoder_half = _UFNOConvBlock(4 * base, 2 * base)
        self.reduce_full = nn.Conv2d(2 * base, base, 1)
        self.decoder_full = _UFNOConvBlock(2 * base, base)
        self.output_projection = nn.Conv2d(base, width, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        full = self.encoder_full(self.input_projection(value))
        half = self.encoder_half(
            F.avg_pool2d(full, kernel_size=2, stride=2, ceil_mode=True)
        )
        quarter = self.bottleneck(
            F.avg_pool2d(half, kernel_size=2, stride=2, ceil_mode=True)
        )
        decoded_half = F.interpolate(
            self.reduce_half(quarter),
            size=half.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_half = self.decoder_half(torch.cat((decoded_half, half), dim=1))
        decoded_full = F.interpolate(
            self.reduce_full(decoded_half),
            size=full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_full = self.decoder_full(torch.cat((decoded_full, full), dim=1))
        return self.output_projection(decoded_full)


class CoupledMultiscaleBlock(nn.Module):
    def __init__(
        self,
        width: int,
        rank: int,
        radius: int,
        *,
        restore_full_ufno: bool = False,
    ) -> None:
        super().__init__()
        self.spatial = _WFPBlock(width, rank=rank, radius=radius)
        self.frequency_depthwise = nn.Conv3d(
            width, width, (3, 1, 1), padding=(1, 0, 0), groups=width
        )
        self.frequency_pointwise = nn.Conv3d(width, width, 1)
        self.local_down = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.local_mid = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
        )
        self.local_scale = nn.Parameter(torch.tensor(0.1))
        self.frequency_scale = nn.Parameter(torch.tensor(0.1))
        self.restored_ufno = RestoredUFNOPath(width) if restore_full_ufno else None
        self.restored_ufno_scale = (
            nn.Parameter(torch.tensor(0.1)) if restore_full_ufno else None
        )

    def forward(self, value: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        batch, frequencies, channels, height, width = value.shape
        flat = value.reshape(batch * frequencies, channels, height, width)
        scalar_flat = scalars.reshape(batch * frequencies, scalars.shape[-1])
        spatial = self.spatial(flat, scalar_flat) - flat
        local = self.local_mid(self.local_down(flat))
        local = F.interpolate(local, size=(height, width), mode="bilinear", align_corners=False)
        spatial = spatial + self.local_scale * local
        if self.restored_ufno is not None:
            spatial = spatial + self.restored_ufno_scale * self.restored_ufno(flat)
        coupled = value.permute(0, 2, 1, 3, 4)
        coupled = self.frequency_pointwise(self.frequency_depthwise(coupled))
        coupled = coupled.permute(0, 2, 1, 3, 4)
        return spatial.reshape_as(value) + self.frequency_scale * coupled


class MHCBlock(nn.Module):
    def __init__(self, block: CoupledMultiscaleBlock, streams: int = 2) -> None:
        super().__init__()
        self.block = block
        self.streams = streams
        mixing = torch.full((streams, streams), -4.0)
        mixing.fill_diagonal_(4.0)
        self.mixing_logits = nn.Parameter(mixing)
        pre = torch.full((streams,), -4.0)
        pre[0] = 4.0
        self.pre_logits = nn.Parameter(pre)
        self.post_logits = nn.Parameter(torch.zeros(streams))

    def forward(self, state: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        matrix = sinkhorn(self.mixing_logits)
        if self.streams == 2:
            mixed = torch.stack(
                (
                    matrix[0, 0] * state[:, :, 0] + matrix[0, 1] * state[:, :, 1],
                    matrix[1, 0] * state[:, :, 0] + matrix[1, 1] * state[:, :, 1],
                ),
                dim=2,
            )
            pre_weight = torch.softmax(self.pre_logits, 0)
            pre = pre_weight[0] * state[:, :, 0] + pre_weight[1] * state[:, :, 1]
        else:
            mixed = torch.einsum("ij,bfjchw->bfichw", matrix, state)
            pre = torch.einsum(
                "i,bfichw->bfchw", torch.softmax(self.pre_logits, 0), state
            )
        update = self.block(pre, scalars)
        post = torch.softmax(self.post_logits, 0)
        return mixed + update[:, :, None] * post[None, None, :, None, None, None]


class CoupledMHCWaveOperator(nn.Module):
    def __init__(
        self,
        *,
        use_mhc: bool,
        medium_channels: int = 16,
        source_channels: int = 5,
        width: int = 192,
        rank: int = 96,
        depth: int = 16,
        streams: int = 2,
        trunk_channels: int = 14,
        branch_trunk_rank: int = 64,
        restored_ufno_blocks: int = 4,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.use_mhc = bool(use_mhc)
        self.streams = int(streams)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.medium_stem = nn.Sequential(
            nn.Conv2d(medium_channels, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width), nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.source_stem = nn.Sequential(
            nn.Conv2d(source_channels, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width), nn.GELU(),
            nn.Conv2d(width, width, 1),
        )
        self.fuse = nn.Conv2d(3 * width, width, 1)
        self.branch_context = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 12 * branch_trunk_rank),
        )
        self.medium_mionet_branch = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 12 * branch_trunk_rank),
        )
        self.source_mionet_branch = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 12 * branch_trunk_rank),
        )
        self.trunk_basis = nn.Sequential(
            nn.Conv2d(trunk_channels, width // 2, 3, padding=1),
            nn.GroupNorm(_groups(width // 2), width // 2),
            nn.GELU(),
            nn.Conv2d(width // 2, branch_trunk_rank, 1),
        )
        self.branch_trunk_rank = int(branch_trunk_rank)
        self.branch_trunk_scale = nn.Parameter(torch.full((12,), 0.1))
        self.mionet_scale = nn.Parameter(torch.full((12,), 0.1))
        if not 0 <= int(restored_ufno_blocks) <= int(depth):
            raise ValueError("restored U-FNO block count must be between zero and depth")
        self.restored_ufno_blocks = int(restored_ufno_blocks)
        radii = (1, 2, 3, 4) * ((depth + 3) // 4)
        restore_from = depth - self.restored_ufno_blocks
        blocks = [
            CoupledMultiscaleBlock(
                width,
                rank,
                radii[index],
                restore_full_ufno=index >= restore_from,
            )
            for index in range(depth)
        ]
        self.blocks = nn.ModuleList(
            [MHCBlock(block, streams=streams) for block in blocks]
            if self.use_mhc else blocks
        )
        if self.use_mhc:
            collapse = torch.full((streams,), -4.0)
            collapse[0] = 4.0
            self.collapse_logits = nn.Parameter(collapse)
        else:
            self.collapse_logits = None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260903)
            self.physical_head = nn.Sequential(
                nn.GroupNorm(_groups(width), width), nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                nn.Conv2d(width, 2, 1),
            )
            self.cpml_head = nn.Sequential(
                nn.GroupNorm(_groups(width), width), nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                nn.Conv2d(width, 10, 1),
            )

    def _run_block(self, block: nn.Module, value: torch.Tensor, scalars: torch.Tensor):
        if self.activation_checkpointing and self.training:
            return checkpoint(block, value, scalars, use_reentrant=False)
        return block(value, scalars)

    def forward(
        self,
        medium: torch.Tensor,
        source: torch.Tensor,
        scalars: torch.Tensor,
        trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if medium.ndim != 5 or source.ndim != 5 or scalars.ndim != 3 or trunk.ndim != 5:
            raise ValueError("coupled operator expects [B,F,C,Z,X], [B,F,5], and trunk")
        batch, frequencies = medium.shape[:2]
        m = self.medium_stem(medium.reshape(batch * frequencies, *medium.shape[2:]))
        s = self.source_stem(source.reshape(batch * frequencies, *source.shape[2:]))
        value = self.fuse(torch.cat((m, s, m * s), dim=1))
        value = value.reshape(batch, frequencies, *value.shape[1:])
        if self.use_mhc:
            state = torch.cat(
                (
                    value[:, :, None],
                    torch.zeros_like(value[:, :, None]).expand(
                        -1, -1, self.streams - 1, -1, -1, -1
                    ),
                ),
                dim=2,
            )
            for block in self.blocks:
                state = self._run_block(block, state, scalars)
            collapse = torch.softmax(self.collapse_logits, 0)
            if self.streams == 2:
                value = collapse[0] * state[:, :, 0] + collapse[1] * state[:, :, 1]
            else:
                value = torch.einsum("i,bfichw->bfchw", collapse, state)
        else:
            for block in self.blocks:
                value = value + self._run_block(block, value, scalars)
        flat = value.reshape(batch * frequencies, *value.shape[2:])
        pooled_medium = F.adaptive_avg_pool2d(m, 1).flatten(1)
        pooled_source = F.adaptive_avg_pool2d(s, 1).flatten(1)
        pooled = torch.cat(
            (
                pooled_medium,
                pooled_source,
                F.adaptive_avg_pool2d(m * s, 1).flatten(1),
            ),
            dim=1,
        )
        coefficients = self.branch_context(pooled).reshape(
            batch * frequencies, 12, self.branch_trunk_rank
        )
        basis = self.trunk_basis(
            trunk.reshape(batch * frequencies, *trunk.shape[2:])
        )
        branch_trunk = torch.einsum("bcr,brhw->bchw", coefficients, basis)
        branch_trunk = branch_trunk * self.branch_trunk_scale[None, :, None, None]
        medium_coefficients = self.medium_mionet_branch(pooled_medium).reshape(
            batch * frequencies, 12, self.branch_trunk_rank
        )
        source_coefficients = self.source_mionet_branch(pooled_source).reshape(
            batch * frequencies, 12, self.branch_trunk_rank
        )
        mionet = torch.einsum(
            "bcr,bcr,brhw->bchw",
            medium_coefficients,
            source_coefficients,
            basis,
        ) / self.branch_trunk_rank**0.5
        branch_trunk = branch_trunk + mionet * self.mionet_scale[None, :, None, None]
        physical = (
            self.physical_head(flat) + branch_trunk[:, :2]
        )[..., :201, 20:221]
        cpml = self.cpml_head(flat) + branch_trunk[:, 2:]
        physical = physical.reshape(batch, frequencies, 2, 201, 201).clone()
        cpml = cpml.reshape(batch, frequencies, 10, 221, 241).clone()
        physical[..., 0, :] = 0.0
        cpml[:, :, :2, 0, :] = 0.0
        cpml[:, :, :2, -1, :] = 0.0
        cpml[:, :, :2, :, 0] = 0.0
        cpml[:, :, :2, :, -1] = 0.0
        return physical, cpml


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


__all__ = [
    "CoupledMHCWaveOperator",
    "RestoredUFNOPath",
    "parameter_count",
    "sinkhorn",
]
