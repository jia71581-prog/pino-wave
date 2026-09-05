"""Persistent medium-pyramid and softly routed expert acoustic operator."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from grouped_ufno_mionet_v3.model.spectral import ComplexSpectralResidualBlock
from saved_time_phase_operator_v4.coupled_mhc_wave import CoupledMHCWaveOperator
from saved_time_phase_operator_v4.wfp import _groups


class PersistentComplexMediumPyramid(nn.Module):
    """Four retained complex-spectral medium levels with physical coordinates."""

    def __init__(
        self,
        *,
        medium_channels: int = 16,
        width: int = 48,
        spectral_rank: int = 24,
        modes: tuple[int, ...] = (20, 16, 12, 8),
        position_bands: int = 2,
    ) -> None:
        super().__init__()
        if len(modes) != 4 or position_bands <= 0:
            raise ValueError("the persistent medium pyramid requires four positive levels")
        self.width = int(width)
        self.position_bands = int(position_bands)
        position_channels = 2 + 4 * self.position_bands
        self.lift = nn.Conv2d(
            int(medium_channels) + position_channels,
            self.width,
            3,
            padding=1,
        )
        self.blocks = nn.ModuleList(
            [
                ComplexSpectralResidualBlock(
                    width=self.width,
                    spectral_rank=int(spectral_rank),
                    modes_y=int(mode),
                    modes_x=int(mode),
                )
                for mode in modes
            ]
        )
        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.width, self.width, 3, stride=2, padding=1),
                    nn.GroupNorm(_groups(self.width), self.width),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )

    def _position_features(self, value: torch.Tensor) -> torch.Tensor:
        height, width = value.shape[-2:]
        z = torch.linspace(0.0, 1.1, height, dtype=value.dtype, device=value.device)
        x = torch.linspace(-0.1, 1.1, width, dtype=value.dtype, device=value.device)
        zz, xx = torch.meshgrid(z, x, indexing="ij")
        features = [xx, zz]
        for index in range(self.position_bands):
            angular = (2.0**index) * math.pi
            features.extend(
                (
                    torch.sin(angular * xx),
                    torch.cos(angular * xx),
                    torch.sin(angular * zz),
                    torch.cos(angular * zz),
                )
            )
        return torch.stack(features, dim=0)[None].expand(value.shape[0], -1, -1, -1)

    def forward(self, medium: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if medium.ndim != 4:
            raise ValueError("pyramid medium must be [batch,channel,z,x]")
        hidden = self.lift(torch.cat((medium, self._position_features(medium)), dim=1))
        levels = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden)
            levels.append(hidden)
            if index < len(self.down):
                hidden = self.down[index](hidden)
        return tuple(levels)


class _DenseResidualBlock(nn.Module):
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


class SoftTop2PyramidExperts(nn.Module):
    """Two always-on common experts plus four input-routed residual experts."""

    def __init__(
        self,
        channels: int,
        *,
        shared_experts: int = 2,
        routed_experts: int = 4,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if not 0 < top_k <= routed_experts:
            raise ValueError("expert top_k must be positive and no larger than expert count")
        self.routed_expert_count = int(routed_experts)
        self.top_k = int(top_k)
        self.shared = nn.ModuleList(
            [_DenseResidualBlock(channels, channels) for _ in range(shared_experts)]
        )
        self.routed = nn.ModuleList(
            [_DenseResidualBlock(channels, channels) for _ in range(routed_experts)]
        )
        self.router = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, routed_experts),
        )
        self.shared_scale = nn.Parameter(torch.tensor(0.1))
        self.routed_scale = nn.Parameter(torch.tensor(0.1))
        self._last_probabilities: torch.Tensor | None = None
        self._last_mask: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool2d(value, 1).flatten(1)
        probabilities = torch.softmax(self.router(pooled).float(), dim=-1)
        _, indices = torch.topk(probabilities, self.top_k, dim=-1)
        mask = torch.zeros_like(probabilities).scatter_(1, indices, 1.0)
        self._last_probabilities = probabilities.detach()
        self._last_mask = mask.detach()
        gates = probabilities * mask
        gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

        shared = torch.stack([expert(value) for expert in self.shared], dim=0).mean(0)
        routed_values = torch.stack([expert(value) for expert in self.routed], dim=1)
        routed = (routed_values * gates[:, :, None, None, None]).sum(dim=1)

        importance = probabilities.mean(dim=0)
        load = mask.mean(dim=0) / float(self.top_k)
        balance = self.routed_expert_count * (importance * load.detach()).sum()
        normalized_entropy = -(
            probabilities.clamp_min(1.0e-8).log() * probabilities
        ).sum(dim=-1).mean() / math.log(self.routed_expert_count)
        router_auxiliary = balance + 0.1 * (1.0 - normalized_entropy)
        update = value + self.shared_scale * shared + self.routed_scale * routed
        return update, router_auxiliary

    def route_statistics(self) -> dict[str, torch.Tensor]:
        if self._last_probabilities is None or self._last_mask is None:
            return {}
        probabilities = self._last_probabilities
        mask = self._last_mask
        entropy = -(
            probabilities.clamp_min(1.0e-8).log() * probabilities
        ).sum(dim=-1).mean() / math.log(self.routed_expert_count)
        result = {"route_entropy": entropy}
        for index in range(self.routed_expert_count):
            result[f"route_probability_{index}"] = probabilities[:, index].mean()
            result[f"route_load_{index}"] = mask[:, index].mean() / float(self.top_k)
        return result


class RoutedPyramidDecoder(nn.Module):
    """Decode all four retained levels after source/frequency-conditioned routing."""

    def __init__(
        self,
        *,
        pyramid_width: int = 48,
        output_width: int = 192,
        source_channels: int = 5,
        scalar_channels: int = 5,
        token_grid: int = 8,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if pyramid_width % attention_heads:
            raise ValueError("pyramid width must be divisible by attention heads")
        self.pyramid_width = int(pyramid_width)
        self.token_grid = int(token_grid)
        self.source_projection = nn.Conv2d(source_channels, pyramid_width, 1)
        self.scalar_projection = nn.Sequential(
            nn.Linear(scalar_channels, pyramid_width),
            nn.GELU(),
            nn.Linear(pyramid_width, pyramid_width),
        )
        self.experts = SoftTop2PyramidExperts(pyramid_width)
        self.query_norm = nn.LayerNorm(pyramid_width)
        self.token_norm = nn.LayerNorm(pyramid_width)
        self.token_attention = nn.MultiheadAttention(
            pyramid_width,
            attention_heads,
            batch_first=True,
        )
        self.attention_scale = nn.Parameter(torch.tensor(0.1))
        self.decode_quarter = _DenseResidualBlock(2 * pyramid_width, pyramid_width)
        self.decode_half = _DenseResidualBlock(2 * pyramid_width, pyramid_width)
        self.decode_full = _DenseResidualBlock(2 * pyramid_width, pyramid_width)
        self.output_projection = nn.Conv2d(pyramid_width, output_width, 1)

    @staticmethod
    def _repeat_level(level: torch.Tensor, frequencies: int) -> torch.Tensor:
        return level[:, None].expand(-1, frequencies, -1, -1, -1).reshape(
            level.shape[0] * frequencies, *level.shape[1:]
        )

    def forward(
        self,
        full: torch.Tensor,
        half: torch.Tensor,
        quarter: torch.Tensor,
        eighth: torch.Tensor,
        source: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.ndim != 5 or scalars.ndim != 3:
            raise ValueError("pyramid decoder source/scalars must be frequency blocked")
        batch, frequencies = source.shape[:2]
        flat_source = source.reshape(batch * frequencies, *source.shape[2:])
        levels = [
            self._repeat_level(level, frequencies)
            for level in (full, half, quarter, eighth)
        ]
        source_quarter = self.source_projection(
            F.interpolate(
                flat_source,
                size=levels[2].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        source_eighth = self.source_projection(
            F.interpolate(
                flat_source,
                size=levels[3].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
        scalar = self.scalar_projection(scalars.reshape(batch * frequencies, -1))
        levels[2] = levels[2] + source_quarter
        levels[3] = levels[3] + source_eighth + scalar[:, :, None, None]
        levels[3], router_auxiliary = self.experts(levels[3])

        tokens = F.adaptive_avg_pool2d(
            levels[3], (self.token_grid, self.token_grid)
        ).flatten(2).transpose(1, 2)
        queries = levels[2].flatten(2).transpose(1, 2)
        attended, _ = self.token_attention(
            self.query_norm(queries),
            self.token_norm(tokens),
            self.token_norm(tokens),
            need_weights=False,
        )
        levels[2] = levels[2] + self.attention_scale * attended.transpose(1, 2).reshape_as(levels[2])

        hidden = self.decode_quarter(
            torch.cat(
                (
                    F.interpolate(
                        levels[3],
                        size=levels[2].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    levels[2],
                ),
                dim=1,
            )
        )
        hidden = self.decode_half(
            torch.cat(
                (
                    F.interpolate(
                        hidden,
                        size=levels[1].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    levels[1],
                ),
                dim=1,
            )
        )
        hidden = self.decode_full(
            torch.cat(
                (
                    F.interpolate(
                        hidden,
                        size=levels[0].shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ),
                    levels[0],
                ),
                dim=1,
            )
        )
        return self.output_projection(hidden), router_auxiliary


class PyramidMoECoupledWaveOperator(CoupledMHCWaveOperator):
    """Add a retained medium pyramid and conditioned experts to the full model."""

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
        pyramid_width: int = 48,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__(
            use_mhc=use_mhc,
            medium_channels=medium_channels,
            source_channels=source_channels,
            width=width,
            rank=rank,
            depth=depth,
            streams=streams,
            trunk_channels=trunk_channels,
            branch_trunk_rank=branch_trunk_rank,
            restored_ufno_blocks=restored_ufno_blocks,
            activation_checkpointing=activation_checkpointing,
        )
        self.medium_pyramid = PersistentComplexMediumPyramid(
            medium_channels=medium_channels,
            width=pyramid_width,
        )
        self.pyramid_decoder = RoutedPyramidDecoder(
            pyramid_width=pyramid_width,
            output_width=width,
            source_channels=source_channels,
        )
        self.pyramid_context = nn.Sequential(
            nn.Linear(4 * pyramid_width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        self.pyramid_field_scale = nn.Parameter(torch.tensor(0.1))
        self.pyramid_context_scale = nn.Parameter(torch.tensor(0.1))
        self._router_auxiliary: torch.Tensor | None = None

    def _base_encode(
        self,
        medium: torch.Tensor,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_medium = self.medium_stem(medium)
        encoded_source = self.source_stem(source)
        interaction = encoded_medium * encoded_source
        value = self.fuse(torch.cat((encoded_medium, encoded_source, interaction), dim=1))
        return (
            value,
            F.adaptive_avg_pool2d(encoded_medium, 1).flatten(1),
            F.adaptive_avg_pool2d(encoded_source, 1).flatten(1),
            F.adaptive_avg_pool2d(interaction, 1).flatten(1),
        )

    def _run_base_encode(self, medium: torch.Tensor, source: torch.Tensor):
        if self.activation_checkpointing and self.training:
            return checkpoint(self._base_encode, medium, source, use_reentrant=False)
        return self._base_encode(medium, source)

    def _run_medium_pyramid(self, medium: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.activation_checkpointing and self.training:
            return checkpoint(self.medium_pyramid, medium, use_reentrant=False)
        return self.medium_pyramid(medium)

    def _run_pyramid_decoder(
        self,
        levels: tuple[torch.Tensor, ...],
        source: torch.Tensor,
        scalars: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation_checkpointing and self.training:
            return checkpoint(
                self.pyramid_decoder,
                *levels,
                source,
                scalars,
                use_reentrant=False,
            )
        return self.pyramid_decoder(*levels, source, scalars)

    def router_auxiliary_loss(self) -> torch.Tensor:
        if self._router_auxiliary is None:
            return next(self.parameters()).new_zeros(())
        return self._router_auxiliary

    def route_statistics(self) -> dict[str, torch.Tensor]:
        return self.pyramid_decoder.experts.route_statistics()

    def forward(
        self,
        medium: torch.Tensor,
        source: torch.Tensor,
        scalars: torch.Tensor,
        trunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if medium.ndim != 5 or source.ndim != 5 or scalars.ndim != 3 or trunk.ndim != 5:
            raise ValueError("pyramid-MoE operator expects frequency-blocked fields")
        batch, frequencies = medium.shape[:2]
        medium_flat = medium.reshape(batch * frequencies, *medium.shape[2:])
        source_flat = source.reshape(batch * frequencies, *source.shape[2:])
        value, pooled_medium, pooled_source, pooled_interaction = self._run_base_encode(
            medium_flat,
            source_flat,
        )

        pyramid = self._run_medium_pyramid(medium[:, 0])
        pyramid_field, router_auxiliary = self._run_pyramid_decoder(
            pyramid,
            source,
            scalars,
        )
        self._router_auxiliary = router_auxiliary
        value = value + self.pyramid_field_scale * pyramid_field
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

        pyramid_context = torch.cat(
            [F.adaptive_avg_pool2d(level, 1).flatten(1) for level in pyramid],
            dim=1,
        )
        pyramid_context = self.pyramid_context(pyramid_context)
        pyramid_context = pyramid_context[:, None].expand(-1, frequencies, -1).reshape(
            batch * frequencies, -1
        )
        pooled_medium = pooled_medium + self.pyramid_context_scale * pyramid_context
        pooled = torch.cat((pooled_medium, pooled_source, pooled_interaction), dim=1)
        coefficients = self.branch_context(pooled).reshape(
            batch * frequencies, 12, self.branch_trunk_rank
        )
        basis = self.trunk_basis(trunk.reshape(batch * frequencies, *trunk.shape[2:]))
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

        physical = (self.physical_head(flat) + branch_trunk[:, :2])[..., :201, 20:221]
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
    "PersistentComplexMediumPyramid",
    "PyramidMoECoupledWaveOperator",
    "RoutedPyramidDecoder",
    "SoftTop2PyramidExperts",
    "parameter_count",
]
