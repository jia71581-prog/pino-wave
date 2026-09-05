"""Propagation-conditioned factorized spectral full-wavefield decoder."""
from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.source import SourceEncoding
from grouped_ufno_mionet_v3.model.travel_time import RayTravelTime

from .band_adapter import BandLimitedFamilyAdapter
from .features import dense_propagation_features
from .experts import FamilyRoutedResidualExperts
from .spectral import FactorizedComplexResidualStack


class QueryInvariantTemporalBasis(nn.Module):
    """Low-rank DeepONet residual whose value is independent of query batching."""

    def __init__(self, *, width: int, rank: int) -> None:
        super().__init__()
        if int(width) <= 0 or int(rank) <= 0:
            raise ValueError("temporal basis width and rank must be positive")
        self.rank = int(rank)
        self.coefficient = nn.Conv2d(int(width), self.rank, kernel_size=1)
        self.time_trunk = nn.Sequential(
            nn.Linear(5, int(width)),
            nn.GELU(),
            nn.Linear(int(width), self.rank),
        )
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def feature_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.coefficient.parameters()) + tuple(self.time_trunk.parameters())

    @torch.no_grad()
    def absorb_gate_into_coefficient(self) -> dict[str, float]:
        """Preserve the function while removing a small multiplicative gate."""

        gate = float(self.gate.detach())
        if not math.isfinite(gate):
            raise FloatingPointError("temporal basis gate must be finite before absorption")
        pre_weight_norm = float(self.coefficient.weight.detach().float().norm())
        pre_bias_norm = float(self.coefficient.bias.detach().float().norm())
        self.coefficient.weight.mul_(gate)
        self.coefficient.bias.mul_(gate)
        self.gate.fill_(1.0)
        report = {
            "pre_gate": gate,
            "post_gate": float(self.gate.detach()),
            "pre_coefficient_weight_norm": pre_weight_norm,
            "post_coefficient_weight_norm": float(
                self.coefficient.weight.detach().float().norm()
            ),
            "pre_coefficient_bias_norm": pre_bias_norm,
            "post_coefficient_bias_norm": float(
                self.coefficient.bias.detach().float().norm()
            ),
        }
        if not all(math.isfinite(value) for value in report.values()):
            raise FloatingPointError("temporal basis gate absorption produced non-finite state")
        return report

    def forward(
        self,
        spatial: torch.Tensor,
        time_features: torch.Tensor,
        *,
        spatial_stack: nn.Module,
    ) -> torch.Tensor:
        if spatial.ndim != 4 or time_features.ndim != 3:
            raise ValueError("temporal basis expects [record,channel,z,x] and [record,time,5]")
        if spatial.shape[0] != time_features.shape[0] or time_features.shape[-1] != 5:
            raise ValueError("temporal basis record or feature dimensions do not match")
        coefficient = self.coefficient(spatial_stack(spatial).float())
        basis = self.time_trunk(time_features.float())
        residual = torch.einsum("btk,bkzx->btzx", basis, coefficient)
        residual = residual / math.sqrt(float(self.rank))
        return self.gate * residual


class _HighFrequencyResidualHead(nn.Module):
    """Zero-initialized local-convolution head for high spatial-frequency detail.

    The factorized spectral corrector (``stack``) is band-limited to ``modes``
    spatial frequencies, so sharp wavefronts / coda above that cutoff are never
    corrected — the measured ``spectrum.high`` residual (~0.58).  A stack of small
    3x3 convolutions has no spectral truncation and represents exactly those high
    frequencies.  The final conv is zero-initialized (ControlNet style), so a
    warm-started model reproduces its prediction exactly at load and this branch
    fades in only as training uses it.
    """

    def __init__(self, width: int, *, hidden: int, depth: int) -> None:
        super().__init__()
        if width <= 0 or hidden <= 0 or depth < 1:
            raise ValueError("high-frequency head width/hidden/depth must be positive")
        layers: list[nn.Module] = [nn.Conv2d(width, hidden, kernel_size=3, padding=1), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.GELU()]
        self.body = nn.Sequential(*layers)
        self.output = nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, shared: torch.Tensor) -> torch.Tensor:
        return self.output(self.body(shared))


class PropagationConditionedDenseDecoder(nn.Module):
    """Refine coarse MIONet fields at exact saved times."""

    def __init__(
        self,
        *,
        width: int,
        spectral_rank: int,
        pyramid_levels: int,
        modes: int,
        depth: int,
        saved_time_count: int,
        time_block: int,
        domain_t_s: float,
        domain_diagonal_m: float,
        use_local_phase: bool,
        activation_checkpointing: bool,
        coupled_axes: bool = False,
        local_differential_residual: bool = False,
        coupled_2d_rank: int = 0,
        temporal_basis_rank: int = 0,
        family_expert_rank: int = 0,
        band_adapter_rank: int = 0,
        band_adapter_architecture: str = "low_rank",
        band_adapter_spectral_rank: int = 32,
        band_adapter_modes: int = 32,
        band_adapter_full_depth: int = 4,
        band_adapter_coarse_depth: int = 2,
        band_adapter_activation_checkpointing: bool = True,
        band_adapter_dropout: float = 0.0,
        gabor_scales_s: Sequence[float] = (0.01, 0.025, 0.05, 0.1),
        high_frequency_residual: bool = False,
        high_frequency_hidden: int = 64,
        high_frequency_depth: int = 3,
    ) -> None:
        super().__init__()
        if pyramid_levels <= 0 or saved_time_count <= 1 or time_block <= 0:
            raise ValueError("V4 decoder pyramid, time count and block must be positive")
        self.time_block = int(time_block)
        self.domain_t_s = float(domain_t_s)
        self.domain_diagonal_m = float(domain_diagonal_m)
        self.use_local_phase = bool(use_local_phase)
        self.gabor_scales_s = tuple(float(value) for value in gabor_scales_s)
        self.multiscale_fuse = nn.Conv2d(width * pyramid_levels, width, kernel_size=1)
        self.source_projection = nn.Linear(width, width)
        self.map_projection = nn.Conv2d(width, width, kernel_size=1)
        self.coarse_lift = nn.Conv2d(1, width, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), nn.Linear(width, 2 * width)
        )
        self.saved_time_embedding = nn.Embedding(saved_time_count, width)
        self.phase_projection = (
            nn.Conv2d(12, width, kernel_size=1) if self.use_local_phase else None
        )
        self.stack = FactorizedComplexResidualStack(
            width,
            spectral_rank,
            modes,
            depth,
            activation_checkpointing=activation_checkpointing,
            coupled_axes=bool(coupled_axes),
            local_differential_residual=bool(local_differential_residual),
            coupled_2d_rank=int(coupled_2d_rank),
        )
        temporal_rank = int(temporal_basis_rank)
        if temporal_rank < 0:
            raise ValueError("temporal basis rank must be nonnegative")
        self.temporal_basis = (
            None
            if temporal_rank == 0
            else QueryInvariantTemporalBasis(width=width, rank=temporal_rank)
        )
        expert_rank = int(family_expert_rank)
        if expert_rank < 0:
            raise ValueError("family expert rank must be nonnegative")
        self.family_experts = (
            None
            if expert_rank == 0
            else FamilyRoutedResidualExperts(width=width, rank=expert_rank)
        )
        adapter_rank = int(band_adapter_rank)
        if adapter_rank < 0:
            raise ValueError("band adapter rank must be nonnegative")
        if (
            adapter_rank > 0
            and self.family_experts is None
            and str(band_adapter_architecture)
            != "shared_dynamic_multiscale_spectral"
        ):
            raise ValueError("band adapter requires the registered family router")
        if (
            adapter_rank > 0
            and self.family_experts is not None
            and str(band_adapter_architecture)
            == "shared_dynamic_multiscale_spectral"
        ):
            raise ValueError("shared dynamic band adapter requires no family router")
        self.band_limited_adapter = (
            None
            if adapter_rank == 0
            else BandLimitedFamilyAdapter(
                width=width,
                rank=adapter_rank,
                architecture=band_adapter_architecture,
                spectral_rank=band_adapter_spectral_rank,
                modes=band_adapter_modes,
                full_depth=band_adapter_full_depth,
                coarse_depth=band_adapter_coarse_depth,
                activation_checkpointing=band_adapter_activation_checkpointing,
                dropout=band_adapter_dropout,
            )
        )
        self.output = nn.Conv2d(width, 1, kernel_size=1)
        # V3 transfer supplies a useful coarse wavefield.  Start the new residual
        # branch close enough to identity that its random features cannot erase it,
        # while retaining non-zero gradients through every upstream V4 block.
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-6)
        nn.init.zeros_(self.output.bias)
        self.correction_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.high_frequency_head = (
            None
            if not bool(high_frequency_residual)
            else _HighFrequencyResidualHead(
                width, hidden=int(high_frequency_hidden), depth=int(high_frequency_depth)
            )
        )

    @torch.no_grad()
    def activate_residual_correction(
        self,
        *,
        scale: float = 1.0,
        output_std: float = 1.0e-3,
    ) -> dict[str, float]:
        """Reset a collapsed residual head and keep its gate away from zero."""

        value = float(scale)
        deviation = float(output_std)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("residual correction scale must be positive and finite")
        if not math.isfinite(deviation) or deviation <= 0.0:
            raise ValueError("residual output initialization must be positive and finite")
        nn.init.normal_(self.output.weight, mean=0.0, std=deviation)
        nn.init.zeros_(self.output.bias)
        self.correction_scale.fill_(value)
        self.correction_scale.requires_grad_(False)
        return {
            "scale": value,
            "output_std": float(self.output.weight.float().std()),
        }

    def forward(
        self,
        velocity_mps: torch.Tensor,
        medium: MediumEncoding,
        source: SourceEncoding,
        source_parameters: torch.Tensor,
        source_features: torch.Tensor,
        record_to_medium: torch.Tensor,
        time_s: torch.Tensor,
        coarse: torch.Tensor,
        travel: RayTravelTime,
        saved_time_indices: torch.Tensor,
    ) -> torch.Tensor:
        corrected, _ = self.forward_with_routing(
            velocity_mps,
            medium,
            source,
            source_parameters,
            source_features,
            record_to_medium,
            time_s,
            coarse,
            travel,
            saved_time_indices,
        )
        return corrected

    def forward_with_anchor_increment_and_routing(
        self,
        velocity_mps: torch.Tensor,
        medium: MediumEncoding,
        source: SourceEncoding,
        source_parameters: torch.Tensor,
        source_features: torch.Tensor,
        record_to_medium: torch.Tensor,
        time_s: torch.Tensor,
        coarse: torch.Tensor,
        travel: RayTravelTime,
        saved_time_indices: torch.Tensor,
        *,
        route_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mapping = torch.as_tensor(record_to_medium, dtype=torch.long, device=coarse.device)
        if coarse.ndim != 4 or time_s.shape != coarse.shape[:2]:
            raise ValueError("V4 coarse frames must be [record,time,z,x]")
        records, count, height, width = coarse.shape
        if (
            source_parameters.shape != (records, 5)
            or source_features.shape != (records, 5)
            or mapping.shape != (records,)
        ):
            raise ValueError("V4 source or medium mapping does not match dense frames")
        if saved_time_indices.shape != (records, count):
            raise ValueError("saved time indices must match dense requested times")
        full_size = medium.pyramid[0].shape[-2:]
        if full_size != (height, width):
            raise ValueError("V4 dense grid must match encoded medium resolution")

        levels = [medium.pyramid[0][mapping]]
        levels.extend(
            F.interpolate(level[mapping], size=full_size, mode="bilinear", align_corners=True)
            for level in medium.pyramid[1:]
        )
        spatial = (
            self.multiscale_fuse(torch.cat(levels, dim=1))
            + self.source_projection(source.hidden)[:, :, None, None]
            + self.map_projection(source.map_field)
        )
        relative_time = time_s - source_parameters[:, None, 3]
        frequency = source_parameters[:, None, 2].expand_as(time_s)
        global_phase = 2.0 * math.pi * frequency * relative_time
        time_features = torch.stack(
            (
                time_s / self.domain_t_s,
                relative_time / self.domain_t_s,
                frequency / 50.0,
                torch.sin(global_phase),
                torch.cos(global_phase),
            ),
            dim=-1,
        )
        scale, bias = self.time_mlp(time_features).chunk(2, dim=-1)
        conditioned = (
            spatial[:, None] * (1.0 + scale[:, :, :, None, None])
            + bias[:, :, :, None, None]
            + self.saved_time_embedding(saved_time_indices)[:, :, :, None, None]
        )
        flat = conditioned.reshape(records * count, -1, height, width)
        flat = flat + self.coarse_lift(coarse.reshape(records * count, 1, height, width))
        if self.phase_projection is not None:
            phase = dense_propagation_features(
                travel,
                time_s,
                source_parameters,
                height=height,
                width=width,
                domain_t_s=self.domain_t_s,
                domain_diagonal_m=self.domain_diagonal_m,
                gabor_scales_s=self.gabor_scales_s,
            )
            flat = flat + self.phase_projection(phase.reshape(records * count, 12, height, width))
        shared = self.stack(flat)
        correction = self.output(shared.float()).reshape(records, count, height, width)
        high_frequency = (
            0.0
            if self.high_frequency_head is None
            else self.high_frequency_head(shared.float()).reshape(records, count, height, width)
        )
        temporal = (
            0.0
            if self.temporal_basis is None
            else self.temporal_basis(
                spatial,
                time_features,
                spatial_stack=self.stack,
            )
        )
        if self.family_experts is None:
            expert_correction: torch.Tensor | float = 0.0
            router_logits = shared.new_empty((0, 3))
            router_probabilities = shared.new_empty((0, 3))
        else:
            expert_output = self.family_experts(
                velocity_mps,
                medium.pyramid[0],
                shared.float(),
                time_features,
                mapping,
                route_override=route_override,
            )
            expert_correction = expert_output.correction
            router_logits = expert_output.router_logits
            router_probabilities = expert_output.effective_router_probabilities
        anchor = (
            coarse
            + self.correction_scale * correction
            + high_frequency
            + temporal
            + expert_correction
        )
        raw_increment = (
            anchor.new_zeros(anchor.shape)
            if self.band_limited_adapter is None
            else self.band_limited_adapter(
                shared.float(),
                time_features,
                mapping,
                router_probabilities,
                velocity_mps=velocity_mps,
                source_features=source_features,
            )
        )
        return anchor, raw_increment, router_logits

    def forward_with_routing(
        self,
        velocity_mps: torch.Tensor,
        medium: MediumEncoding,
        source: SourceEncoding,
        source_parameters: torch.Tensor,
        source_features: torch.Tensor,
        record_to_medium: torch.Tensor,
        time_s: torch.Tensor,
        coarse: torch.Tensor,
        travel: RayTravelTime,
        saved_time_indices: torch.Tensor,
        *,
        route_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchor, raw_increment, router_logits = (
            self.forward_with_anchor_increment_and_routing(
                velocity_mps,
                medium,
                source,
                source_parameters,
                source_features,
                record_to_medium,
                time_s,
                coarse,
                travel,
                saved_time_indices,
                route_override=route_override,
            )
        )
        return anchor + raw_increment, router_logits

    def required_gradient_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        phase = () if self.phase_projection is None else tuple(self.phase_projection.parameters())
        groups = {
            "dense_spectral": tuple(self.stack.parameters()),
            "dense_phase": phase,
            "dense_time": tuple(self.time_mlp.parameters())
            + tuple(self.saved_time_embedding.parameters()),
            "dense_local": tuple(self.multiscale_fuse.parameters())
            + tuple(self.source_projection.parameters())
            + tuple(self.map_projection.parameters())
            + tuple(self.coarse_lift.parameters())
            + tuple(self.output.parameters())
            + (self.correction_scale,),
        }
        if self.temporal_basis is not None:
            groups["dense_temporal_basis_gate"] = (self.temporal_basis.gate,)
            groups["dense_temporal_basis_features"] = (
                self.temporal_basis.feature_parameters()
            )
        if self.band_limited_adapter is not None:
            groups["dense_band_limited_adapter"] = tuple(
                self.band_limited_adapter.parameters()
            )
        return groups


__all__ = ["PropagationConditionedDenseDecoder", "QueryInvariantTemporalBasis"]
