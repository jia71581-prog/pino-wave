"""Parameter-efficient early-feature adaptation for a frozen neural operator."""
from __future__ import annotations

import torch
from torch import nn

from grouped_ufno_mionet_v3.model.medium import MediumEncoding
from grouped_ufno_mionet_v3.model.operator import EncodedMediumState

from .adapters import OnsetSnapshotConditioner


def energy_balanced_relative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    energy_floor_fraction: float = 0.01,
) -> torch.Tensor:
    """Average per-frame relative L2 with a record-relative energy floor."""

    actual = torch.as_tensor(prediction)
    truth = torch.as_tensor(target, dtype=actual.dtype, device=actual.device)
    if actual.shape != truth.shape or actual.ndim != 4:
        raise ValueError("prediction and target must be matching [record,time,z,x]")
    floor_fraction = float(energy_floor_fraction)
    if not 0.0 < floor_fraction <= 1.0:
        raise ValueError("energy floor fraction must lie in (0,1]")
    target_norm = truth.flatten(2).norm(dim=-1)
    peak = target_norm.amax(dim=1, keepdim=True)
    tiny = torch.finfo(actual.dtype).tiny
    denominator = torch.maximum(target_norm, peak * floor_fraction).clamp_min(tiny)
    relative = (actual - truth).flatten(2).norm(dim=-1) / denominator
    return relative.mean()


def metric_aligned_relative_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean per-record relative L2 over the jointly sampled space-time field.

    This is the differentiable training analogue of
    :class:`ExactWavefieldMetricAccumulator`: energy from every sampled time is
    accumulated before taking the norm, so near-zero onset frames cannot
    dominate a record merely because they were sampled.
    """

    actual = torch.as_tensor(prediction)
    truth = torch.as_tensor(target, dtype=actual.dtype, device=actual.device)
    if actual.shape != truth.shape or actual.ndim != 4:
        raise ValueError("prediction and target must be matching [record,time,z,x]")
    error_norm = (actual.float() - truth.float()).flatten(1).norm(dim=-1)
    target_norm = truth.float().flatten(1).norm(dim=-1)
    return (error_norm / target_norm.clamp_min(1.0e-8)).mean()


class EarlyFeatureModulator(nn.Module):
    """Apply bounded per-instance FiLM to frozen medium-encoder features."""

    def __init__(
        self,
        *,
        latent_dim: int,
        pyramid_widths: tuple[int, ...],
        token_width: int,
        rank_width: int,
        max_scale: float = 0.1,
        max_bias: float = 0.1,
    ) -> None:
        super().__init__()
        widths = tuple(int(value) for value in pyramid_widths)
        if (
            int(latent_dim) <= 0
            or not widths
            or min(widths) <= 0
            or int(token_width) <= 0
            or int(rank_width) <= 0
        ):
            raise ValueError("feature modulation dimensions must be positive")
        if float(max_scale) <= 0.0 or float(max_bias) < 0.0:
            raise ValueError("feature modulation bounds are invalid")
        self.pyramid_widths = widths
        self.token_width = int(token_width)
        self.rank_width = int(rank_width)
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.feature_width = sum(widths) + self.token_width + self.rank_width
        self.affine = nn.Linear(int(latent_dim), 2 * self.feature_width)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def _film(
        self,
        value: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        trailing = (1,) * (value.ndim - 2)
        scale = scale.reshape(scale.shape[0], scale.shape[1], *trailing)
        bias = bias.reshape(bias.shape[0], bias.shape[1], *trailing)
        return value * (1.0 + self.max_scale * torch.tanh(scale)) + (
            self.max_bias * torch.tanh(bias)
        )

    def forward(
        self,
        medium: MediumEncoding,
        latent: torch.Tensor,
        record_to_medium: torch.Tensor,
        modulation_gate: torch.Tensor | None = None,
    ) -> MediumEncoding:
        code = torch.as_tensor(latent)
        mapping = torch.as_tensor(
            record_to_medium, dtype=torch.long, device=code.device
        )
        if code.ndim != 2 or mapping.shape != (code.shape[0],):
            raise ValueError("latent and record-to-medium mapping shapes do not match")
        medium_count = medium.pyramid[0].shape[0]
        if bool(torch.any(mapping < 0)) or bool(torch.any(mapping >= medium_count)):
            raise ValueError("record-to-medium mapping is outside medium support")
        if len(medium.pyramid) != len(self.pyramid_widths):
            raise ValueError("medium pyramid depth does not match modulator")
        coefficients = self.affine(code)
        if modulation_gate is not None:
            gate = torch.as_tensor(
                modulation_gate, dtype=code.dtype, device=code.device
            )
            if gate.ndim == 1:
                gate = gate[:, None]
            if gate.shape != (code.shape[0], 1) or not bool(torch.isfinite(gate).all()):
                raise ValueError("modulation gate must be one finite scalar per record")
            if bool(torch.any(gate < 0.0)) or bool(torch.any(gate > 1.0)):
                raise ValueError("modulation gate must lie in [0,1]")
            coefficients = coefficients * gate
        scale, bias = coefficients.chunk(2, dim=-1)
        offset = 0
        pyramid: list[torch.Tensor] = []
        for level, width in zip(medium.pyramid, self.pyramid_widths):
            if level.shape[1] != width:
                raise ValueError("medium pyramid width does not match modulator")
            selected = level[mapping]
            pyramid.append(
                self._film(
                    selected,
                    scale[:, offset : offset + width],
                    bias[:, offset : offset + width],
                )
            )
            offset += width
        if medium.tokens.shape[-1] != self.token_width:
            raise ValueError("medium token width does not match modulator")
        tokens = medium.tokens[mapping].transpose(1, 2)
        tokens = self._film(
            tokens,
            scale[:, offset : offset + self.token_width],
            bias[:, offset : offset + self.token_width],
        ).transpose(1, 2)
        offset += self.token_width
        if medium.rank.shape[-1] != self.rank_width:
            raise ValueError("medium rank width does not match modulator")
        rank = self._film(
            medium.rank[mapping],
            scale[:, offset : offset + self.rank_width],
            bias[:, offset : offset + self.rank_width],
        )
        return MediumEncoding(
            pyramid=tuple(pyramid),
            tokens=tokens,
            token_positions=medium.token_positions[mapping],
            rank=rank,
        )


class EarlyFeatureOnsetAdapter(nn.Module):
    """Freeze a parent operator and condition its earliest medium features."""

    def __init__(
        self,
        parent: nn.Module,
        *,
        latent_dim: int = 16,
        lora_rank: int = 4,
        max_scale: float = 0.1,
        max_bias: float = 0.1,
    ) -> None:
        super().__init__()
        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        encoder = getattr(parent, "medium_encoder", None)
        width = int(getattr(encoder, "width", 0))
        rank_width = int(getattr(encoder, "rank_size", 0))
        depth = len(getattr(encoder, "blocks", ()))
        if min(width, rank_width, depth) <= 0:
            raise ValueError("parent medium encoder does not expose feature dimensions")
        self.conditioner = OnsetSnapshotConditioner(
            latent_dim=int(latent_dim), lora_rank=int(lora_rank)
        )
        self.modulator = EarlyFeatureModulator(
            latent_dim=int(latent_dim),
            pyramid_widths=(width,) * depth,
            token_width=width,
            rank_width=rank_width,
            max_scale=float(max_scale),
            max_bias=float(max_bias),
        )

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )

    @staticmethod
    def _mapping(
        medium_count: int,
        record_count: int,
        record_to_medium: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if record_to_medium is None:
            if medium_count == 1:
                return torch.zeros(record_count, dtype=torch.long, device=device)
            if medium_count == record_count:
                return torch.arange(record_count, dtype=torch.long, device=device)
            raise ValueError("record-to-medium mapping is required for grouped inputs")
        mapping = torch.as_tensor(
            record_to_medium, dtype=torch.long, device=device
        )
        if mapping.shape != (record_count,):
            raise ValueError("record-to-medium mapping shape does not match sources")
        if bool(torch.any(mapping < 0)) or bool(torch.any(mapping >= medium_count)):
            raise ValueError("record-to-medium mapping is outside medium support")
        return mapping

    def prepare_sources(
        self,
        velocity_mps: torch.Tensor,
        source_parameters: torch.Tensor,
        source_map: torch.Tensor,
        observed_wavefield: torch.Tensor,
        normalizer,
        *,
        record_to_medium: torch.Tensor | None = None,
        latent_delta: torch.Tensor | None = None,
        conditioner_wavefield: torch.Tensor | None = None,
        modulation_gate: torch.Tensor | None = None,
    ):
        velocity = torch.as_tensor(velocity_mps).float()
        source = torch.as_tensor(source_parameters, device=velocity.device).float()
        source_map_tensor = torch.as_tensor(source_map, device=velocity.device).float()
        observed = torch.as_tensor(
            observed_wavefield, device=velocity.device
        ).float()
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity must be [medium,1,z,x]")
        if source.ndim != 2 or source.shape[1] != 5:
            raise ValueError("source must be [record,5]")
        mapping = self._mapping(
            velocity.shape[0], source.shape[0], record_to_medium, device=velocity.device
        )
        medium = self.parent.encode_medium(velocity, normalizer)
        velocity_by_record = velocity[mapping]
        conditioner_input = (
            observed
            if conditioner_wavefield is None
            else torch.as_tensor(
                conditioner_wavefield, dtype=observed.dtype, device=observed.device
            )
        )
        if conditioner_input.shape != observed.shape or not bool(
            torch.isfinite(conditioner_input).all()
        ):
            raise ValueError(
                "conditioner wavefield must be finite and match observed snapshots"
            )
        latent = self.conditioner(velocity_by_record, source, conditioner_input)
        if latent_delta is not None:
            delta = torch.as_tensor(
                latent_delta, dtype=latent.dtype, device=latent.device
            )
            if delta.shape != latent.shape or not bool(torch.isfinite(delta).all()):
                raise ValueError("latent delta must be finite and match the onset code")
            latent = latent + delta
        modulated = self.modulator(
            medium.encoding,
            latent,
            mapping,
            modulation_gate=modulation_gate,
        )
        expanded_medium = EncodedMediumState(
            velocity_mps=medium.velocity_mps[mapping], encoding=modulated
        )
        identity_mapping = torch.arange(
            source.shape[0], dtype=torch.long, device=velocity.device
        )
        prepared = self.parent.prepare_sources(
            expanded_medium,
            source,
            source_map_tensor,
            normalizer,
            record_to_medium=identity_mapping,
        )
        return prepared, latent


__all__ = [
    "EarlyFeatureModulator",
    "EarlyFeatureOnsetAdapter",
    "energy_balanced_relative_loss",
    "metric_aligned_relative_loss",
]
