"""Small zero-initialized onset-conditioned residual adapters."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .chonknoris import ReducedCholeskyPredictor


ADAPTER_SCHEMA_VERSION = 2


def normalize_source_parameters(source: torch.Tensor) -> torch.Tensor:
    """Normalize one or more physical sources without batch-dependent statistics."""

    value = torch.as_tensor(source).float()
    if value.ndim != 2 or value.shape[1] != 5:
        raise ValueError("source must be [record,5]")
    if not torch.isfinite(value).all():
        raise ValueError("source parameters must be finite")
    scales = torch.tensor(
        (2000.0, 2000.0, 50.0, 1.2, 1.0),
        dtype=value.dtype,
        device=value.device,
    )
    return value / scales


def normalize_velocity_contrast(
    velocity: torch.Tensor,
    *,
    minimum_dynamic_range_mps: float = 1.0e-3,
    validate_finite: bool = True,
) -> torch.Tensor:
    """Normalize heterogeneous contrast while mapping constant media to zero.

    A spatially constant float32 field can acquire a device-dependent nonzero
    mean-subtraction residue while its computed standard deviation is zero.
    Dividing that residue by an epsilon created extreme conditioner inputs on
    some uniform media.  The physical max-min range is exact for a constant
    tensor and is therefore used to select the zero-contrast branch.
    """

    value = torch.as_tensor(velocity).float()
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("velocity must be [record,1,z,x]")
    if bool(validate_finite) and not bool(torch.isfinite(value).all()):
        raise ValueError("velocity must be finite")
    threshold = float(minimum_dynamic_range_mps)
    if threshold <= 0.0:
        raise ValueError("minimum velocity dynamic range must be positive")
    minimum = value.amin(dim=(-2, -1), keepdim=True)
    maximum = value.amax(dim=(-2, -1), keepdim=True)
    heterogeneous = (maximum - minimum) > threshold
    centered = value - value.mean(dim=(-2, -1), keepdim=True)
    standard_deviation = value.std(dim=(-2, -1), keepdim=True)
    safe_scale = torch.where(
        heterogeneous,
        standard_deviation.clamp_min(threshold),
        torch.ones_like(standard_deviation),
    )
    normalized = centered / safe_scale
    return torch.where(heterogeneous, normalized, torch.zeros_like(normalized))


def normalize_observed_snapshots(
    observed_wavefield: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize tiny physical snapshots by per-record RMS without a fixed floor."""

    observed = torch.as_tensor(observed_wavefield).float()
    if observed.ndim != 4 or observed.shape[1] != 2:
        raise ValueError("observed wavefield must be [record,2,z,x]")
    if not bool(torch.isfinite(observed).all()):
        raise ValueError("observed wavefield must be finite")
    scale = observed.square().flatten(1).mean(dim=1, keepdim=True).sqrt()
    active = scale > torch.finfo(observed.dtype).tiny
    safe_scale = torch.where(active, scale, torch.ones_like(scale))
    normalized = observed / safe_scale.view(-1, 1, 1, 1)
    normalized = torch.where(
        active.view(-1, 1, 1, 1), normalized, torch.zeros_like(normalized)
    )
    return normalized, scale


def apply_low_rank_time_film(
    spatial: torch.Tensor,
    temporal: torch.Tensor,
) -> torch.Tensor:
    """Form time-varying spatial bases from per-time scale and bias coefficients."""

    base = torch.as_tensor(spatial)
    coefficients = torch.as_tensor(
        temporal, dtype=base.dtype, device=base.device
    )
    if base.ndim != 4:
        raise ValueError("spatial features must be [record,channel,z,x]")
    records, channels = base.shape[:2]
    if coefficients.ndim != 3 or coefficients.shape[0] != records or coefficients.shape[2] != 2 * channels:
        raise ValueError("temporal coefficients must be [record,time,2*channel]")
    scale, bias = coefficients.chunk(2, dim=-1)
    return (
        base[:, None]
        * (1.0 + torch.tanh(scale)[:, :, :, None, None])
        + bias[:, :, :, None, None]
    )


class ZeroInitLoRA(nn.Module):
    """A low-rank linear residual whose initial contribution is exactly zero."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        if min(int(in_features), int(out_features), int(rank)) <= 0:
            raise ValueError("LoRA dimensions must be positive")
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value))


class OnsetSnapshotConditioner(nn.Module):
    """Encode velocity, two early snapshots, and physical source parameters."""

    def __init__(self, latent_dim: int = 32, width: int = 32, lora_rank: int = 4):
        super().__init__()
        if min(int(latent_dim), int(width), int(lora_rank)) <= 0:
            raise ValueError("conditioner dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.spatial = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.source = nn.Sequential(
            nn.Linear(5, width), nn.GELU(), ZeroInitLoRA(width, width, rank=lora_rank)
        )
        self.output = nn.Sequential(
            nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, latent_dim)
        )

    def forward(
        self,
        velocity: torch.Tensor,
        source: torch.Tensor,
        observed_wavefield: torch.Tensor,
    ) -> torch.Tensor:
        velocity = torch.as_tensor(velocity).float()
        source = torch.as_tensor(source, device=velocity.device).float()
        observed = torch.as_tensor(observed_wavefield, device=velocity.device).float()
        if velocity.ndim != 4 or velocity.shape[1] != 1:
            raise ValueError("velocity must be [record,1,z,x]")
        if observed.ndim != 4 or observed.shape[:2] != (velocity.shape[0], 2):
            raise ValueError("observed wavefield must be [record,2,z,x]")
        if source.shape != (velocity.shape[0], 5):
            raise ValueError("source must be [record,5]")
        observed, _ = normalize_observed_snapshots(observed)
        velocity = normalize_velocity_contrast(velocity)
        spatial = self.spatial(torch.cat((velocity, observed), dim=1)).flatten(1)
        source = normalize_source_parameters(source)
        return self.output(torch.cat((spatial, self.source(source)), dim=-1))


class LowRankFieldResidual(nn.Module):
    """Generate a separable low-rank space-time correction for a full field."""

    def __init__(self, latent_dim: int = 32, width: int = 32):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1), nn.GELU()
        )
        self.latent = nn.Linear(latent_dim, width)
        self.time = nn.Sequential(
            nn.Linear(1, width), nn.GELU(), nn.Linear(width, 2 * width)
        )
        self.output = nn.Conv2d(width, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def sampled_linear_features(
        self,
        velocity: torch.Tensor,
        source: torch.Tensor,
        observed_wavefield: torch.Tensor,
        latent: torch.Tensor,
        time_s: torch.Tensor,
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return pre-output features only at audited per-frame spatial points."""

        del source
        records, _, height, width = observed_wavefield.shape
        times = torch.as_tensor(time_s, dtype=velocity.dtype, device=velocity.device)
        if times.ndim == 1:
            times = times[None].expand(records, -1)
        indices = torch.as_tensor(
            spatial_indices, dtype=torch.long, device=velocity.device
        )
        if times.ndim != 2 or times.shape[0] != records:
            raise ValueError("time_s must be [time] or [record,time]")
        if indices.ndim != 3 or indices.shape[:2] != times.shape:
            raise ValueError("spatial indices must be [record,time,point]")
        if bool(torch.any(indices < 0)) or bool(torch.any(indices >= height * width)):
            raise ValueError("sampled residual index is outside the spatial grid")
        observed_scale = (
            observed_wavefield.detach()
            .flatten(1)
            .std(dim=1, keepdim=True)
            .clamp_min(1.0e-8)
        )
        obs = observed_wavefield / observed_scale.view(records, 1, 1, 1)
        vel = normalize_velocity_contrast(velocity)
        features = self.spatial(torch.cat((vel, obs), dim=1))
        base = F.gelu(features + self.latent(latent)[:, :, None, None])
        base = base.permute(0, 2, 3, 1).reshape(records, height * width, -1)
        batch = torch.arange(records, device=velocity.device)[:, None, None]
        sampled = base[batch, indices]
        temporal = self.time(times.unsqueeze(-1))
        scale, bias = temporal.chunk(2, dim=-1)
        return sampled * (1.0 + torch.tanh(scale)[:, :, None]) + bias[:, :, None]

    def forward(
        self,
        velocity: torch.Tensor,
        source: torch.Tensor,
        observed_wavefield: torch.Tensor,
        latent: torch.Tensor,
        time_s: torch.Tensor,
        time_block: int = 4,
        output_scale: torch.Tensor | None = None,
        validate_output_scale: bool = True,
    ) -> torch.Tensor:
        del source
        records, _, height, width = observed_wavefield.shape
        times = torch.as_tensor(time_s, dtype=velocity.dtype, device=velocity.device)
        if times.ndim == 1:
            times = times[None].expand(records, -1)
        if times.ndim != 2 or times.shape[0] != records:
            raise ValueError("time_s must be [time] or [record,time]")
        observed_scale = (
            observed_wavefield.detach()
            .flatten(1)
            .std(dim=1, keepdim=True)
            .clamp_min(1.0e-8)
        )
        obs = observed_wavefield / observed_scale.view(records, 1, 1, 1)
        vel = normalize_velocity_contrast(
            velocity, validate_finite=bool(validate_output_scale)
        )
        features = self.spatial(torch.cat((vel, obs), dim=1))
        features = features + self.latent(latent)[:, :, None, None]
        base = F.gelu(features)
        block = max(1, int(time_block))
        outputs: list[torch.Tensor] = []
        for start in range(0, times.shape[1], block):
            current = times[:, start : start + block]
            temporal = self.time(current.unsqueeze(-1))
            combined = apply_low_rank_time_film(base, temporal)
            correction = self.output(combined.reshape(records * current.shape[1], -1, height, width))
            outputs.append(correction.reshape(records, current.shape[1], height, width))
        if output_scale is None:
            correction_scale = observed_scale[:, 0]
        else:
            correction_scale = torch.as_tensor(
                output_scale, dtype=velocity.dtype, device=velocity.device
            ).flatten()
            if correction_scale.shape != (records,):
                raise ValueError("adapter output scale must contain one value per record")
            if bool(validate_output_scale) and (
                not torch.isfinite(correction_scale).all()
                or torch.any(correction_scale <= 0.0)
            ):
                raise ValueError("adapter output scale must be finite and positive")
        return torch.tanh(torch.cat(outputs, dim=1)) * correction_scale.view(
            records, 1, 1, 1
        )


class OnsetAdaptedV5(nn.Module):
    """Frozen V5 parent plus a small onset-conditioned full-field residual.

    Two adaptation regimes share one wrapper:

    * ``adapter_parameters`` trains the whole conditioner-plus-residual meta
      network, used for the offline meta-training stage.
    * ``deployment_parameters`` freezes that meta network and trains only a
      per-instance zero-initialized ``latent_delta`` (the deployment LoRA),
      giving a second-scale, near-parameter-free fine-tune at inference time.
    """

    def __init__(self, parent: nn.Module, latent_dim: int = 32, lora_rank: int = 4):
        super().__init__()
        self.parent = parent
        for parameter in self.parent.parameters():
            parameter.requires_grad_(False)
        self.conditioner = OnsetSnapshotConditioner(latent_dim=latent_dim, lora_rank=lora_rank)
        self.residual = LowRankFieldResidual(latent_dim=latent_dim)
        # Deployment LoRA: a zero-initialized shift added to the onset latent so
        # the frozen meta network can be nudged for a single new instance without
        # touching any shared weight.  Starts at zero -> deployment begins on the
        # exact meta-network prediction.
        self.latent_delta = nn.Parameter(torch.zeros(int(latent_dim)))
        # Per-instance gate on the whole meta residual (deployment LoRA).  Starts
        # at 1 so deployment begins on the meta-network prediction; if the shared
        # meta residual is harmful for this instance (e.g. uniform media), the
        # gate can shrink toward 0 and the safe rollback sets it to 0 (== frozen
        # parent), so a bad meta residual can never survive the gates.
        self.residual_gate = nn.Parameter(torch.ones(()))
        # CHONKNORIS is applied only in the small deployment state
        # [latent_delta, residual_gate], never in the full wavefield space.
        self.chonknoris = ReducedCholeskyPredictor(
            context_dim=int(latent_dim),
            state_dim=int(latent_dim) + 1,
        )
        self._chonknoris_pretrained = False
        self._deployment_mode = False

    def adapter_parameters(self) -> tuple[nn.Parameter, ...]:
        """All trainable weights except the per-instance deployment parameters."""
        deployment = {"latent_delta", "residual_gate"}
        return tuple(
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
            and name not in deployment
            and not name.startswith("chonknoris.")
        )

    def chonknoris_parameters(self) -> tuple[nn.Parameter, ...]:
        """Offline-only parameters of the learned reduced Cholesky model."""
        return tuple(self.chonknoris.parameters())

    def deployment_parameters(self) -> tuple[nn.Parameter, ...]:
        """Per-instance parameters trained during deployment (latent + gate)."""
        return (self.latent_delta, self.residual_gate)

    def set_deployment_mode(self, enabled: bool) -> None:
        """Freeze the meta network and train only the deployment parameters."""
        self._deployment_mode = bool(enabled)
        for parameter in self.conditioner.parameters():
            parameter.requires_grad_(not self._deployment_mode)
        for parameter in self.residual.parameters():
            parameter.requires_grad_(not self._deployment_mode)
        self.latent_delta.requires_grad_(True)
        self.residual_gate.requires_grad_(True)

    def active_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters the trainer should optimize given the current mode."""
        return self.deployment_parameters() if self._deployment_mode else self.adapter_parameters()

    def deployment_state(self) -> torch.Tensor:
        """Return ``[latent_delta, residual_gate]`` without detaching it."""
        return torch.cat((self.latent_delta, self.residual_gate.reshape(1)))

    def set_deployment_state_(self, state: torch.Tensor) -> None:
        """Copy one accepted reduced CHONKNORIS state into model parameters."""
        value = torch.as_tensor(
            state, dtype=self.latent_delta.dtype, device=self.latent_delta.device
        )
        if value.shape != (self.latent_delta.numel() + 1,) or not torch.isfinite(value).all():
            raise ValueError("deployment state has an invalid shape or nonfinite value")
        with torch.no_grad():
            self.latent_delta.copy_(value[:-1])
            self.residual_gate.copy_(value[-1])

    def chonknoris_context(
        self,
        velocity: torch.Tensor,
        source: torch.Tensor,
        observed_wavefield: torch.Tensor,
    ) -> torch.Tensor:
        """Problem context used to predict the reduced Cholesky factor."""
        return self.conditioner(velocity, source, observed_wavefield)

    def raw_wavefield_from_state(
        self,
        parent_field: torch.Tensor,
        velocity: torch.Tensor,
        source: torch.Tensor,
        observed_wavefield: torch.Tensor,
        time_s: torch.Tensor,
        deployment_state: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        vmap_compatible: bool = False,
    ) -> torch.Tensor:
        """Evaluate a field at an explicit differentiable reduced state."""
        latent = (
            self.chonknoris_context(velocity, source, observed_wavefield)
            if context is None
            else torch.as_tensor(
                context, dtype=velocity.dtype, device=velocity.device
            )
        )
        if latent.ndim != 2 or latent.shape != (
            velocity.shape[0], self.latent_delta.numel()
        ):
            raise ValueError("cached CHONKNORIS context has an invalid shape")
        state = torch.as_tensor(
            deployment_state, dtype=latent.dtype, device=latent.device
        )
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(latent.shape[0], -1)
        expected = (latent.shape[0], self.latent_delta.numel() + 1)
        if state.shape != expected:
            raise ValueError("deployment state must be [record,latent_dim+1]")
        latent = latent + state[:, :-1]
        gate = state[:, -1]
        parent_scale = (
            parent_field.detach().float().flatten(1).std(dim=1).clamp_min(1.0e-8)
        )
        correction = self.residual(
            velocity,
            source,
            observed_wavefield,
            latent,
            time_s,
            output_scale=parent_scale,
            validate_output_scale=not bool(vmap_compatible),
        )
        if correction.shape != parent_field.shape:
            raise ValueError("parent field and adapter correction shapes do not match")
        return parent_field + gate[:, None, None, None].to(correction.dtype) * correction

    def raw_wavefield(
        self,
        parent_field: torch.Tensor,
        velocity: torch.Tensor,
        source: torch.Tensor,
        observed_wavefield: torch.Tensor,
        time_s: torch.Tensor,
    ) -> torch.Tensor:
        return self.raw_wavefield_from_state(
            parent_field,
            velocity,
            source,
            observed_wavefield,
            time_s,
            self.deployment_state(),
        )

    @staticmethod
    def hard_project(
        prediction: torch.Tensor,
        observed_wavefield: torch.Tensor,
        observed_indices: tuple[int, int],
    ) -> torch.Tensor:
        result = prediction.clone()
        indices = torch.as_tensor(observed_indices, dtype=torch.long, device=result.device)
        result[:, indices] = observed_wavefield.to(result.device)
        return result


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "apply_low_rank_time_film",
    "LowRankFieldResidual",
    "normalize_observed_snapshots",
    "normalize_source_parameters",
    "normalize_velocity_contrast",
    "OnsetAdaptedV5",
    "OnsetSnapshotConditioner",
    "ZeroInitLoRA",
]
