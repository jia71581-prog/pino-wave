"""Family-routed wavefield increments with a structural high-band guard."""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .experts import LowRankResidualExpert, _routed_expert_mix
from .spectral import FactorizedComplexResidualStack


def normalized_radial_frequency(
    height: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return the registered normalized radius for a real 2-D FFT grid."""

    rows = int(height)
    columns = int(width)
    if rows <= 0 or columns <= 0:
        raise ValueError("spatial frequency dimensions must be positive")
    z_frequency = torch.fft.fftfreq(rows, device=device).abs()
    x_frequency = torch.fft.rfftfreq(columns, device=device).abs()
    radius = torch.sqrt(
        z_frequency[:, None].square() + x_frequency[None, :].square()
    )
    return radius / radius.max().clamp_min(torch.finfo(radius.dtype).eps)


def registered_high_band_mask(
    height: int,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Select the high band used by the saved-time validation metric."""

    return normalized_radial_frequency(height, width, device) > (2.0 / 3.0)


def project_low_mid_increment(value: torch.Tensor) -> torch.Tensor:
    """Remove every registered high-band coefficient from an output increment."""

    increment = torch.as_tensor(value)
    if increment.ndim != 4:
        raise ValueError("band-limited increment must be [record,time,z,x]")
    height, width = increment.shape[-2:]
    spectrum = torch.fft.rfft2(increment.float(), norm="ortho")
    keep = ~registered_high_band_mask(height, width, spectrum.device)
    spectrum = spectrum * keep
    retained = keep.sum(dim=0).clamp_min(1).to(dtype=spectrum.real.dtype)
    vertical_sum = spectrum.sum(dim=-2, keepdim=True)
    spectrum = torch.where(
        keep[None, None, :, :],
        spectrum - vertical_sum / retained[None, None, None, :],
        torch.zeros_like(spectrum),
    )
    projected = torch.fft.irfft2(
        spectrum,
        s=(height, width),
        norm="ortho",
    )
    projected[..., 0, :] = 0.0
    return projected


class MultiscaleSpectralResidualExpert(nn.Module):
    """U-FNO-like exact-time residual expert with a zero-output head."""

    def __init__(
        self,
        *,
        width: int,
        latent_width: int,
        spectral_rank: int,
        modes: int,
        full_depth: int,
        coarse_depth: int,
        activation_checkpointing: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dimensions = (
            int(width),
            int(latent_width),
            int(spectral_rank),
            int(modes),
            int(full_depth),
            int(coarse_depth),
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("multiscale band adapter capacities must be positive")
        if int(modes) > 101:
            raise ValueError("multiscale band adapter modes cannot exceed 101")
        if not isinstance(activation_checkpointing, bool):
            raise ValueError("band adapter activation checkpointing must be boolean")
        dropout_probability = float(dropout)
        if not math.isfinite(dropout_probability) or not 0.0 <= dropout_probability < 1.0:
            raise ValueError("band adapter dropout must be in [0, 1)")
        self.latent_width = int(latent_width)
        self.input_projection = nn.Conv2d(int(width), self.latent_width, 1)
        self.full_stack = FactorizedComplexResidualStack(
            self.latent_width,
            int(spectral_rank),
            int(modes),
            int(full_depth),
            activation_checkpointing=activation_checkpointing,
        )
        self.coarse_stack = FactorizedComplexResidualStack(
            self.latent_width,
            int(spectral_rank),
            int(modes),
            int(coarse_depth),
            activation_checkpointing=activation_checkpointing,
        )
        self.fuse = nn.Conv2d(2 * self.latent_width, self.latent_width, 1)
        self.time = nn.Sequential(
            nn.Linear(5, self.latent_width),
            nn.GELU(),
            nn.Linear(self.latent_width, 2 * self.latent_width),
        )
        self.dropout = nn.Dropout2d(p=dropout_probability)
        self.output = nn.Conv2d(self.latent_width, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self, shared: torch.Tensor, time_features: torch.Tensor
    ) -> torch.Tensor:
        records, times = time_features.shape[:2]
        lifted = self.input_projection(shared)
        full = self.full_stack(lifted)
        coarse = F.avg_pool2d(
            lifted,
            kernel_size=2,
            stride=2,
            ceil_mode=True,
        )
        coarse = self.coarse_stack(coarse)
        coarse = F.interpolate(
            coarse,
            size=full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.fuse(torch.cat((full, coarse), dim=1))
        scale, shift = self.time(time_features).reshape(
            records * times,
            2 * self.latent_width,
            1,
            1,
        ).chunk(2, dim=1)
        hidden = self.dropout(F.gelu(hidden * (1.0 + scale) + shift))
        return self.output(hidden).reshape(
            records,
            times,
            *shared.shape[-2:],
        )


class DynamicMultiscaleSpectralResidualExpert(MultiscaleSpectralResidualExpert):
    """Multiscale propagator modulated by one encoded medium and each source."""

    def forward(
        self,
        shared: torch.Tensor,
        time_features: torch.Tensor,
        medium_features: torch.Tensor,
        source_modulation: torch.Tensor,
    ) -> torch.Tensor:
        records, times = time_features.shape[:2]
        expected_medium = (records, self.latent_width, *shared.shape[-2:])
        if medium_features.shape != expected_medium:
            raise ValueError("dynamic adapter medium features do not match records")
        if source_modulation.shape != (records, 2 * self.latent_width):
            raise ValueError("dynamic adapter source modulation does not match records")
        lifted = self.input_projection(shared)
        medium = medium_features[:, None].expand(
            -1, times, -1, -1, -1
        ).reshape(records * times, self.latent_width, *shared.shape[-2:])
        lifted = lifted + medium
        full = self.full_stack(lifted)
        coarse = F.avg_pool2d(lifted, kernel_size=2, stride=2, ceil_mode=True)
        coarse = self.coarse_stack(coarse)
        coarse = F.interpolate(
            coarse,
            size=full.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.fuse(torch.cat((full, coarse), dim=1))
        modulation = self.time(time_features) + source_modulation[:, None, :]
        scale, shift = modulation.reshape(
            records * times,
            2 * self.latent_width,
            1,
            1,
        ).chunk(2, dim=1)
        hidden = self.dropout(F.gelu(hidden * (1.0 + scale) + shift))
        return self.output(hidden).reshape(
            records,
            times,
            *shared.shape[-2:],
        )


class BandLimitedFamilyAdapter(nn.Module):
    """Apply zero-output residual heads, optionally using an existing family route."""

    def __init__(
        self,
        *,
        width: int,
        rank: int,
        architecture: str = "low_rank",
        spectral_rank: int = 32,
        modes: int = 32,
        full_depth: int = 4,
        coarse_depth: int = 2,
        activation_checkpointing: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(width) <= 0 or int(rank) <= 0:
            raise ValueError("band adapter width and rank must be positive")
        architecture_name = str(architecture)
        if architecture_name not in {
            "low_rank",
            "multiscale_spectral",
            "dynamic_multiscale_spectral",
            "shared_dynamic_multiscale_spectral",
        }:
            raise ValueError("band adapter architecture is unsupported")
        self.architecture = architecture_name
        self.medium_encoder: nn.Module | None = None
        self.source_encoder: nn.Module | None = None
        dropout_probability = float(dropout)
        if not math.isfinite(dropout_probability) or not 0.0 <= dropout_probability < 1.0:
            raise ValueError("band adapter dropout must be in [0, 1)")
        if self.architecture == "low_rank":
            if dropout_probability != 0.0:
                raise ValueError("band adapter dropout requires a multiscale architecture")
            factory = lambda: LowRankResidualExpert(
                width=int(width),
                rank=int(rank),
            )
        elif self.architecture == "multiscale_spectral":
            factory = lambda: MultiscaleSpectralResidualExpert(
                width=int(width),
                latent_width=int(rank),
                spectral_rank=int(spectral_rank),
                modes=int(modes),
                full_depth=int(full_depth),
                coarse_depth=int(coarse_depth),
                activation_checkpointing=activation_checkpointing,
                dropout=dropout_probability,
            )
        else:
            latent_width = int(rank)
            self.medium_encoder = nn.Sequential(
                nn.Conv2d(2, latent_width, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(
                    latent_width,
                    latent_width,
                    kernel_size=3,
                    padding=1,
                    groups=latent_width,
                ),
            )
            self.source_encoder = nn.Sequential(
                nn.Linear(5, latent_width),
                nn.GELU(),
                nn.Linear(latent_width, 2 * latent_width),
            )
            factory = lambda: DynamicMultiscaleSpectralResidualExpert(
                width=int(width),
                latent_width=latent_width,
                spectral_rank=int(spectral_rank),
                modes=int(modes),
                full_depth=int(full_depth),
                coarse_depth=int(coarse_depth),
                activation_checkpointing=activation_checkpointing,
                dropout=dropout_probability,
            )
        expert_count = (
            1 if self.architecture == "shared_dynamic_multiscale_spectral" else 3
        )
        self.experts = nn.ModuleList(factory() for _ in range(expert_count))

    def forward(
        self,
        shared: torch.Tensor,
        time_features: torch.Tensor,
        record_to_medium: torch.Tensor,
        router_probabilities: torch.Tensor,
        *,
        velocity_mps: torch.Tensor | None = None,
        source_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(shared)
        times = torch.as_tensor(time_features)
        if times.ndim != 3 or times.shape[-1] != 5:
            raise ValueError("band adapter time features must have shape [record,time,5]")
        records, time_count = times.shape[:2]
        if values.ndim != 4 or values.shape[0] != records * time_count:
            raise ValueError("band adapter shared features do not match record/time axes")

        mapping = torch.as_tensor(
            record_to_medium,
            dtype=torch.long,
            device=values.device,
        )
        if mapping.shape != (records,) or mapping.numel() == 0 or int(mapping.min()) < 0:
            raise ValueError("band adapter record_to_medium contains invalid indices")

        route_free = self.architecture == "shared_dynamic_multiscale_spectral"
        probabilities = torch.as_tensor(
            router_probabilities,
            dtype=torch.float32,
            device=values.device,
        )
        if route_free:
            if probabilities.numel() != 0:
                raise ValueError("shared dynamic band adapter does not accept a family route")
        elif probabilities.ndim != 2 or probabilities.shape[1] != len(self.experts):
            raise ValueError("band adapter router probabilities must have shape [medium,3]")
        if not route_free and (
            probabilities.numel() == 0
            or not bool(torch.isfinite(probabilities).all())
            or bool((probabilities < 0.0).any())
        ):
            raise ValueError("band adapter router probabilities must be finite and nonnegative")
        if not route_free and not bool(
            torch.isclose(
                probabilities.sum(dim=-1),
                torch.ones(probabilities.shape[0], device=values.device),
                rtol=0.0,
                atol=1.0e-4,
            ).all()
        ):
            raise ValueError("band adapter router probabilities must sum to one")

        if not route_free and int(mapping.max()) >= probabilities.shape[0]:
            raise ValueError("band adapter record_to_medium contains invalid indices")

        if self.architecture in {
            "dynamic_multiscale_spectral",
            "shared_dynamic_multiscale_spectral",
        }:
            if velocity_mps is None or source_features is None:
                raise ValueError(
                    "dynamic band adapter requires velocity_mps and source_features"
                )
            velocity = torch.as_tensor(
                velocity_mps, dtype=torch.float32, device=values.device
            )
            sources = torch.as_tensor(
                source_features, dtype=torch.float32, device=values.device
            )
            if (
                velocity.ndim != 4
                or velocity.shape[1] != 1
                or velocity.shape[-2:] != values.shape[-2:]
                or int(mapping.max()) >= velocity.shape[0]
            ):
                raise ValueError("dynamic adapter velocity_mps has invalid shape")
            if sources.shape != (records, 5):
                raise ValueError("dynamic adapter source_features has invalid shape")
            centered = velocity - velocity.mean(dim=(-2, -1), keepdim=True)
            physical_velocity = torch.cat(
                (velocity / 5000.0, centered / 1000.0), dim=1
            )
            medium_by_medium = self.medium_encoder(physical_velocity)
            medium_by_record = medium_by_medium[mapping]
            source_modulation = self.source_encoder(sources)
            if route_free:
                return self.experts[0](
                    values,
                    times,
                    medium_by_record,
                    source_modulation,
                )
            routed = probabilities[mapping].to(dtype=values.dtype)
            return self._dynamic_routed_mix(
                values,
                times,
                routed,
                medium_by_record,
                source_modulation,
            )
        routed = probabilities[mapping].to(dtype=values.dtype)
        return _routed_expert_mix(
            self.experts,
            values,
            times,
            routed,
        )

    def _dynamic_routed_mix(
        self,
        shared: torch.Tensor,
        time_features: torch.Tensor,
        probabilities: torch.Tensor,
        medium_features: torch.Tensor,
        source_modulation: torch.Tensor,
    ) -> torch.Tensor:
        records, times = time_features.shape[:2]
        exact_one_hot = bool(
            torch.logical_or(probabilities == 0.0, probabilities == 1.0).all()
        )
        if not exact_one_hot:
            corrections = torch.stack(
                [
                    expert(
                        shared,
                        time_features,
                        medium_features,
                        source_modulation,
                    )
                    for expert in self.experts
                ],
                dim=2,
            )
            return (
                corrections * probabilities[:, None, :, None, None]
            ).sum(dim=2)

        routes = probabilities.argmax(dim=-1)
        shared_by_record = shared.reshape(records, times, *shared.shape[1:])
        mixed = shared.new_zeros(
            (records, times, shared.shape[-2], shared.shape[-1])
        )
        for expert_index, expert in enumerate(self.experts):
            selected = torch.nonzero(
                routes == expert_index, as_tuple=False
            ).flatten()
            if selected.numel() == 0:
                continue
            selected_shared = shared_by_record.index_select(0, selected).reshape(
                selected.numel() * times, *shared.shape[1:]
            )
            correction = expert(
                selected_shared,
                time_features.index_select(0, selected),
                medium_features.index_select(0, selected),
                source_modulation.index_select(0, selected),
            )
            mixed = mixed.index_copy(0, selected, correction)
        return mixed


@torch.no_grad()
def activate_band_adapter_output(
    adapter: BandLimitedFamilyAdapter,
    *,
    std: float,
    seed: int,
) -> dict[str, object]:
    """Wake dynamic features with a small deterministic nonzero readout."""

    deviation = float(std)
    if not math.isfinite(deviation) or deviation <= 0.0:
        raise ValueError("band adapter output initialization must be positive and finite")
    if adapter.architecture not in {
        "dynamic_multiscale_spectral",
        "shared_dynamic_multiscale_spectral",
    }:
        raise ValueError("band adapter output initialization requires dynamic architecture")
    generator = torch.Generator(device=adapter.experts[0].output.weight.device)
    generator.manual_seed(int(seed))
    weights: list[torch.Tensor] = []
    for expert in adapter.experts:
        expert.output.weight.normal_(mean=0.0, std=deviation, generator=generator)
        expert.output.bias.zero_()
        weights.append(expert.output.weight.detach().float().flatten())
    combined = torch.cat(weights)
    return {
        "architecture": adapter.architecture,
        "requested_output_std": deviation,
        "output_std": float(combined.std()),
        "seed": int(seed),
    }


def build_band_adapter_adamw(
    model: nn.Module,
    *,
    feature_lr: float,
    output_lr: float,
    weight_decay: float,
    implementation: str = "single_tensor",
) -> torch.optim.AdamW:
    """Use a small zero-head step without starving spectral feature learning."""

    from .full_support import adamw_backend_options

    rates = {
        "adapter_feature": float(feature_lr),
        "adapter_output": float(output_lr),
    }
    if any(
        not math.isfinite(value) or value <= 0.0 for value in rates.values()
    ):
        raise ValueError("band adapter learning rates must be positive and finite")
    decay_value = float(weight_decay)
    if not math.isfinite(decay_value) or decay_value < 0.0:
        raise ValueError("band adapter weight decay must be finite and nonnegative")
    path = getattr(getattr(model, "dense_decoder", None), "band_limited_adapter", None)
    if path is None:
        raise ValueError("band adapter optimizer requires an enabled adapter")
    prefix = "dense_decoder.band_limited_adapter."
    grouped: dict[str, dict[str, list[nn.Parameter]]] = {
        name: {"decay": [], "no_decay": []}
        for name in ("adapter_feature", "adapter_output")
    }
    selected: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not name.startswith(prefix):
            continue
        role = "adapter_output" if ".output." in name else "adapter_feature"
        decay_class = (
            "no_decay"
            if parameter.ndim <= 1 or name.endswith("bias")
            else "decay"
        )
        grouped[role][decay_class].append(parameter)
        selected.append(parameter)
    expected_ids = {id(parameter) for parameter in path.parameters()}
    selected_ids = [id(parameter) for parameter in selected]
    if len(selected_ids) != len(set(selected_ids)) or set(selected_ids) != expected_ids:
        raise ValueError("band adapter optimizer parameter partition is incomplete")
    if any(not any(parts.values()) for parts in grouped.values()):
        raise ValueError("band adapter optimizer roles must both be nonempty")
    parameter_groups: list[dict[str, object]] = []
    for role in ("adapter_feature", "adapter_output"):
        for decay_class in ("decay", "no_decay"):
            parameters = grouped[role][decay_class]
            if not parameters:
                continue
            parameter_groups.append(
                {
                    "params": parameters,
                    "lr": rates[role],
                    "initial_lr": rates[role],
                    "group_name": f"{role}_{decay_class}",
                    "weight_decay": (
                        decay_value if decay_class == "decay" else 0.0
                    ),
                }
            )
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=0.0,
        **adamw_backend_options(implementation),
    )


def registered_band_adapter_learning_rates(
    config: Mapping[str, object],
) -> tuple[float, float] | None:
    """Resolve an all-or-nothing feature/output learning-rate registration."""

    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ValueError("optimizer must be a mapping")
    feature_key = "band_adapter_feature_learning_rate"
    output_key = "band_adapter_output_learning_rate"
    feature_present = feature_key in optimizer
    output_present = output_key in optimizer
    if not feature_present and not output_present:
        return None
    if feature_present != output_present:
        raise ValueError("band adapter learning rates must be registered together")
    if config.get("band_limited_adapter") is None:
        raise ValueError("band adapter learning rates require an enabled adapter")
    feature_lr = float(optimizer[feature_key])
    output_lr = float(optimizer[output_key])
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (feature_lr, output_lr)
    ):
        raise ValueError("band adapter learning rates must be positive and finite")
    return feature_lr, output_lr


def configure_band_adapter_stage(model: nn.Module, *, epoch: int):
    """Freeze the high-band anchor and train only the registered adapter."""

    from .full_support import TrainableStage

    current = int(epoch)
    if current <= 0:
        raise ValueError("band adapter stage epoch must be positive")
    path = getattr(getattr(model, "dense_decoder", None), "band_limited_adapter", None)
    if path is None:
        raise ValueError("band adapter stage requires an enabled adapter")
    prefix = "dense_decoder.band_limited_adapter."
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        active = name.startswith(prefix)
        parameter.requires_grad_(active)
        if active:
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    if trainable <= 0:
        raise ValueError("band adapter stage selected no trainable parameters")
    return TrainableStage(
        epoch=current,
        trainable_prefixes=(prefix.removesuffix("."),),
        trainable_parameters=int(trainable),
        frozen_parameters=int(frozen),
    )


__all__ = [
    "BandLimitedFamilyAdapter",
    "DynamicMultiscaleSpectralResidualExpert",
    "MultiscaleSpectralResidualExpert",
    "activate_band_adapter_output",
    "build_band_adapter_adamw",
    "configure_band_adapter_stage",
    "normalized_radial_frequency",
    "project_low_mid_increment",
    "registered_band_adapter_learning_rates",
    "registered_high_band_mask",
]
