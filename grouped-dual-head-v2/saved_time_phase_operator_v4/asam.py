"""Adaptive sharpness-aware refinement policies with exact restoration."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class ASAMPerturbation:
    parameters: tuple[nn.Parameter, ...]
    originals: tuple[torch.Tensor, ...]
    gradient_norm: float
    perturbation_norm: float
    _restored: bool = False

    @torch.no_grad()
    def restore(self) -> None:
        if self._restored:
            return
        for parameter, original in zip(self.parameters, self.originals, strict=True):
            parameter.copy_(original)
        self._restored = True


def freeze_for_asam(
    model: nn.Module, *, trainable_prefixes: Iterable[str]
) -> tuple[nn.Parameter, ...]:
    """Freeze a model and enable only explicitly registered refinement prefixes."""

    prefixes = tuple(str(value).strip() for value in trainable_prefixes)
    if not prefixes or any(not value for value in prefixes):
        raise ValueError("ASAM trainable prefixes cannot be empty")
    if len(prefixes) != len(set(prefixes)):
        raise ValueError("ASAM trainable prefixes must be unique")
    if any(
        left.startswith(f"{right}.") or right.startswith(f"{left}.")
        for index, left in enumerate(prefixes)
        for right in prefixes[index + 1 :]
    ):
        raise ValueError("ASAM trainable prefixes cannot overlap")
    named = tuple(model.named_parameters())
    selected_names = {
        name
        for name, _ in named
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for prefix in prefixes:
        if not any(
            name == prefix or name.startswith(f"{prefix}.") for name, _ in named
        ):
            raise ValueError(f"ASAM trainable prefix has no parameters: {prefix}")
    for _, parameter in named:
        parameter.requires_grad_(False)
    selected: list[nn.Parameter] = []
    for name, parameter in named:
        if name in selected_names:
            parameter.requires_grad_(True)
            selected.append(parameter)
    if not selected:
        raise ValueError("ASAM stage has no trainable parameters")
    return tuple(selected)


@torch.no_grad()
def asam_perturb(
    parameters: tuple[nn.Parameter, ...],
    *,
    rho: float,
    eta: float,
    minimum_parameter_ndim: int = 0,
) -> ASAMPerturbation:
    if rho <= 0 or eta <= 0:
        raise ValueError("ASAM rho and eta must be positive")
    minimum_ndim = int(minimum_parameter_ndim)
    if minimum_ndim < 0:
        raise ValueError("ASAM minimum parameter ndim must be nonnegative")
    selected = tuple(
        parameter
        for parameter in parameters
        if parameter.requires_grad and parameter.ndim >= minimum_ndim
    )
    if not selected or any(parameter.grad is None for parameter in selected):
        raise ValueError("every ASAM parameter must have a gradient")
    scaled_gradients: list[torch.Tensor] = []
    for parameter in selected:
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("ASAM gradient is non-finite")
        scaled_gradients.append((parameter.detach().float().abs() + eta) * gradient)
    gradient_norm = torch.sqrt(
        sum(value.square().sum() for value in scaled_gradients)
    ).clamp_min(1.0e-12)
    originals = tuple(parameter.detach().clone() for parameter in selected)
    perturbation_square = torch.zeros((), device=selected[0].device)
    scale = float(rho) / gradient_norm
    for parameter in selected:
        weight_scale = parameter.detach().float().abs() + eta
        perturbation = weight_scale.square() * parameter.grad.detach().float() * scale
        parameter.add_(perturbation.to(parameter.dtype))
        perturbation_square += perturbation.square().sum()
    return ASAMPerturbation(
        parameters=selected,
        originals=originals,
        gradient_norm=float(gradient_norm),
        perturbation_norm=float(torch.sqrt(perturbation_square)),
    )


def asam_rho_for_update(
    update: int,
    *,
    total_updates: int,
    maximum_rho: float,
    minimum_rho: float,
    warmup_updates: int = 0,
) -> float:
    """Warm up the ASAM radius, then decay it to a nonzero polishing floor."""

    current = int(update)
    total = int(total_updates)
    warmup = int(warmup_updates)
    maximum = float(maximum_rho)
    minimum = float(minimum_rho)
    if total <= 0 or current <= 0 or current > total:
        raise ValueError("ASAM update must lie inside the registered horizon")
    if warmup < 0 or warmup >= total:
        raise ValueError("ASAM warmup must be nonnegative and shorter than the horizon")
    if not 0.0 < minimum <= maximum or not all(
        math.isfinite(value) for value in (minimum, maximum)
    ):
        raise ValueError("ASAM radius bounds are invalid")
    if warmup and current <= warmup:
        progress = current / warmup
        return minimum + (maximum - minimum) * progress
    decay_steps = total - warmup
    if decay_steps == 1:
        return maximum
    decay_progress = (current - warmup - 1) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return minimum + (maximum - minimum) * cosine


def asam_adamw_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    weight_decay: float,
    learning_rate: float | None = None,
    learning_rates_by_prefix: Mapping[str, float] | None = None,
) -> tuple[dict[str, object], ...]:
    """Apply decay only to matrix-like weights used by the ASAM base optimizer."""

    decay_value = float(weight_decay)
    if decay_value < 0.0 or not math.isfinite(decay_value):
        raise ValueError("ASAM weight decay must be finite and nonnegative")
    default_lr = None if learning_rate is None else float(learning_rate)
    if default_lr is not None and (
        not math.isfinite(default_lr) or default_lr <= 0.0
    ):
        raise ValueError("ASAM default learning rate must be finite and positive")
    raw_rates = dict(learning_rates_by_prefix or {})
    prefix_rates = {str(name): float(value) for name, value in raw_rates.items()}
    if any(not name for name in prefix_rates) or any(
        not math.isfinite(value) or value <= 0.0 for value in prefix_rates.values()
    ):
        raise ValueError("ASAM prefix learning rates must be finite and positive")
    if prefix_rates and default_lr is None:
        raise ValueError("ASAM prefix learning rates require a default learning rate")
    ordered_prefixes = tuple(sorted(prefix_rates, key=len, reverse=True))
    buckets: dict[tuple[str, str], list[nn.Parameter]] = {}
    matched_prefixes: set[str] = set()
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"duplicate ASAM optimizer parameter: {name}")
        seen.add(identity)
        decay_class = (
            "decay"
            if parameter.ndim >= 2 and not str(name).endswith(".bias")
            else "no_decay"
        )
        selected_prefix = next(
            (
                prefix
                for prefix in ordered_prefixes
                if str(name) == prefix or str(name).startswith(f"{prefix}.")
            ),
            None,
        )
        if selected_prefix is not None:
            matched_prefixes.add(selected_prefix)
        rate_name = "default" if selected_prefix is None else selected_prefix
        buckets.setdefault((rate_name, decay_class), []).append(parameter)
    if not buckets:
        raise ValueError("ASAM optimizer has no trainable parameters")
    unmatched = sorted(set(prefix_rates) - matched_prefixes)
    if unmatched:
        raise ValueError(f"ASAM learning-rate prefix has no parameters: {unmatched}")
    groups: list[dict[str, object]] = []
    for (rate_name, decay_class), parameters in buckets.items():
        group_name = (
            decay_class
            if not prefix_rates
            else f"{rate_name}_{decay_class}"
        )
        group: dict[str, object] = {
            "params": tuple(parameters),
            "weight_decay": decay_value if decay_class == "decay" else 0.0,
            "group_name": group_name,
        }
        if default_lr is not None:
            rate = prefix_rates.get(rate_name, default_lr)
            group["lr"] = rate
            group["initial_lr"] = rate
        groups.append(group)
    return tuple(groups)


def build_asam_adamw(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.99),
    eps: float = 1.0e-8,
    learning_rates_by_prefix: Mapping[str, float] | None = None,
) -> torch.optim.AdamW:
    """Build a fresh AdamW base optimizer suitable for late ASAM refinement."""

    lr = float(learning_rate)
    beta1, beta2 = (float(value) for value in betas)
    epsilon = float(eps)
    if lr <= 0.0 or not math.isfinite(lr):
        raise ValueError("ASAM learning rate must be finite and positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("ASAM AdamW betas must lie in [0, 1)")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("ASAM AdamW epsilon must be finite and positive")
    groups = asam_adamw_parameter_groups(
        named_parameters,
        weight_decay=float(weight_decay),
        learning_rate=lr,
        learning_rates_by_prefix=learning_rates_by_prefix,
    )
    return torch.optim.AdamW(groups, lr=lr, betas=(beta1, beta2), eps=epsilon)


@dataclass(frozen=True)
class ASAMValidationDecision:
    accepted: bool
    score_improved: bool
    family_safe: bool
    learning_rate_multiplier: float
    rho_multiplier: float


def asam_validation_decision(
    *,
    accepted_score: float,
    candidate_score: float,
    accepted_family: Mapping[str, float],
    candidate_family: Mapping[str, float],
    family_regression_tolerance: float,
    learning_rate_multiplier: float = 1.0,
    rho_multiplier: float = 1.0,
    rejection_backoff: float = 0.5,
    minimum_multiplier: float = 1.0 / 16.0,
) -> ASAMValidationDecision:
    """Accept only a strict same-protocol improvement with no family regression."""

    previous = float(accepted_score)
    current = float(candidate_score)
    tolerance = float(family_regression_tolerance)
    lr_multiplier = float(learning_rate_multiplier)
    sharpness_multiplier = float(rho_multiplier)
    backoff = float(rejection_backoff)
    floor = float(minimum_multiplier)
    scalars = (previous, current, tolerance, lr_multiplier, sharpness_multiplier, backoff, floor)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("ASAM validation controls must be finite")
    if previous <= 0.0 or current < 0.0 or tolerance < 0.0:
        raise ValueError("ASAM validation scores or tolerance are invalid")
    if not 0.0 < lr_multiplier <= 1.0 or not 0.0 < sharpness_multiplier <= 1.0:
        raise ValueError("ASAM validation multipliers are invalid")
    if not 0.0 < backoff < 1.0 or not 0.0 < floor <= 1.0:
        raise ValueError("ASAM validation backoff is invalid")
    families = ("uniform", "layered", "marmousi")
    if any(name not in accepted_family or name not in candidate_family for name in families):
        raise ValueError("ASAM validation is missing a medium family")
    family_values = tuple(
        float(mapping[name])
        for mapping in (accepted_family, candidate_family)
        for name in families
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in family_values):
        raise ValueError("ASAM family validation scores are invalid")
    score_improved = current < previous
    family_safe = all(
        float(candidate_family[name])
        <= (1.0 + tolerance) * float(accepted_family[name])
        for name in families
    )
    accepted = score_improved and family_safe
    if accepted:
        next_lr = lr_multiplier
        next_rho = sharpness_multiplier
    else:
        next_lr = max(floor, lr_multiplier * backoff)
        next_rho = max(floor, sharpness_multiplier * backoff)
    return ASAMValidationDecision(
        accepted=accepted,
        score_improved=score_improved,
        family_safe=family_safe,
        learning_rate_multiplier=next_lr,
        rho_multiplier=next_rho,
    )


__all__ = [
    "ASAMPerturbation",
    "ASAMValidationDecision",
    "asam_adamw_parameter_groups",
    "asam_perturb",
    "asam_rho_for_update",
    "asam_validation_decision",
    "build_asam_adamw",
    "freeze_for_asam",
]
