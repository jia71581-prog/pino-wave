"""Muon/AdamW hybrid optimizer utilities.

Muon is reserved for hidden matrix-like weights.  Scale-sensitive parameters
(biases, normalization gains, embeddings, and output heads) remain on AdamW.
The small facade at the bottom deliberately implements checkpoint round trips,
so the hybrid can be used by the resumable full-support trainer as well as by
the original one-off optimizer probe.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn

from saved_time_phase_operator_v4.full_support import (
    BACKBONE_PREFIXES,
    DENSE_PREFIXES,
    GEOMETRY_PREFIXES,
    _parameter_prefix,
    adamw_backend_options,
)


@torch.no_grad()
def zeropower_via_newtonschulz5(gradient: torch.Tensor, *, steps: int = 5) -> torch.Tensor:
    """Approximate the orthogonal factor of a matrix with Newton-Schulz iterations."""

    if gradient.ndim != 2:
        raise ValueError("Muon Newton-Schulz input must be a matrix")
    if gradient.numel() == 0:
        raise ValueError("Muon Newton-Schulz input cannot be empty")
    transposed = gradient.shape[0] > gradient.shape[1]
    value = gradient
    if transposed:
        value = value.T
    original_dtype = value.dtype
    value = value.bfloat16() if value.is_cuda else value.float()
    value = value / (value.norm() + 1.0e-7)
    a, b, c = (3.4445, -4.7750, 2.0315)
    for _ in range(int(steps)):
        gram = value @ value.T
        value = a * value + (b * gram + c * gram @ gram) @ value
    if transposed:
        value = value.T
    return value.to(dtype=original_dtype)


class Muon(torch.optim.Optimizer):
    """Single-process Muon optimizer for matrix-like neural-network weights.

    This implementation is intentionally local and minimal: it supports one GPU
    process, decoupled weight decay, momentum, optional Nesterov momentum, and
    flattening convolutional kernels to matrices.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, object]],
        *,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("Muon learning rate must be positive")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Muon momentum must be in [0, 1)")
        if weight_decay < 0.0:
            raise ValueError("Muon weight_decay must be nonnegative")
        if ns_steps <= 0:
            raise ValueError("Muon ns_steps must be positive")
        defaults = {
            "lr": float(lr),
            "momentum": float(momentum),
            "weight_decay": float(weight_decay),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401 - torch optimizer API
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            nesterov = bool(group["nesterov"])
            ns_steps = int(group["ns_steps"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim < 2:
                    raise ValueError("Muon only supports matrix-like parameters")
                if weight_decay:
                    parameter.mul_(1.0 - lr * weight_decay)
                grad = parameter.grad
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if nesterov else buffer
                matrix = update.reshape(update.shape[0], -1)
                orthogonalized = zeropower_via_newtonschulz5(
                    matrix,
                    steps=ns_steps,
                ).reshape_as(parameter)
                fan_out, fan_in = matrix.shape
                scale = math.sqrt(max(1.0, float(fan_out) / float(fan_in)))
                parameter.add_(orthogonalized, alpha=-lr * scale)
        return loss


def dense_muon_parameter_groups(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable dense-decoder parameters into Muon and AdamW groups."""

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith("dense_decoder."):
            raise ValueError(f"dense Muon probe found unexpected trainable parameter: {name}")
        if name == "dense_decoder.correction_scale":
            continue
        if parameter.ndim >= 2:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)
    if not muon_params:
        raise ValueError("Muon dense probe has no matrix-like parameters")
    return muon_params, adamw_params


class HybridMuonAdamW:
    """Optimizer facade exposing durable Muon + AdamW state as one object."""

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW | None) -> None:
        self.muon = muon
        self.adamw = adamw
        self.param_groups = list(muon.param_groups) + (
            [] if adamw is None else list(adamw.param_groups)
        )

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.muon.step(closure=closure)
        if self.adamw is not None:
            self.adamw.step()
        return loss

    def state_dict(self) -> dict[str, object]:
        return {
            "format": "hybrid_muon_adamw_v1",
            "muon": self.muon.state_dict(),
            "adamw": None if self.adamw is None else self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if state_dict.get("format") != "hybrid_muon_adamw_v1":
            raise ValueError("unexpected hybrid Muon/AdamW optimizer state format")
        self.muon.load_state_dict(state_dict["muon"])
        adamw_state = state_dict.get("adamw")
        if self.adamw is None:
            if adamw_state is not None:
                raise ValueError("checkpoint has AdamW state but optimizer does not")
        else:
            if adamw_state is None:
                raise ValueError("checkpoint is missing hybrid AdamW state")
            self.adamw.load_state_dict(adamw_state)
        # Optimizer.load_state_dict replaces its param-group dictionaries.  Keep
        # the facade's scheduler-facing view attached to those replacements.
        self.param_groups = list(self.muon.param_groups) + (
            [] if self.adamw is None else list(self.adamw.param_groups)
        )


_OUTPUT_HEAD_PREFIXES = (
    "dense_decoder.output.",
    "local_field.output.",
    "local_field.warp.shift_head.2.",
    "fusion.local_output.3.",
)


def parameter_uses_muon(name: str, parameter: nn.Parameter) -> bool:
    """Return whether a parameter is a hidden matrix suitable for Muon."""

    if parameter.ndim < 2 or "embedding" in str(name):
        return False
    if any(str(name).startswith(prefix) for prefix in _OUTPUT_HEAD_PREFIXES):
        return False
    matrix = parameter.reshape(parameter.shape[0], -1)
    return min(int(matrix.shape[0]), int(matrix.shape[1])) > 1


def build_staged_muon_adamw(
    model: nn.Module,
    *,
    dense_lr: float,
    geometry_lr: float,
    backbone_lr: float,
    weight_decay: float,
    temporal_basis_lr: float | None = None,
    expert_lr: float | None = None,
    temporal_operator_lr: float | None = None,
    warp_lr: float | None = None,
    green_kernel_lr: float | None = None,
    temporal_latent_lr: float | None = None,
    multi_arrival_lr: float | None = None,
    dispersive_modal_lr: float | None = None,
    windowed_propagation_lr: float | None = None,
    local_field_lr: float | None = None,
    muon_lr_scale: float = 40.0,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adamw_implementation: str = "single_tensor",
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1.0e-8,
) -> HybridMuonAdamW:
    """Build full-model staged groups with Muon for hidden matrices.

    ``*_lr`` values remain the auxiliary AdamW rates.  Muon uses the same
    stage ratios multiplied by ``muon_lr_scale``; the epoch scheduler and
    validation backoff subsequently scale both optimizer families together.
    """

    learning_rates = {
        "dense": float(dense_lr),
        "geometry": float(geometry_lr),
        "backbone": float(backbone_lr),
    }
    optional_rates = {
        "temporal": temporal_basis_lr,
        "expert": expert_lr,
        "temporal_operator": temporal_operator_lr,
        "warp": warp_lr,
        "green_kernel": green_kernel_lr,
        "temporal_latent": temporal_latent_lr,
        "multi_arrival": multi_arrival_lr,
        "dispersive_modal": dispersive_modal_lr,
        "windowed_propagation": windowed_propagation_lr,
        "local_field": local_field_lr,
    }
    learning_rates.update(
        {name: float(value) for name, value in optional_rates.items() if value is not None}
    )
    if any(value <= 0.0 for value in learning_rates.values()):
        raise ValueError("hybrid optimizer learning rates must be positive")
    if float(muon_lr_scale) <= 0.0:
        raise ValueError("Muon learning-rate scale must be positive")
    if float(weight_decay) < 0.0 or float(adamw_eps) <= 0.0:
        raise ValueError("hybrid weight decay and AdamW epsilon are invalid")
    beta1, beta2 = (float(value) for value in adamw_betas)
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("hybrid AdamW betas must be in [0, 1)")

    prefix_to_group = {
        **{prefix: "dense" for prefix in DENSE_PREFIXES},
        **{prefix: "geometry" for prefix in GEOMETRY_PREFIXES},
        **{prefix: "backbone" for prefix in BACKBONE_PREFIXES},
    }
    grouped: dict[str, dict[str, list[nn.Parameter]]] = {
        name: {"muon": [], "decay": [], "no_decay": []}
        for name in learning_rates
    }
    for name, parameter in model.named_parameters():
        prefix = _parameter_prefix(name)
        if prefix not in prefix_to_group:
            raise ValueError(f"unregistered hybrid optimizer parameter prefix: {prefix}")
        group_name = prefix_to_group[prefix]
        if expert_lr is not None and name.startswith("dense_decoder.family_experts."):
            group_name = "expert"
        elif temporal_basis_lr is not None and name.startswith("dense_decoder.temporal_basis."):
            group_name = "temporal"
        elif temporal_operator_lr is not None and name.startswith("local_field.temporal_operator."):
            group_name = "temporal_operator"
        elif warp_lr is not None and name.startswith("local_field.warp."):
            group_name = "warp"
        elif green_kernel_lr is not None and name.startswith("local_field.green_kernel."):
            group_name = "green_kernel"
        elif temporal_latent_lr is not None and name.startswith("local_field.temporal_latent."):
            group_name = "temporal_latent"
        elif multi_arrival_lr is not None and name.startswith("local_field.multi_arrival."):
            group_name = "multi_arrival"
        elif dispersive_modal_lr is not None and name.startswith("local_field.dispersive_modal."):
            group_name = "dispersive_modal"
        elif windowed_propagation_lr is not None and name.startswith("local_field.windowed_propagation."):
            group_name = "windowed_propagation"
        elif local_field_lr is not None and name.startswith("local_field."):
            group_name = "local_field"
        if parameter_uses_muon(name, parameter):
            parameter_class = "muon"
        elif parameter.ndim <= 1 or name.endswith("bias") or "embedding" in name:
            parameter_class = "no_decay"
        else:
            parameter_class = "decay"
        grouped[group_name][parameter_class].append(parameter)

    missing = [name for name, parts in grouped.items() if not any(parts.values())]
    if missing:
        raise ValueError(f"hybrid optimizer parameter groups are empty: {missing}")
    group_order = tuple(
        name
        for name in (
            "dense", "local_field", "temporal", "temporal_operator", "warp",
            "green_kernel", "temporal_latent", "multi_arrival",
            "dispersive_modal", "windowed_propagation", "expert", "geometry",
            "backbone",
        )
        if name in learning_rates
    )
    muon_groups: list[dict[str, object]] = []
    adamw_groups: list[dict[str, object]] = []
    for name in group_order:
        base_lr = learning_rates[name]
        if grouped[name]["muon"]:
            muon_lr = base_lr * float(muon_lr_scale)
            muon_groups.append(
                {
                    "params": grouped[name]["muon"],
                    "lr": muon_lr,
                    "initial_lr": muon_lr,
                    "group_name": f"{name}_muon",
                    "weight_decay": float(weight_decay),
                }
            )
        for decay_class in ("decay", "no_decay"):
            parameters = grouped[name][decay_class]
            if parameters:
                adamw_groups.append(
                    {
                        "params": parameters,
                        "lr": base_lr,
                        "initial_lr": base_lr,
                        "group_name": f"{name}_adamw_{decay_class}",
                        "weight_decay": float(weight_decay) if decay_class == "decay" else 0.0,
                    }
                )
    if not muon_groups:
        raise ValueError("hybrid optimizer selected no hidden matrices for Muon")
    muon = Muon(
        muon_groups,
        lr=1.0,
        momentum=float(momentum),
        weight_decay=0.0,
        nesterov=bool(nesterov),
        ns_steps=int(ns_steps),
    )
    adamw = torch.optim.AdamW(
        adamw_groups,
        weight_decay=0.0,
        betas=(beta1, beta2),
        eps=float(adamw_eps),
        **adamw_backend_options(adamw_implementation),
    )
    return HybridMuonAdamW(muon, adamw)


def build_dense_muon_hybrid(
    model: nn.Module,
    *,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> HybridMuonAdamW:
    muon_params, adamw_params = dense_muon_parameter_groups(model)
    muon = Muon(
        muon_params,
        lr=float(muon_lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
        nesterov=True,
        ns_steps=int(ns_steps),
    )
    adamw = (
        torch.optim.AdamW(
            [{"params": adamw_params, "lr": float(adamw_lr), "weight_decay": 0.0}],
            weight_decay=0.0,
        )
        if adamw_params
        else None
    )
    return HybridMuonAdamW(muon, adamw)


def build_module_muon_adamw(
    module: nn.Module,
    *,
    adamw_lr: float,
    muon_lr_scale: float,
    weight_decay: float,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1.0e-8,
) -> HybridMuonAdamW:
    """Build a generic hybrid while keeping scale-sensitive heads on AdamW."""

    base_lr = float(adamw_lr)
    scale = float(muon_lr_scale)
    decay = float(weight_decay)
    if base_lr <= 0.0 or scale <= 0.0 or decay < 0.0:
        raise ValueError("generic hybrid optimizer rates are invalid")
    if float(adamw_eps) <= 0.0:
        raise ValueError("generic hybrid AdamW epsilon must be positive")
    beta1, beta2 = (float(value) for value in adamw_betas)
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("generic hybrid AdamW betas must lie in [0,1)")
    output_prefixes = ("spatial_encoder.6.", "temporal_encoder.4.", "trust_head.2.")
    muon_params: list[nn.Parameter] = []
    adamw_decay: list[nn.Parameter] = []
    adamw_no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise ValueError(f"duplicate generic hybrid parameter: {name}")
        seen.add(id(parameter))
        is_output = any(str(name).startswith(prefix) for prefix in output_prefixes)
        matrix = parameter.reshape(parameter.shape[0], -1) if parameter.ndim >= 2 else None
        if (
            not is_output
            and matrix is not None
            and min(int(matrix.shape[0]), int(matrix.shape[1])) > 1
        ):
            muon_params.append(parameter)
        elif parameter.ndim >= 2 and not str(name).endswith(".bias"):
            adamw_decay.append(parameter)
        else:
            adamw_no_decay.append(parameter)
    if not muon_params or not (adamw_decay or adamw_no_decay):
        raise ValueError("generic hybrid optimizer has empty Muon or AdamW partition")
    muon_lr = base_lr * scale
    muon = Muon(
        [{
            "params": muon_params,
            "lr": muon_lr,
            "initial_lr": muon_lr,
            "group_name": "shared_hidden_muon",
            "weight_decay": decay,
        }],
        lr=muon_lr,
        momentum=float(momentum),
        weight_decay=decay,
        nesterov=bool(nesterov),
        ns_steps=int(ns_steps),
    )
    adamw_groups: list[dict[str, object]] = []
    if adamw_decay:
        adamw_groups.append({
            "params": adamw_decay,
            "lr": base_lr,
            "initial_lr": base_lr,
            "group_name": "scale_head_adamw_decay",
            "weight_decay": decay,
        })
    if adamw_no_decay:
        adamw_groups.append({
            "params": adamw_no_decay,
            "lr": base_lr,
            "initial_lr": base_lr,
            "group_name": "scale_head_adamw_no_decay",
            "weight_decay": 0.0,
        })
    adamw = torch.optim.AdamW(
        adamw_groups,
        weight_decay=0.0,
        betas=(beta1, beta2),
        eps=float(adamw_eps),
    )
    return HybridMuonAdamW(muon, adamw)


__all__ = [
    "HybridMuonAdamW",
    "Muon",
    "build_dense_muon_hybrid",
    "build_module_muon_adamw",
    "build_staged_muon_adamw",
    "dense_muon_parameter_groups",
    "parameter_uses_muon",
    "zeropower_via_newtonschulz5",
]
