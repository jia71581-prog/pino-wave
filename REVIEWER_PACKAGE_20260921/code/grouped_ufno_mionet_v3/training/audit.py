"""Accuracy, phase, family, and gradient diagnostics for V3 gates."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES


def _float(value) -> torch.Tensor:
    return torch.as_tensor(value).float()


def zero_prediction_relative_l2(target: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    target = _float(target)
    return target.norm() / target.norm().clamp_min(eps)


def energy_ratio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    prediction = _float(prediction)
    target = _float(target)
    return prediction.square().sum() / target.square().sum().clamp_min(eps)


def optimal_amplitude_scale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    prediction = _float(prediction)
    target = _float(target)
    return (prediction * target).sum() / prediction.square().sum().clamp_min(eps)


def radial_centroid_displacement_m(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    source_xy_m: torch.Tensor,
    x_m: torch.Tensor,
    z_m: torch.Tensor,
    eps: float = 1.0e-20,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = _float(prediction)
    target = _float(target)
    source = _float(source_xy_m).to(prediction.device)
    x = _float(x_m).to(prediction.device)
    z = _float(z_m).to(prediction.device)
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("radial centroid tensors must match [record,time,z,x]")
    if source.shape != (prediction.shape[0], 2) or prediction.shape[-2:] != (len(z), len(x)):
        raise ValueError("source or spatial axes do not match radial centroid tensors")
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    radius = torch.sqrt(
        (xx[None] - source[:, 0, None, None]).square()
        + (zz[None] - source[:, 1, None, None]).square()
    )
    prediction_energy = prediction.square()
    target_energy = target.square()
    prediction_sum = prediction_energy.sum(dim=(-2, -1))
    target_sum = target_energy.sum(dim=(-2, -1))
    valid = (prediction_sum > eps) & (target_sum > eps)
    prediction_centroid = (prediction_energy * radius[:, None]).sum(dim=(-2, -1)) / prediction_sum.clamp_min(eps)
    target_centroid = (target_energy * radius[:, None]).sum(dim=(-2, -1)) / target_sum.clamp_min(eps)
    displacement = (prediction_centroid - target_centroid).abs()
    displacement = torch.where(valid, displacement, torch.full_like(displacement, float("nan")))
    return displacement, valid


def audit_required_gradients(
    groups: Mapping[str, Sequence[nn.Parameter]],
) -> dict[str, int]:
    failures: list[str] = []
    report: dict[str, int] = {}
    for group_name, parameters in groups.items():
        parameters = tuple(parameters)
        if not parameters:
            failures.append(f"{group_name}:empty")
            continue
        report[group_name] = len(parameters)
        for index, parameter in enumerate(parameters):
            gradient = parameter.grad
            if gradient is None:
                failures.append(f"{group_name}[{index}]:missing")
            elif not torch.isfinite(gradient).all():
                failures.append(f"{group_name}[{index}]:nonfinite")
            elif gradient.abs().sum() == 0:
                failures.append(f"{group_name}[{index}]:zero")
    if failures:
        raise RuntimeError("required gradient audit failed: " + ", ".join(failures))
    return report


def assert_report_families(families: Sequence[str]) -> None:
    observed = {str(value) for value in families}
    if "anomaly" in observed:
        raise ValueError("anomaly medium is forbidden in V3 reports")
    unknown = observed - set(ALLOWED_MEDIUM_TYPES)
    if unknown:
        raise ValueError(f"unknown report families: {sorted(unknown)}")
    missing = set(ALLOWED_MEDIUM_TYPES) - observed
    if missing:
        raise ValueError(f"missing V3 report families: {sorted(missing)}")


__all__ = [
    "assert_report_families",
    "audit_required_gradients",
    "energy_ratio",
    "optimal_amplitude_scale",
    "radial_centroid_displacement_m",
    "zero_prediction_relative_l2",
]
