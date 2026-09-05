"""Leakage-safe frequency-domain Transfer DG instance-adaptation core.

The module deliberately does not lift a predicted flux through the rejected
fixed skeleton projector.  DG defects only select coefficients in an already
learned pressure-correction basis.  The online API accepts exactly two adjacent
onset frames and never accepts a future target tensor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch


@dataclass(frozen=True)
class TransferDGConfig:
    dx_m: float = 10.0
    dz_m: float = 10.0
    element_intervals: int = 20
    boundary_margin: int = 1
    volume_stride: int = 4
    observed_weight: float = 1.0
    volume_weight: float = 0.1
    flux_weight: float = 0.1
    interface_floor: float = 0.05
    ridge_weight: float = 1.0e-4
    trust_ratio: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.dx_m,
            self.dz_m,
            self.element_intervals,
            self.boundary_margin,
            self.volume_stride,
            self.ridge_weight,
            self.trust_ratio,
        )
        if any(float(value) <= 0.0 for value in positive):
            raise ValueError("Transfer DG spacings, sizes, ridge, and trust ratio must be positive")
        if min(self.observed_weight, self.volume_weight, self.flux_weight) < 0.0:
            raise ValueError("Transfer DG objective weights must be nonnegative")
        if not 0.0 <= self.interface_floor <= 1.0:
            raise ValueError("interface floor must lie in [0,1]")


@dataclass(frozen=True)
class DGDefect:
    volume: torch.Tensor
    flux_x: torch.Tensor
    flux_z: torch.Tensor
    interface_x: torch.Tensor
    interface_z: torch.Tensor


@dataclass(frozen=True)
class TransferDGLinearSystem:
    design: torch.Tensor
    residual: torch.Tensor
    parent_objective: torch.Tensor
    observation_scale: torch.Tensor
    volume_scale: torch.Tensor | None
    flux_scale: torch.Tensor | None

    def objective(self, coefficients: torch.Tensor, *, ridge_weight: float) -> torch.Tensor:
        value = self.design @ coefficients + self.residual
        return value.square().sum() + float(ridge_weight) * coefficients.square().sum()


@dataclass(frozen=True)
class TransferDGAdaptationResult:
    coefficients: torch.Tensor
    candidate: torch.Tensor
    accepted: bool
    parent_objective: float
    candidate_objective: float
    correction_ratio: float
    condition_number: float
    future_truth_used: bool = False


def pairs_to_complex(value: torch.Tensor) -> torch.Tensor:
    """Convert ``[...,2,Z,X]`` real/imag pairs to a complex tensor."""
    tensor = torch.as_tensor(value)
    if tensor.ndim < 3 or tensor.shape[-3] != 2:
        raise ValueError("complex pairs must have a two-channel axis before [z,x]")
    if not tensor.dtype.is_floating_point:
        tensor = tensor.float()
    return torch.complex(tensor.select(-3, 0), tensor.select(-3, 1))


def complex_to_pairs(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_complex():
        raise ValueError("value must be complex")
    return torch.stack((tensor.real, tensor.imag), dim=-3)


def hard_free_surface_pairs(value: torch.Tensor) -> torch.Tensor:
    """Return a copy with pressure exactly zero on the physical top surface."""
    tensor = torch.as_tensor(value).clone()
    if tensor.shape[-3] != 2:
        raise ValueError("free-surface projection expects complex pairs")
    tensor[..., 0, :] = 0.0
    return tensor


def validate_two_onset_frames(
    observed: torch.Tensor,
    observed_indices: Sequence[int],
    *,
    spatial_shape: tuple[int, int],
) -> tuple[int, int]:
    indices = tuple(int(value) for value in observed_indices)
    if len(indices) != 2 or indices[0] < 0 or indices[1] != indices[0] + 1:
        raise ValueError("online Transfer DG requires exactly two adjacent onset frames")
    value = torch.as_tensor(observed)
    if tuple(value.shape) != (2, *spatial_shape):
        raise ValueError("observed onset tensor must be [2,z,x]")
    if not torch.isfinite(value).all():
        raise ValueError("observed onset frames must be finite")
    return indices


def _broadcast_frequency(
    frequency_hz: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    frequency = torch.as_tensor(
        frequency_hz, device=target.device, dtype=target.real.dtype
    )
    if frequency.ndim != 1 or frequency.numel() != target.shape[-3]:
        raise ValueError("frequency_hz must match the Fourier axis")
    shape = (1,) * (target.ndim - 3) + (frequency.numel(), 1, 1)
    return frequency.reshape(shape)


def helmholtz_volume_residual(
    pressure_pairs: torch.Tensor,
    velocity_mps: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    source_pairs: torch.Tensor | None = None,
    dx_m: float = 10.0,
    dz_m: float = 10.0,
    boundary_margin: int = 1,
) -> torch.Tensor:
    """Evaluate ``Laplacian(p) + (omega/c)^2 p - source`` in the interior."""
    pressure = pairs_to_complex(pressure_pairs)
    if pressure.ndim not in (3, 4):
        raise ValueError("pressure pairs must be [F,2,Z,X] or [K,F,2,Z,X]")
    velocity = torch.as_tensor(
        velocity_mps, device=pressure.device, dtype=pressure.real.dtype
    )
    if velocity.ndim != 2 or tuple(velocity.shape) != tuple(pressure.shape[-2:]):
        raise ValueError("velocity must be [Z,X] and match pressure")
    if min(dx_m, dz_m) <= 0.0 or boundary_margin < 1:
        raise ValueError("positive spacings and at least one boundary-margin node are required")
    center = pressure[..., 1:-1, 1:-1]
    laplacian = (
        (pressure[..., 1:-1, 2:] - 2.0 * center + pressure[..., 1:-1, :-2])
        / float(dx_m * dx_m)
        + (pressure[..., 2:, 1:-1] - 2.0 * center + pressure[..., :-2, 1:-1])
        / float(dz_m * dz_m)
    )
    omega = 2.0 * math.pi * _broadcast_frequency(frequency_hz, pressure)
    wave_number_square = (omega / velocity[1:-1, 1:-1]) ** 2
    residual = laplacian + wave_number_square * center
    if source_pairs is not None:
        source = pairs_to_complex(torch.as_tensor(source_pairs, device=pressure.device))
        if source.shape != pressure.shape:
            raise ValueError("source pairs must match pressure pairs")
        residual = residual - source[..., 1:-1, 1:-1]
    extra = int(boundary_margin) - 1
    if extra:
        if residual.shape[-2] <= 2 * extra or residual.shape[-1] <= 2 * extra:
            raise ValueError("boundary margin removes the complete residual grid")
        residual = residual[..., extra:-extra, extra:-extra]
    return residual


def _interface_weight(left: torch.Tensor, right: torch.Tensor, floor: float) -> torch.Tensor:
    reflection = (right - left).abs() / (right + left).abs().clamp_min(1.0)
    return float(floor) + (1.0 - float(floor)) * reflection.clamp(0.0, 1.0)


def dg_normal_flux_jumps(
    pressure_pairs: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dx_m: float = 10.0,
    dz_m: float = 10.0,
    element_intervals: int = 20,
    boundary_margin: int = 1,
    interface_floor: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one-sided derivative jumps and coefficient-derived face weights."""
    pressure = pairs_to_complex(pressure_pairs)
    velocity = torch.as_tensor(
        velocity_mps, device=pressure.device, dtype=pressure.real.dtype
    )
    if velocity.ndim != 2 or tuple(velocity.shape) != tuple(pressure.shape[-2:]):
        raise ValueError("velocity must match pressure")
    height, width = velocity.shape
    size, margin = int(element_intervals), int(boundary_margin)
    if size <= 0 or margin < 1 or (height - 1) % size or (width - 1) % size:
        raise ValueError("element intervals must partition the physical grid")
    vertical, vertical_weight = [], []
    for x in range(size, width - 1, size):
        if x - 1 < margin or x + 1 >= width - margin:
            continue
        left = (pressure[..., margin:-margin, x] - pressure[..., margin:-margin, x - 1]) / float(dx_m)
        right = (pressure[..., margin:-margin, x + 1] - pressure[..., margin:-margin, x]) / float(dx_m)
        vertical.append(right - left)
        weight = _interface_weight(
            velocity[margin:-margin, x - 1],
            velocity[margin:-margin, x + 1],
            interface_floor,
        )
        vertical_weight.append(weight)
    horizontal, horizontal_weight = [], []
    for z in range(size, height - 1, size):
        if z - 1 < margin or z + 1 >= height - margin:
            continue
        top = (pressure[..., z, margin:-margin] - pressure[..., z - 1, margin:-margin]) / float(dz_m)
        bottom = (pressure[..., z + 1, margin:-margin] - pressure[..., z, margin:-margin]) / float(dz_m)
        horizontal.append(bottom - top)
        weight = _interface_weight(
            velocity[z - 1, margin:-margin],
            velocity[z + 1, margin:-margin],
            interface_floor,
        )
        horizontal_weight.append(weight)
    if not vertical or not horizontal:
        raise ValueError("DG partition has no internal faces")
    lead = pressure.shape[:-3]
    frequency_count = pressure.shape[-3]
    flux_x = torch.stack(vertical, dim=-2).reshape(*lead, frequency_count, -1)
    flux_z = torch.stack(horizontal, dim=-2).reshape(*lead, frequency_count, -1)
    weight_x = torch.stack(vertical_weight).reshape(-1)
    weight_z = torch.stack(horizontal_weight).reshape(-1)
    return flux_x, flux_z, weight_x, weight_z


def frequency_dg_defect(
    pressure_pairs: torch.Tensor,
    velocity_mps: torch.Tensor,
    frequency_hz: torch.Tensor,
    *,
    source_pairs: torch.Tensor | None = None,
    config: TransferDGConfig = TransferDGConfig(),
) -> DGDefect:
    volume = helmholtz_volume_residual(
        pressure_pairs,
        velocity_mps,
        frequency_hz,
        source_pairs=source_pairs,
        dx_m=config.dx_m,
        dz_m=config.dz_m,
        boundary_margin=config.boundary_margin,
    )
    flux_x, flux_z, interface_x, interface_z = dg_normal_flux_jumps(
        pressure_pairs,
        velocity_mps,
        dx_m=config.dx_m,
        dz_m=config.dz_m,
        element_intervals=config.element_intervals,
        boundary_margin=config.boundary_margin,
        interface_floor=config.interface_floor,
    )
    return DGDefect(volume, flux_x, flux_z, interface_x, interface_z)


def _real_vector(value: torch.Tensor, physical_dims: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    flattened_real = tensor.real.flatten(start_dim=-physical_dims)
    flattened_imag = tensor.imag.flatten(start_dim=-physical_dims)
    return torch.cat((flattened_real, flattened_imag), dim=-1)


def dg_feature_groups(
    defect: DGDefect,
    *,
    volume_stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    stride = int(volume_stride)
    if stride <= 0:
        raise ValueError("volume stride must be positive")
    volume = defect.volume[..., ::stride, ::stride]
    volume_vector = _real_vector(volume, 3)
    weighted_x = defect.flux_x * defect.interface_x.sqrt()
    weighted_z = defect.flux_z * defect.interface_z.sqrt()
    flux_vector = torch.cat(
        (_real_vector(weighted_x, 2), _real_vector(weighted_z, 2)), dim=-1
    )
    return volume_vector, flux_vector


def irfft_observed_frames(
    coefficients: torch.Tensor,
    *,
    time_count: int,
    observed_indices: Sequence[int],
) -> torch.Tensor:
    """Render selected frames from leading retained orthonormal rFFT bins."""
    spectrum = pairs_to_complex(coefficients)
    count = int(time_count)
    full_frequency_count = count // 2 + 1
    if count <= 1 or spectrum.shape[-3] > full_frequency_count:
        raise ValueError("invalid time count or retained frequency count")
    shape = (*spectrum.shape[:-3], full_frequency_count, *spectrum.shape[-2:])
    full = torch.zeros(shape, dtype=spectrum.dtype, device=spectrum.device)
    full[..., : spectrum.shape[-3], :, :] = spectrum
    time_field = torch.fft.irfft(full, n=count, dim=-3, norm="ortho")
    index = torch.tensor(tuple(int(v) for v in observed_indices), device=spectrum.device)
    return torch.index_select(time_field, -3, index)


def _normalized_group(
    parent: torch.Tensor,
    basis: torch.Tensor,
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = parent.square().mean().sqrt().detach().clamp_min(1.0e-8)
    normalizer = math.sqrt(max(parent.numel(), 1)) * scale
    residual = parent / normalizer * math.sqrt(float(weight))
    design = basis.T / normalizer * math.sqrt(float(weight))
    return design, residual, scale


def build_transfer_dg_linear_system(
    parent_coefficients: torch.Tensor,
    correction_basis: torch.Tensor,
    velocity_mps: torch.Tensor,
    frequency_hz: torch.Tensor,
    observed_onset: torch.Tensor,
    observed_indices: Sequence[int],
    *,
    time_count: int = 401,
    source_pairs: torch.Tensor | None = None,
    config: TransferDGConfig = TransferDGConfig(),
) -> TransferDGLinearSystem:
    """Build ``min ||A alpha+b||^2 + ridge||alpha||^2`` without future truth."""
    parent = torch.as_tensor(parent_coefficients)
    basis = torch.as_tensor(
        correction_basis, device=parent.device, dtype=parent.dtype
    )
    if parent.ndim != 4 or parent.shape[-3] != 2:
        raise ValueError("parent coefficients must be [F,2,Z,X]")
    if basis.ndim != 5 or tuple(basis.shape[1:]) != tuple(parent.shape):
        raise ValueError("correction basis must be [K,F,2,Z,X]")
    if not torch.isfinite(parent).all() or not torch.isfinite(basis).all():
        raise ValueError("parent and correction basis must be finite")
    indices = validate_two_onset_frames(
        observed_onset,
        observed_indices,
        spatial_shape=tuple(parent.shape[-2:]),
    )
    if indices[1] >= int(time_count):
        raise ValueError("onset indices lie outside the requested time axis")
    observed = torch.as_tensor(observed_onset, device=parent.device, dtype=parent.dtype)
    parent_observed = irfft_observed_frames(
        parent, time_count=time_count, observed_indices=indices
    )
    basis_observed = irfft_observed_frames(
        basis, time_count=time_count, observed_indices=indices
    )
    observation_residual = (parent_observed - observed).reshape(-1)
    observation_basis = basis_observed.reshape(basis.shape[0], -1)
    observation_scale = torch.maximum(
        observed.square().mean().sqrt(), parent_observed.square().mean().sqrt()
    ).detach().clamp_min(1.0e-8)
    observation_normalizer = math.sqrt(max(observation_residual.numel(), 1)) * observation_scale
    designs = [
        observation_basis.T / observation_normalizer * math.sqrt(config.observed_weight)
    ]
    residuals = [
        observation_residual / observation_normalizer * math.sqrt(config.observed_weight)
    ]

    parent_defect = frequency_dg_defect(
        parent,
        velocity_mps,
        frequency_hz,
        source_pairs=source_pairs,
        config=config,
    )
    basis_defect = frequency_dg_defect(
        basis,
        velocity_mps,
        frequency_hz,
        source_pairs=None,
        config=config,
    )
    parent_volume, parent_flux = dg_feature_groups(
        parent_defect, volume_stride=config.volume_stride
    )
    basis_volume, basis_flux = dg_feature_groups(
        basis_defect, volume_stride=config.volume_stride
    )
    volume_scale = flux_scale = None
    if config.volume_weight > 0.0:
        design, residual, volume_scale = _normalized_group(
            parent_volume,
            basis_volume,
            weight=config.volume_weight,
        )
        designs.append(design); residuals.append(residual)
    if config.flux_weight > 0.0:
        design, residual, flux_scale = _normalized_group(
            parent_flux,
            basis_flux,
            weight=config.flux_weight,
        )
        designs.append(design); residuals.append(residual)
    design = torch.cat(designs, dim=0).float()
    residual = torch.cat(residuals, dim=0).float()
    parent_objective = residual.square().sum()
    return TransferDGLinearSystem(
        design=design,
        residual=residual,
        parent_objective=parent_objective,
        observation_scale=observation_scale.float(),
        volume_scale=None if volume_scale is None else volume_scale.float(),
        flux_scale=None if flux_scale is None else flux_scale.float(),
    )


def solve_transfer_dg_system(
    system: TransferDGLinearSystem,
    parent_coefficients: torch.Tensor,
    correction_basis: torch.Tensor,
    *,
    config: TransferDGConfig = TransferDGConfig(),
) -> TransferDGAdaptationResult:
    """Solve the reduced ridge problem and apply an exact parent rollback gate."""
    design, residual = system.design, system.residual
    basis = torch.as_tensor(correction_basis, device=design.device, dtype=design.dtype)
    parent = torch.as_tensor(parent_coefficients, device=design.device, dtype=design.dtype)
    if basis.shape[0] != design.shape[1] or tuple(basis.shape[1:]) != tuple(parent.shape):
        raise ValueError("system and correction basis dimensions disagree")
    normal = design.T @ design
    normal = normal + float(config.ridge_weight) * torch.eye(
        normal.shape[0], device=normal.device, dtype=normal.dtype
    )
    right = -(design.T @ residual)
    factor, info = torch.linalg.cholesky_ex(normal)
    if int(info.max()) == 0:
        coefficients = torch.cholesky_solve(right[:, None], factor)[:, 0]
    else:
        coefficients = torch.linalg.lstsq(normal, right[:, None]).solution[:, 0]
    correction = torch.einsum("k,kfczx->fczx", coefficients, basis)
    correction_ratio = float(
        correction.norm() / parent.norm().clamp_min(1.0e-12)
    )
    if correction_ratio > config.trust_ratio:
        coefficients = coefficients * (config.trust_ratio / correction_ratio)
        correction = torch.einsum("k,kfczx->fczx", coefficients, basis)
        correction_ratio = float(
            correction.norm() / parent.norm().clamp_min(1.0e-12)
        )
    candidate_objective = system.objective(
        coefficients, ridge_weight=config.ridge_weight
    )
    accepted = bool(
        torch.isfinite(candidate_objective)
        and candidate_objective < system.parent_objective
        and correction_ratio <= config.trust_ratio * (1.0 + 1.0e-6)
    )
    if not accepted:
        coefficients = torch.zeros_like(coefficients)
        correction = torch.zeros_like(parent)
        candidate_objective = system.parent_objective
        correction_ratio = 0.0
    candidate = parent + correction
    singular = torch.linalg.svdvals(normal.double())
    condition_number = float(singular.max() / singular.min().clamp_min(1.0e-16))
    return TransferDGAdaptationResult(
        coefficients=coefficients.detach(),
        candidate=candidate.detach(),
        accepted=accepted,
        parent_objective=float(system.parent_objective),
        candidate_objective=float(candidate_objective),
        correction_ratio=correction_ratio,
        condition_number=condition_number,
    )


__all__ = [
    "DGDefect",
    "TransferDGAdaptationResult",
    "TransferDGConfig",
    "TransferDGLinearSystem",
    "build_transfer_dg_linear_system",
    "complex_to_pairs",
    "dg_feature_groups",
    "dg_normal_flux_jumps",
    "frequency_dg_defect",
    "hard_free_surface_pairs",
    "helmholtz_volume_residual",
    "irfft_observed_frames",
    "pairs_to_complex",
    "solve_transfer_dg_system",
    "validate_two_onset_frames",
]
