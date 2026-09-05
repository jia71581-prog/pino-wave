from __future__ import annotations

import torch
from collections.abc import Callable


def _normalized_importance_weights(probabilities: torch.Tensor, maximum: float = 20.0) -> torch.Tensor:
    inverse = probabilities.clamp_min(1.0e-8).reciprocal().clamp(max=maximum)
    return inverse / inverse.mean().clamp_min(1.0e-12)


def query_pressure_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    probabilities: torch.Tensor | None = None,
    charbonnier_epsilon: float = 1.0e-3,
    mse_fraction: float = 0.5,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    scale = target.square().mean().sqrt().detach().clamp_min(1.0e-12)
    error = (prediction - target) / scale
    charbonnier = torch.sqrt(error.square() + charbonnier_epsilon**2) - charbonnier_epsilon
    point_loss = (1.0 - mse_fraction) * charbonnier + mse_fraction * error.square()
    if probabilities is not None:
        if probabilities.shape != point_loss.shape:
            raise ValueError("probabilities must match query loss shape")
        point_loss = point_loss * _normalized_importance_weights(probabilities)
    return point_loss.mean()


def normalized_trace_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("trace shapes differ")
    numerator = (prediction - target).square().sum(dim=-1).sqrt()
    denominator = target.square().sum(dim=-1).sqrt().clamp_min(1.0e-6)
    return (numerator / denominator).mean()


def spectral_trace_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    phase_weight: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("trace shapes differ")
    predicted_spectrum = torch.fft.rfft(prediction, dim=-1)
    target_spectrum = torch.fft.rfft(target, dim=-1)
    magnitude = (
        torch.log1p(predicted_spectrum.abs()) - torch.log1p(target_spectrum.abs())
    ).square().mean()
    predicted_phase = predicted_spectrum / predicted_spectrum.abs().clamp_min(1.0e-8)
    target_phase = target_spectrum / target_spectrum.abs().clamp_min(1.0e-8)
    valid = target_spectrum.abs() > 1.0e-8
    phase_error = (predicted_phase - target_phase).abs().square()
    phase = phase_error[valid].mean() if bool(valid.any()) else magnitude.new_zeros(())
    return magnitude + phase_weight * phase


def wave_equation_residual(
    pressure: torch.Tensor,
    query_coords: torch.Tensor,
    velocity_mps: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``p_tt - c²(p_xx+p_zz)`` using continuous-query autograd."""

    if pressure.shape != query_coords.shape[:-1] or query_coords.shape[-1] != 3:
        raise ValueError("pressure/query coordinate shapes differ")
    first = torch.autograd.grad(
        pressure.sum(), query_coords, create_graph=True, retain_graph=True
    )[0]
    second_x = torch.autograd.grad(
        first[..., 0].sum(), query_coords, create_graph=True, retain_graph=True
    )[0][..., 0]
    second_z = torch.autograd.grad(
        first[..., 1].sum(), query_coords, create_graph=True, retain_graph=True
    )[0][..., 1]
    second_t = torch.autograd.grad(
        first[..., 2].sum(), query_coords, create_graph=True, retain_graph=True
    )[0][..., 2]
    return second_t - velocity_mps.square() * (second_x + second_z)


def finite_difference_wave_residual(
    query_pressure: Callable[[torch.Tensor], torch.Tensor],
    query_coords: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    dt_s: float,
) -> torch.Tensor:
    """Parameter-differentiable central-difference wave residual for model training."""

    if min(dx_m, dz_m, dt_s) <= 0.0:
        raise ValueError("finite-difference spacings must be positive")

    def shifted(axis: int, amount: float) -> torch.Tensor:
        offset = torch.zeros_like(query_coords)
        offset[..., axis] = amount
        return query_pressure(query_coords + offset)

    center = query_pressure(query_coords)
    p_x_plus, p_x_minus = shifted(0, dx_m), shifted(0, -dx_m)
    p_z_plus, p_z_minus = shifted(1, dz_m), shifted(1, -dz_m)
    p_t_plus, p_t_minus = shifted(2, dt_s), shifted(2, -dt_s)
    p_xx = (p_x_plus - 2.0 * center + p_x_minus) / (dx_m * dx_m)
    p_zz = (p_z_plus - 2.0 * center + p_z_minus) / (dz_m * dz_m)
    p_tt = (p_t_plus - 2.0 * center + p_t_minus) / (dt_s * dt_s)
    return p_tt - velocity_mps.square() * (p_xx + p_zz)


def pde_residual_loss(
    pressure: torch.Tensor,
    query_coords: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    source_coords: torch.Tensor | None = None,
    source_exclusion_radius_m: float = 0.0,
    domain_margins: tuple[float, float, float] | None = None,
    domain_extent: tuple[float, float] | None = None,
) -> torch.Tensor:
    residual = wave_equation_residual(pressure, query_coords, velocity_mps)
    mask = torch.ones_like(residual, dtype=torch.bool)
    if source_coords is not None and source_exclusion_radius_m > 0.0:
        distance = torch.linalg.vector_norm(query_coords[..., :2] - source_coords[..., None, :2], dim=-1)
        mask &= distance >= source_exclusion_radius_m
    if domain_margins is not None:
        if domain_extent is None:
            raise ValueError("domain_extent is required with domain_margins")
        side, bottom, top = domain_margins
        x, z = query_coords[..., 0], query_coords[..., 1]
        mask &= (x >= side) & (x <= domain_extent[0] - side)
        mask &= (z >= top) & (z <= domain_extent[1] - bottom)
    if not bool(mask.any()):
        raise ValueError("PDE exclusion mask removed every query")
    return residual[mask].square().mean()
