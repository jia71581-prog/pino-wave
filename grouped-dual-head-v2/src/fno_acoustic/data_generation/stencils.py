from __future__ import annotations

from collections.abc import Sequence

import torch

from .free_surface import pad_pressure_halo
from fno_acoustic.numerics.drp_coefficients import DRPCoefficients


SECOND_DERIVATIVE_8: tuple[float, ...] = (
    -1.0 / 560.0,
    8.0 / 315.0,
    -1.0 / 5.0,
    8.0 / 5.0,
    -205.0 / 72.0,
    8.0 / 5.0,
    -1.0 / 5.0,
    8.0 / 315.0,
    -1.0 / 560.0,
)
FIRST_DERIVATIVE_8: tuple[float, ...] = (
    1.0 / 280.0,
    -4.0 / 105.0,
    1.0 / 5.0,
    -4.0 / 5.0,
    0.0,
    4.0 / 5.0,
    -1.0 / 5.0,
    4.0 / 105.0,
    -1.0 / 280.0,
)
STENCIL_RADIUS = 4


def _periodic_second_derivative(field: torch.Tensor, spacing_m: float, dim: int) -> torch.Tensor:
    result = torch.zeros_like(field)
    for offset, coefficient in zip(range(-STENCIL_RADIUS, STENCIL_RADIUS + 1), SECOND_DERIVATIVE_8):
        result = result + float(coefficient) * torch.roll(field, shifts=-offset, dims=dim)
    return result / float(spacing_m) ** 2


def _periodic_first_derivative(field: torch.Tensor, spacing_m: float, dim: int) -> torch.Tensor:
    result = torch.zeros_like(field)
    for offset, coefficient in zip(range(-STENCIL_RADIUS, STENCIL_RADIUS + 1), FIRST_DERIVATIVE_8):
        result = result + float(coefficient) * torch.roll(field, shifts=-offset, dims=dim)
    return result / float(spacing_m)


def _padded_second_derivatives(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    side_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    radius = STENCIL_RADIUS
    padded = pad_pressure_halo(field, radius=radius, side_mode=side_mode)
    nz, nx = int(field.shape[-2]), int(field.shape[-1])
    dxx = torch.zeros_like(field)
    dzz = torch.zeros_like(field)
    for stencil_index, coefficient in enumerate(SECOND_DERIVATIVE_8):
        dxx = dxx + float(coefficient) * padded[
            ..., radius : radius + nz, stencil_index : stencil_index + nx
        ]
        dzz = dzz + float(coefficient) * padded[
            ..., stencil_index : stencil_index + nz, radius : radius + nx
        ]
    return dxx / float(dx_m) ** 2, dzz / float(dz_m) ** 2


def _validate_symmetric_second_derivative_coefficients(
    coefficients: DRPCoefficients,
) -> None:
    if not isinstance(coefficients, DRPCoefficients):
        raise TypeError("coefficients must be DRPCoefficients")
    if coefficients.radius < 1:
        raise ValueError("coefficient radius must be positive")


def second_derivatives_symmetric(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    coefficients: DRPCoefficients,
    boundary: str = "free_surface",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a registered symmetric second-derivative stencil.

    This is separate from :func:`second_derivatives8` so the production
    Taylor stencil remains the unmodified default.  DRP experiments therefore
    require an explicit coefficient object at every physical L-operator call.
    """

    if not isinstance(field, torch.Tensor) or field.ndim < 2:
        raise TypeError("field must be a torch tensor with trailing [z,x] dimensions")
    if float(dx_m) <= 0.0 or float(dz_m) <= 0.0:
        raise ValueError("dx_m and dz_m must be positive")
    _validate_symmetric_second_derivative_coefficients(coefficients)
    radius = int(coefficients.radius)
    if min(field.shape[-2:]) <= 2 * radius:
        raise ValueError("grid is too small for the configured symmetric stencil")
    stencil = coefficients.coefficients
    if boundary == "periodic":
        dxx = torch.zeros_like(field)
        dzz = torch.zeros_like(field)
        for offset, coefficient in zip(range(-radius, radius + 1), stencil):
            dxx = dxx + float(coefficient) * torch.roll(field, shifts=-offset, dims=-1)
            dzz = dzz + float(coefficient) * torch.roll(field, shifts=-offset, dims=-2)
        return dxx / float(dx_m) ** 2, dzz / float(dz_m) ** 2
    side_mode = (
        "zero"
        if boundary == "free_surface"
        else "replicate"
        if boundary == "replicate"
        else None
    )
    if side_mode is None:
        raise ValueError(f"unsupported boundary mode: {boundary}")
    padded = pad_pressure_halo(field, radius=radius, side_mode=side_mode)
    nz, nx = int(field.shape[-2]), int(field.shape[-1])
    dxx = torch.zeros_like(field)
    dzz = torch.zeros_like(field)
    for stencil_index, coefficient in enumerate(stencil):
        dxx = dxx + float(coefficient) * padded[
            ..., radius : radius + nz, stencil_index : stencil_index + nx
        ]
        dzz = dzz + float(coefficient) * padded[
            ..., stencil_index : stencil_index + nz, radius : radius + nx
        ]
    return dxx / float(dx_m) ** 2, dzz / float(dz_m) ** 2


def second_derivatives8(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    boundary: str = "free_surface",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(field, torch.Tensor) or field.ndim < 2:
        raise TypeError("field must be a torch tensor with trailing [z,x] dimensions")
    if float(dx_m) <= 0.0 or float(dz_m) <= 0.0:
        raise ValueError("dx_m and dz_m must be positive")
    if min(field.shape[-2:]) <= 2 * STENCIL_RADIUS:
        raise ValueError("grid is too small for the radius-four stencil")
    if boundary == "periodic":
        return (
            _periodic_second_derivative(field, float(dx_m), -1),
            _periodic_second_derivative(field, float(dz_m), -2),
        )
    if boundary == "free_surface":
        return _padded_second_derivatives(field, dx_m=float(dx_m), dz_m=float(dz_m), side_mode="zero")
    if boundary == "replicate":
        return _padded_second_derivatives(field, dx_m=float(dx_m), dz_m=float(dz_m), side_mode="replicate")
    raise ValueError(f"unsupported boundary mode: {boundary}")


def first_derivatives8(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    boundary: str = "free_surface",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(field, torch.Tensor) or field.ndim < 2:
        raise TypeError("field must be a torch tensor with trailing [z,x] dimensions")
    if min(field.shape[-2:]) <= 2 * STENCIL_RADIUS:
        raise ValueError("grid is too small for the radius-four stencil")
    if boundary == "periodic":
        return (
            _periodic_first_derivative(field, float(dx_m), -1),
            _periodic_first_derivative(field, float(dz_m), -2),
        )
    side_mode = "zero" if boundary == "free_surface" else "replicate" if boundary == "replicate" else None
    if side_mode is None:
        raise ValueError(f"unsupported boundary mode: {boundary}")
    radius = STENCIL_RADIUS
    padded = pad_pressure_halo(field, radius=radius, side_mode=side_mode)
    nz, nx = int(field.shape[-2]), int(field.shape[-1])
    dx = torch.zeros_like(field)
    dz = torch.zeros_like(field)
    for stencil_index, coefficient in enumerate(FIRST_DERIVATIVE_8):
        dx = dx + float(coefficient) * padded[
            ..., radius : radius + nz, stencil_index : stencil_index + nx
        ]
        dz = dz + float(coefficient) * padded[
            ..., stencil_index : stencil_index + nz, radius : radius + nx
        ]
    return dx / float(dx_m), dz / float(dz_m)


def laplacian8(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    boundary: str = "free_surface",
) -> torch.Tensor:
    dxx, dzz = second_derivatives8(field, dx_m=dx_m, dz_m=dz_m, boundary=boundary)
    return dxx + dzz


def laplacian_symmetric(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    coefficients: DRPCoefficients,
    boundary: str = "free_surface",
) -> torch.Tensor:
    dxx, dzz = second_derivatives_symmetric(
        field,
        dx_m=dx_m,
        dz_m=dz_m,
        coefficients=coefficients,
        boundary=boundary,
    )
    return dxx + dzz


def stencil_coefficients_jsonable() -> Sequence[float]:
    return [float(value) for value in SECOND_DERIVATIVE_8]
