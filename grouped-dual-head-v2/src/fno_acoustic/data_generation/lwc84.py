from __future__ import annotations

import torch

from fno_acoustic.numerics.drp_coefficients import DRPCoefficients

from .stencils import laplacian8, laplacian_symmetric


def _validate_medium(field: torch.Tensor, velocity_mps: torch.Tensor) -> None:
    if field.shape != velocity_mps.shape:
        raise ValueError(f"field and velocity shapes differ: {field.shape} != {velocity_mps.shape}")
    if not torch.isfinite(velocity_mps).all():
        raise ValueError("velocity must be finite")
    if torch.any(velocity_mps <= 0.0):
        raise ValueError("velocity must be positive")


def apply_l_operator(
    field: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    boundary: str = "free_surface",
    spatial_coefficients: DRPCoefficients | None = None,
) -> torch.Tensor:
    _validate_medium(field, velocity_mps)
    if spatial_coefficients is None:
        laplacian = laplacian8(
            field,
            dx_m=float(dx_m),
            dz_m=float(dz_m),
            boundary=boundary,
        )
    else:
        laplacian = laplacian_symmetric(
            field,
            dx_m=float(dx_m),
            dz_m=float(dz_m),
            coefficients=spatial_coefficients,
            boundary=boundary,
        )
    return velocity_mps.square() * laplacian


def lwc84_startup(
    *,
    p0: torch.Tensor,
    pt0: torch.Tensor,
    q0: torch.Tensor,
    qt0: torch.Tensor,
    qtt0: torch.Tensor,
    velocity_mps: torch.Tensor,
    dx_m: float,
    dz_m: float,
    dt_s: float,
    boundary: str = "free_surface",
    spatial_coefficients: DRPCoefficients | None = None,
) -> torch.Tensor:
    _validate_medium(p0, velocity_mps)
    for name, value in {"pt0": pt0, "q0": q0, "qt0": qt0, "qtt0": qtt0}.items():
        if value.shape != p0.shape:
            raise ValueError(f"{name} shape differs from p0")
    dt = float(dt_s)
    if dt <= 0.0:
        raise ValueError("dt_s must be positive")
    acceleration0 = apply_l_operator(
        p0,
        velocity_mps,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
        spatial_coefficients=spatial_coefficients,
    ) + q0
    third0 = apply_l_operator(
        pt0,
        velocity_mps,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
        spatial_coefficients=spatial_coefficients,
    ) + qt0
    fourth0 = apply_l_operator(
        acceleration0,
        velocity_mps,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
        spatial_coefficients=spatial_coefficients,
    ) + qtt0
    return p0 + dt * pt0 + 0.5 * dt**2 * acceleration0 + dt**3 * third0 / 6.0 + dt**4 * fourth0 / 24.0


def lwc84_step(
    *,
    p_nm1: torch.Tensor,
    p_n: torch.Tensor,
    q_n: torch.Tensor,
    qtt_n: torch.Tensor,
    velocity_mps: torch.Tensor,
    dx_m: float,
    dz_m: float,
    dt_s: float,
    boundary: str = "free_surface",
    spatial_coefficients: DRPCoefficients | None = None,
) -> torch.Tensor:
    _validate_medium(p_n, velocity_mps)
    if p_nm1.shape != p_n.shape or q_n.shape != p_n.shape or qtt_n.shape != p_n.shape:
        raise ValueError("all LWC-84 state and source arrays must have identical shapes")
    dt = float(dt_s)
    acceleration = apply_l_operator(
        p_n,
        velocity_mps,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
        spatial_coefficients=spatial_coefficients,
    ) + q_n
    fourth = apply_l_operator(
        acceleration,
        velocity_mps,
        dx_m=dx_m,
        dz_m=dz_m,
        boundary=boundary,
        spatial_coefficients=spatial_coefficients,
    ) + qtt_n
    return 2.0 * p_n - p_nm1 + dt**2 * acceleration + dt**4 * fourth / 12.0
