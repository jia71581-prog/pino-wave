"""Fixed train-only DG-skeleton flux projection for oracle capacity tests."""
from __future__ import annotations

import torch

from fno_acoustic.data_generation.stencils import first_derivatives8


def skeleton_hat_basis(
    height: int = 201,
    width: int = 201,
    *,
    element_intervals: int = 20,
    cpml_margin: int = 20,
    device=None,
    dtype=torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return zero-boundary bilinear skeleton basis and its coarse node centers."""
    nz, nx = int(height), int(width)
    spacing, margin = int(element_intervals), int(cpml_margin)
    if (nz - 1) % spacing or (nx - 1) % spacing:
        raise ValueError("element spacing must divide the grid intervals")
    if margin != spacing or nx <= 2 * margin + 2 or nz <= margin + 2:
        raise ValueError("oracle v1 requires one element of CPML margin")
    z_centers = range(spacing, nz - margin - spacing + 1, spacing)
    x_centers = range(margin + spacing, nx - margin - spacing + 1, spacing)
    centers = torch.tensor(
        [(z, x) for z in z_centers for x in x_centers],
        device=device,
        dtype=torch.int64,
    )
    z_axis = torch.arange(nz, device=device, dtype=dtype)
    x_axis = torch.arange(nx, device=device, dtype=dtype)
    rows = []
    for z_center, x_center in centers.tolist():
        z_hat = (1.0 - (z_axis - float(z_center)).abs() / spacing).clamp_min(0.0)
        x_hat = (1.0 - (x_axis - float(x_center)).abs() / spacing).clamp_min(0.0)
        rows.append(z_hat[:, None] * x_hat[None, :])
    if not rows:
        raise RuntimeError("skeleton basis has no free degrees of freedom")
    return torch.stack(rows), centers


def extract_internal_skeleton_flux(
    derivative_x: torch.Tensor,
    derivative_z: torch.Tensor,
    *,
    element_intervals: int = 20,
    cpml_margin: int = 20,
) -> torch.Tensor:
    """Flatten x/z normal derivatives on non-CPML internal coarse faces."""
    dx, dz = torch.as_tensor(derivative_x), torch.as_tensor(derivative_z)
    if dx.shape != dz.shape or dx.ndim < 2:
        raise ValueError("derivative fields must be aligned [...,Z,X]")
    nz, nx = dx.shape[-2:]
    spacing, margin = int(element_intervals), int(cpml_margin)
    vertical_lines = range(
        margin + spacing, nx - margin - spacing + 1, spacing
    )
    horizontal_lines = range(spacing, nz - margin - spacing + 1, spacing)
    pieces = [dx[..., 1 : nz - margin, x] for x in vertical_lines]
    pieces.extend(dz[..., z, margin + 1 : nx - margin] for z in horizontal_lines)
    if not pieces:
        raise RuntimeError("skeleton extraction has no internal faces")
    return torch.cat(pieces, dim=-1)


class FixedSkeletonFluxProjector:
    """Least-squares Neumann-to-pressure lift in a fixed bilinear skeleton space."""

    def __init__(
        self,
        *,
        height: int = 201,
        width: int = 201,
        element_intervals: int = 20,
        cpml_margin: int = 20,
        dx_m: float = 10.0,
        dz_m: float = 10.0,
        pseudoinverse_rcond: float = 1.0e-8,
        device=None,
        dtype=torch.float32,
    ) -> None:
        basis, centers = skeleton_hat_basis(
            height,
            width,
            element_intervals=element_intervals,
            cpml_margin=cpml_margin,
            dtype=torch.float64,
        )
        derivative_x, derivative_z = first_derivatives8(
            basis,
            dx_m=dx_m,
            dz_m=dz_m,
            boundary="free_surface",
        )
        response = extract_internal_skeleton_flux(
            derivative_x,
            derivative_z,
            element_intervals=element_intervals,
            cpml_margin=cpml_margin,
        ).T.contiguous()
        singular = torch.linalg.svdvals(response)
        rank = int(torch.linalg.matrix_rank(response, tol=pseudoinverse_rcond).item())
        if rank != response.shape[1]:
            raise RuntimeError("fixed skeleton response is rank deficient")
        pseudoinverse = torch.linalg.pinv(response, rcond=pseudoinverse_rcond)
        self.basis = basis.to(device=device, dtype=dtype)
        self.pseudoinverse = pseudoinverse.to(device=device, dtype=dtype)
        self.centers = centers.to(device=device)
        self.element_intervals = int(element_intervals)
        self.cpml_margin = int(cpml_margin)
        self.dx_m = float(dx_m)
        self.dz_m = float(dz_m)
        self.report = {
            "degrees_of_freedom": int(response.shape[1]),
            "flux_constraints": int(response.shape[0]),
            "matrix_rank": rank,
            "largest_singular_value": float(singular.max()),
            "smallest_singular_value": float(singular.min()),
            "condition_number": float(singular.max() / singular.min()),
            "pseudoinverse_rcond": float(pseudoinverse_rcond),
        }

    def flux_vector(self, field: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(field)
        derivative_x, derivative_z = first_derivatives8(
            value,
            dx_m=self.dx_m,
            dz_m=self.dz_m,
            boundary="free_surface",
        )
        return extract_internal_skeleton_flux(
            derivative_x,
            derivative_z,
            element_intervals=self.element_intervals,
            cpml_margin=self.cpml_margin,
        )

    def project_flux(self, flux: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(flux, device=self.basis.device, dtype=self.basis.dtype)
        if value.shape[-1] != self.pseudoinverse.shape[1]:
            raise ValueError("flux vector length does not match the skeleton projector")
        coefficients = torch.einsum("...m,dm->...d", value, self.pseudoinverse)
        return torch.einsum("...d,dzx->...zx", coefficients, self.basis)

    def project_field_flux(self, field: torch.Tensor) -> torch.Tensor:
        return self.project_flux(self.flux_vector(field))


__all__ = [
    "FixedSkeletonFluxProjector",
    "extract_internal_skeleton_flux",
    "skeleton_hat_basis",
]
