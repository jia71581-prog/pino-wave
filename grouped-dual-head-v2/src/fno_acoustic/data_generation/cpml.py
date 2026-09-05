from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fno_acoustic.numerics.drp_coefficients import DRPCoefficients

from .grid import AcousticGrid, BoundaryConfig
from .stencils import (
    first_derivatives8,
    second_derivatives8,
    second_derivatives_symmetric,
)


@dataclass(frozen=True)
class CFSCPMLProfiles:
    sigma_x: torch.Tensor
    sigma_z: torch.Tensor
    kappa_x: torch.Tensor
    kappa_z: torch.Tensor
    alpha_x: torch.Tensor
    alpha_z: torch.Tensor
    a_x: torch.Tensor
    a_z: torch.Tensor
    b_x: torch.Tensor
    b_z: torch.Tensor
    inv_kappa_x: torch.Tensor
    inv_kappa_z: torch.Tensor
    active_x: torch.Tensor
    active_z: torch.Tensor
    sigma_max_s_inv: float
    alpha_max_rad_s: float


def _axis_profile(
    n: int,
    *,
    npml: int,
    left: bool,
    right: bool,
    spacing_m: float,
    dt_s: float,
    c_ref_mps: float,
    target_reflection: float,
    polynomial_order: int,
    kappa_max: float,
    alpha_max_rad_s: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    sigma = torch.zeros(n, device=device, dtype=dtype)
    kappa = torch.ones(n, device=device, dtype=dtype)
    alpha = torch.zeros(n, device=device, dtype=dtype)
    thickness_m = float(npml) * float(spacing_m)
    sigma_max = -(int(polynomial_order) + 1.0) * float(c_ref_mps) * math.log(float(target_reflection)) / (2.0 * thickness_m)

    def fill(index: int, depth: float) -> None:
        power = float(depth) ** int(polynomial_order)
        sigma[index] = sigma_max * power
        kappa[index] = 1.0 + (float(kappa_max) - 1.0) * power
        alpha[index] = float(alpha_max_rad_s) * (1.0 - float(depth))

    if left:
        for index in range(int(npml)):
            fill(index, (int(npml) - index) / float(npml))
    if right:
        start = int(n) - int(npml)
        for local in range(int(npml)):
            fill(start + local, (local + 1) / float(npml))
    decay = sigma / kappa + alpha
    b = torch.exp(-decay * float(dt_s))
    denominator = kappa * (sigma + kappa * alpha)
    a = torch.where(denominator > 0.0, sigma * (b - 1.0) / denominator, torch.zeros_like(sigma))
    return sigma, kappa, alpha, a, b, kappa.reciprocal(), sigma > 0.0


def build_cfs_cpml_profiles(
    grid: AcousticGrid,
    boundaries: BoundaryConfig,
    *,
    dt_s: float,
    c_ref_mps: float,
    target_reflection: float = 1.0e-8,
    polynomial_order: int = 3,
    kappa_max: float = 3.0,
    minimum_frequency_hz: float = 8.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CFSCPMLProfiles:
    for name, value in {
        "dt_s": dt_s,
        "c_ref_mps": c_ref_mps,
        "target_reflection": target_reflection,
        "kappa_max": kappa_max,
        "minimum_frequency_hz": minimum_frequency_hz,
    }.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0.0 < float(target_reflection) < 1.0:
        raise ValueError("target_reflection must lie between zero and one")
    device = torch.device(device)
    nz, nx = boundaries.extended_shape(grid)
    alpha_max = math.pi * float(minimum_frequency_hz)
    x = _axis_profile(
        nx,
        npml=boundaries.npml,
        left=True,
        right=True,
        spacing_m=grid.dx_m,
        dt_s=dt_s,
        c_ref_mps=c_ref_mps,
        target_reflection=target_reflection,
        polynomial_order=polynomial_order,
        kappa_max=kappa_max,
        alpha_max_rad_s=alpha_max,
        device=device,
        dtype=dtype,
    )
    z = _axis_profile(
        nz,
        npml=boundaries.npml,
        left=False,
        right=True,
        spacing_m=grid.dz_m,
        dt_s=dt_s,
        c_ref_mps=c_ref_mps,
        target_reflection=target_reflection,
        polynomial_order=polynomial_order,
        kappa_max=kappa_max,
        alpha_max_rad_s=alpha_max,
        device=device,
        dtype=dtype,
    )

    def x2(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(0).expand(nz, nx)

    def z2(value: torch.Tensor) -> torch.Tensor:
        return value.unsqueeze(1).expand(nz, nx)

    return CFSCPMLProfiles(
        sigma_x=x2(x[0]),
        sigma_z=z2(z[0]),
        kappa_x=x2(x[1]),
        kappa_z=z2(z[1]),
        alpha_x=x2(x[2]),
        alpha_z=z2(z[2]),
        a_x=x2(x[3]),
        a_z=z2(z[3]),
        b_x=x2(x[4]),
        b_z=z2(z[4]),
        inv_kappa_x=x2(x[5]),
        inv_kappa_z=z2(z[5]),
        active_x=x2(x[6]),
        active_z=z2(z[6]),
        sigma_max_s_inv=float(max(x[0].max().item(), z[0].max().item())),
        alpha_max_rad_s=alpha_max,
    )


class CFSCPMLOperator:
    """Collocated unsplit CFS-CPML correction around a direct D2 LWC core."""

    def __init__(
        self,
        profiles: CFSCPMLProfiles,
        *,
        dx_m: float,
        dz_m: float,
        physical_second_derivative_coefficients: DRPCoefficients | None = None,
    ) -> None:
        self.profiles = profiles
        self.dx_m = float(dx_m)
        self.dz_m = float(dz_m)
        self.physical_second_derivative_coefficients = (
            physical_second_derivative_coefficients
        )
        self.psi_x: torch.Tensor | None = None
        self.psi_z: torch.Tensor | None = None
        self.phi_x: torch.Tensor | None = None
        self.phi_z: torch.Tensor | None = None

    def _ensure_memory(self, field: torch.Tensor) -> None:
        if self.psi_x is None or self.psi_x.shape != field.shape or self.psi_x.device != field.device or self.psi_x.dtype != field.dtype:
            self.psi_x = torch.zeros_like(field)
            self.psi_z = torch.zeros_like(field)
            self.phi_x = torch.zeros_like(field)
            self.phi_z = torch.zeros_like(field)

    def reset(self) -> None:
        for value in (self.psi_x, self.psi_z, self.phi_x, self.phi_z):
            if value is not None:
                value.zero_()

    def memory_norm(self) -> float:
        values = [value.abs().amax() for value in (self.psi_x, self.psi_z, self.phi_x, self.phi_z) if value is not None]
        return 0.0 if not values else float(torch.stack(values).amax().item())

    def apply(self, field: torch.Tensor, velocity_mps: torch.Tensor, *, update_memory: bool) -> torch.Tensor:
        if field.shape != velocity_mps.shape:
            raise ValueError("field and velocity must have identical shapes")
        self._ensure_memory(field)
        assert self.psi_x is not None and self.psi_z is not None and self.phi_x is not None and self.phi_z is not None
        dpx, dpz = first_derivatives8(field, dx_m=self.dx_m, dz_m=self.dz_m, boundary="free_surface")
        psi_x = self.profiles.b_x * self.psi_x + self.profiles.a_x * dpx
        psi_z = self.profiles.b_z * self.psi_z + self.profiles.a_z * dpz
        stretched_dx = self.profiles.inv_kappa_x * dpx + psi_x
        stretched_dz = self.profiles.inv_kappa_z * dpz + psi_z
        d_stretched_x, _ = first_derivatives8(
            stretched_dx, dx_m=self.dx_m, dz_m=self.dz_m, boundary="free_surface"
        )
        _, d_stretched_z = first_derivatives8(
            stretched_dz, dx_m=self.dx_m, dz_m=self.dz_m, boundary="free_surface"
        )
        phi_x = self.profiles.b_x * self.phi_x + self.profiles.a_x * d_stretched_x
        phi_z = self.profiles.b_z * self.phi_z + self.profiles.a_z * d_stretched_z
        cpml_x = self.profiles.inv_kappa_x * d_stretched_x + phi_x
        cpml_z = self.profiles.inv_kappa_z * d_stretched_z + phi_z
        if self.physical_second_derivative_coefficients is None:
            core_x, core_z = second_derivatives8(
                field,
                dx_m=self.dx_m,
                dz_m=self.dz_m,
                boundary="free_surface",
            )
        else:
            core_x, core_z = second_derivatives_symmetric(
                field,
                dx_m=self.dx_m,
                dz_m=self.dz_m,
                coefficients=self.physical_second_derivative_coefficients,
                boundary="free_surface",
            )
        spatial_x = torch.where(self.profiles.active_x, cpml_x, core_x)
        spatial_z = torch.where(self.profiles.active_z, cpml_z, core_z)
        if update_memory:
            self.psi_x, self.psi_z = psi_x, psi_z
            self.phi_x, self.phi_z = phi_x, phi_z
        return velocity_mps.square() * (spatial_x + spatial_z)

    def apply_repeated_zero_order_hold(
        self,
        field: torch.Tensor,
        velocity_mps: torch.Tensor,
        *,
        substeps: int,
        update_memory: bool,
    ) -> torch.Tensor:
        """Apply ``substeps`` exact CFS recurrences for a held pressure field.

        This is mathematically identical to calling :meth:`apply` repeatedly
        with the same field, including the nested ``psi -> phi`` dependence.
        The substep axis is evaluated as one tensor expression, avoiding Python
        and kernel-launch overhead while preserving all four final memories.
        """

        return self.apply_linear_trajectory(
            field,
            field,
            velocity_mps,
            substeps=substeps,
            update_memory=update_memory,
        )

    def apply_linear_trajectory(
        self,
        previous_field: torch.Tensor,
        field: torch.Tensor,
        velocity_mps: torch.Tensor,
        *,
        substeps: int,
        update_memory: bool,
    ) -> torch.Tensor:
        """Advance CFS memories along a causal linear pressure trajectory.

        The ``j``-th internal field is ``previous + j/N * (field-previous)``
        for ``j=1..N``.  Spatial differentiation is linear, so the complete
        sequence of first derivatives and both nested memory recurrences can
        be evaluated in tensor form.  The returned acceleration corresponds
        to the final field after all ``N`` memory updates.
        """

        count = int(substeps)
        if count <= 0:
            raise ValueError("CPML repeated substeps must be positive")
        if previous_field.shape != field.shape or field.shape != velocity_mps.shape:
            raise ValueError(
                "previous field, field, and velocity must have identical shapes"
            )
        self._ensure_memory(field)
        assert self.psi_x is not None
        assert self.psi_z is not None
        assert self.phi_x is not None
        assert self.phi_z is not None

        leading = (1,) * (field.ndim - 2)

        def expanded(profile: torch.Tensor) -> torch.Tensor:
            return profile.reshape(*leading, *profile.shape)

        exponent_shape = (count,) + (1,) * field.ndim
        exponents = torch.arange(
            1, count + 1, dtype=field.dtype, device=field.device
        ).reshape(exponent_shape)
        previous_dpx, previous_dpz = first_derivatives8(
            previous_field,
            dx_m=self.dx_m,
            dz_m=self.dz_m,
            boundary="free_surface",
        )
        dpx, dpz = first_derivatives8(
            field, dx_m=self.dx_m, dz_m=self.dz_m, boundary="free_surface"
        )
        fractions = torch.arange(
            1, count + 1, dtype=field.dtype, device=field.device
        ).reshape(exponent_shape) / float(count)
        dpx_sequence = previous_dpx.unsqueeze(0) + fractions * (
            dpx - previous_dpx
        ).unsqueeze(0)
        dpz_sequence = previous_dpz.unsqueeze(0) + fractions * (
            dpz - previous_dpz
        ).unsqueeze(0)

        def memory_sequence(
            initial: torch.Tensor,
            driving_sequence: torch.Tensor,
            *,
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            a_value = expanded(a)
            b_value = expanded(b)
            powers = b_value.unsqueeze(0).pow(exponents)
            weighted_driving = (
                a_value.unsqueeze(0) * driving_sequence / powers
            )
            sequence = powers * (
                initial.unsqueeze(0) + weighted_driving.cumsum(dim=0)
            )
            return sequence, sequence[-1]

        psi_x_sequence, psi_x = memory_sequence(
            self.psi_x,
            dpx_sequence,
            a=self.profiles.a_x,
            b=self.profiles.b_x,
        )
        psi_z_sequence, psi_z = memory_sequence(
            self.psi_z,
            dpz_sequence,
            a=self.profiles.a_z,
            b=self.profiles.b_z,
        )
        stretched_x = (
            expanded(self.profiles.inv_kappa_x).unsqueeze(0) * dpx_sequence
            + psi_x_sequence
        )
        stretched_z = (
            expanded(self.profiles.inv_kappa_z).unsqueeze(0) * dpz_sequence
            + psi_z_sequence
        )
        d_stretched_x, _ = first_derivatives8(
            stretched_x,
            dx_m=self.dx_m,
            dz_m=self.dz_m,
            boundary="free_surface",
        )
        _, d_stretched_z = first_derivatives8(
            stretched_z,
            dx_m=self.dx_m,
            dz_m=self.dz_m,
            boundary="free_surface",
        )

        _, phi_x = memory_sequence(
            self.phi_x,
            d_stretched_x,
            a=self.profiles.a_x,
            b=self.profiles.b_x,
        )
        _, phi_z = memory_sequence(
            self.phi_z,
            d_stretched_z,
            a=self.profiles.a_z,
            b=self.profiles.b_z,
        )
        final_dx = d_stretched_x[-1]
        final_dz = d_stretched_z[-1]
        cpml_x = expanded(self.profiles.inv_kappa_x) * final_dx + phi_x
        cpml_z = expanded(self.profiles.inv_kappa_z) * final_dz + phi_z
        if self.physical_second_derivative_coefficients is None:
            core_x, core_z = second_derivatives8(
                field,
                dx_m=self.dx_m,
                dz_m=self.dz_m,
                boundary="free_surface",
            )
        else:
            core_x, core_z = second_derivatives_symmetric(
                field,
                dx_m=self.dx_m,
                dz_m=self.dz_m,
                coefficients=self.physical_second_derivative_coefficients,
                boundary="free_surface",
            )
        spatial_x = torch.where(
            expanded(self.profiles.active_x), cpml_x, core_x
        )
        spatial_z = torch.where(
            expanded(self.profiles.active_z), cpml_z, core_z
        )
        if update_memory:
            self.psi_x, self.psi_z = psi_x, psi_z
            self.phi_x, self.phi_z = phi_x, phi_z
        return velocity_mps.square() * (spatial_x + spatial_z)
