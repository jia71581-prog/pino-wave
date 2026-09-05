from __future__ import annotations

from collections.abc import Callable

import torch

from fno_acoustic.numerics.drp_coefficients import DRPCoefficients

from .cpml import CFSCPMLProfiles
from .stencils import (
    first_derivatives8,
    second_derivatives8,
    second_derivatives_symmetric,
)


LWC84Step = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]

LWC84Block = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
]


def _second_derivatives(
    field: torch.Tensor,
    *,
    dx_m: float,
    dz_m: float,
    coefficients: DRPCoefficients | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coefficients is None:
        return second_derivatives8(
            field,
            dx_m=dx_m,
            dz_m=dz_m,
            boundary="free_surface",
        )
    return second_derivatives_symmetric(
        field,
        dx_m=dx_m,
        dz_m=dz_m,
        coefficients=coefficients,
        boundary="free_surface",
    )


def functional_cfs_cpml_lwc84_step(
    p_nm1: torch.Tensor,
    p_n: torch.Tensor,
    psi_x: torch.Tensor,
    psi_z: torch.Tensor,
    phi_x: torch.Tensor,
    phi_z: torch.Tensor,
    q_n: torch.Tensor,
    qtt_n: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    profiles: CFSCPMLProfiles,
    dx_m: float,
    dz_m: float,
    dt_s: float,
    spatial_coefficients: DRPCoefficients | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """One allocation-safe LWC84/CFS-CPML step suitable for graph fusion.

    Public solvers validate shapes, positivity and finite inputs before entering
    the time loop.  This inner kernel intentionally contains no tensor-valued
    Python branch, host synchronization, or mutation of input state.  Its
    arithmetic is the same as ``CFSCPMLOperator.apply`` followed by the
    production fourth-time-order LWC correction.
    """

    dpx, dpz = first_derivatives8(
        p_n,
        dx_m=float(dx_m),
        dz_m=float(dz_m),
        boundary="free_surface",
    )
    next_psi_x = profiles.b_x * psi_x + profiles.a_x * dpx
    next_psi_z = profiles.b_z * psi_z + profiles.a_z * dpz
    stretched_x = profiles.inv_kappa_x * dpx + next_psi_x
    stretched_z = profiles.inv_kappa_z * dpz + next_psi_z
    d_stretched_x, _ = first_derivatives8(
        stretched_x,
        dx_m=float(dx_m),
        dz_m=float(dz_m),
        boundary="free_surface",
    )
    _, d_stretched_z = first_derivatives8(
        stretched_z,
        dx_m=float(dx_m),
        dz_m=float(dz_m),
        boundary="free_surface",
    )
    next_phi_x = profiles.b_x * phi_x + profiles.a_x * d_stretched_x
    next_phi_z = profiles.b_z * phi_z + profiles.a_z * d_stretched_z
    cpml_x = profiles.inv_kappa_x * d_stretched_x + next_phi_x
    cpml_z = profiles.inv_kappa_z * d_stretched_z + next_phi_z
    core_x, core_z = _second_derivatives(
        p_n,
        dx_m=float(dx_m),
        dz_m=float(dz_m),
        coefficients=spatial_coefficients,
    )
    acceleration = velocity_mps.square() * (
        torch.where(profiles.active_x, cpml_x, core_x)
        + torch.where(profiles.active_z, cpml_z, core_z)
    ) + q_n
    fourth_x, fourth_z = _second_derivatives(
        acceleration,
        dx_m=float(dx_m),
        dz_m=float(dz_m),
        coefficients=spatial_coefficients,
    )
    fourth = velocity_mps.square() * (fourth_x + fourth_z) + qtt_n
    dt = float(dt_s)
    p_np1 = (
        2.0 * p_n
        - p_nm1
        + dt**2 * acceleration
        + dt**4 * fourth / 12.0
    )
    # These are the same four explicit outer boundary assignments used by the
    # production solver.  ``p_np1`` is newly allocated, so input state remains
    # functional even though the boundary write is in-place on the result.
    p_np1[..., 0, :] = 0.0
    p_np1[..., -1, :] = 0.0
    p_np1[..., :, 0] = 0.0
    p_np1[..., :, -1] = 0.0
    return (
        p_n,
        p_np1,
        next_psi_x,
        next_psi_z,
        next_phi_x,
        next_phi_z,
    )


def make_cfs_cpml_lwc84_step(
    *,
    profiles: CFSCPMLProfiles,
    dx_m: float,
    dz_m: float,
    dt_s: float,
    spatial_coefficients: DRPCoefficients | None = None,
    compile_graph: bool = False,
    compile_mode: str = "reduce-overhead",
) -> LWC84Step:
    """Bind constants and optionally compile the exact inner time step."""

    def step(
        p_nm1: torch.Tensor,
        p_n: torch.Tensor,
        psi_x: torch.Tensor,
        psi_z: torch.Tensor,
        phi_x: torch.Tensor,
        phi_z: torch.Tensor,
        q_n: torch.Tensor,
        qtt_n: torch.Tensor,
        velocity_mps: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return functional_cfs_cpml_lwc84_step(
            p_nm1,
            p_n,
            psi_x,
            psi_z,
            phi_x,
            phi_z,
            q_n,
            qtt_n,
            velocity_mps,
            profiles=profiles,
            dx_m=dx_m,
            dz_m=dz_m,
            dt_s=dt_s,
            spatial_coefficients=spatial_coefficients,
        )

    if not compile_graph:
        return step
    if not hasattr(torch, "compile"):
        raise RuntimeError("compiled LWC84 execution requires torch.compile")
    return torch.compile(
        step,
        fullgraph=True,
        dynamic=False,
        mode=str(compile_mode),
    )


def make_cfs_cpml_lwc84_block(
    *,
    profiles: CFSCPMLProfiles,
    dx_m: float,
    dz_m: float,
    dt_s: float,
    step_count: int,
    spatial_coefficients: DRPCoefficients | None = None,
    compile_graph: bool = False,
    compile_mode: str = "reduce-overhead",
) -> LWC84Block:
    """Bind a fixed number of exact inner steps into one deployment call.

    ``q_sequence`` and ``qtt_sequence`` have leading length ``step_count``.
    The static loop is deliberately captured as one graph so a saved-frame
    interval incurs one dispatcher transition while the four CFS-CPML memory
    recurrences still advance at every internal time step.
    """

    count = int(step_count)
    if count <= 0:
        raise ValueError("LWC84 block step_count must be positive")

    def block(
        p_nm1: torch.Tensor,
        p_n: torch.Tensor,
        psi_x: torch.Tensor,
        psi_z: torch.Tensor,
        phi_x: torch.Tensor,
        phi_z: torch.Tensor,
        q_sequence: torch.Tensor,
        qtt_sequence: torch.Tensor,
        velocity_mps: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = (p_nm1, p_n, psi_x, psi_z, phi_x, phi_z)
        for index in range(count):
            state = functional_cfs_cpml_lwc84_step(
                *state,
                q_sequence[index],
                qtt_sequence[index],
                velocity_mps,
                profiles=profiles,
                dx_m=dx_m,
                dz_m=dz_m,
                dt_s=dt_s,
                spatial_coefficients=spatial_coefficients,
            )
        return state

    if not compile_graph:
        return block
    if not hasattr(torch, "compile"):
        raise RuntimeError("compiled LWC84 execution requires torch.compile")
    return torch.compile(
        block,
        fullgraph=True,
        dynamic=False,
        mode=str(compile_mode),
    )


__all__ = [
    "LWC84Step",
    "LWC84Block",
    "functional_cfs_cpml_lwc84_step",
    "make_cfs_cpml_lwc84_block",
    "make_cfs_cpml_lwc84_step",
]
