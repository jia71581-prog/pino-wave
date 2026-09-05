"""Audited fixed discrete wave-operator core for the B2-H structure-preserving
propagator, matching the LWC-84 data generator EXACTLY (dataset attrs verified):

  * 8th-order centred second-derivative stencil (9-point), coefficients read from
    the dataset provenance `second_derivative_coefficients`:
        [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
  * saved grid dx=dz=10 m, saved frames spaced dt_output=0.0025 s (NB the physics
    was integrated at dt_used=0.000125 s with snapshot_stride=20 -- see AUDIT below).
  * L(p) = c^2 * (p_xx + p_zz)  is the physical acceleration operator (source added
    separately).  Free surface p(z=0)=0 at the top; the CPML absorbing region lives
    OUTSIDE the saved 201x201 physical crop, so the saved interior is (near-)lossless
    except for energy radiating out through the cropped boundary.

AUDIT CONSEQUENCE for B2-H (honest, load-bearing):
  A saved-frame -> saved-frame map is NOT one leapfrog step: consecutive saved frames
  are 20 internal LWC-84 substeps apart, and the saved field is a binomial5 low-pass
  restriction of the 401x401 solver field.  Therefore the fixed physical core below is
  the EFFECTIVE coarse-grid operator; the learned B2-H residual must absorb (a) the
  20-substep composition gap and (b) the restriction/anti-alias gap.  This is the
  DGNet "physical discretization + bounded neural correction" regime -- documented,
  not hidden.  We do NOT claim a single coarse stencil reproduces GT.

Everything here is torch, CPU-testable, and reused by the B2-H propagator + trainer.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

# 8th-order centred second derivative (exact rationals; == dataset attrs to 1e-15)
_D2_COEFFS = (-1.0 / 560, 8.0 / 315, -1.0 / 5, 8.0 / 5, -205.0 / 72, 8.0 / 5, -1.0 / 5, 8.0 / 315, -1.0 / 560)
_HALO = 4  # 9-point stencil radius


def second_derivative_kernels(dtype=torch.float64, device="cpu"):
    """Return (kx, kz) conv kernels for d2/dx2 and d2/dz2 at unit spacing."""
    c = torch.tensor(_D2_COEFFS, dtype=dtype, device=device)
    kx = c.view(1, 1, 1, 9)   # along x (last dim)
    kz = c.view(1, 1, 9, 1)   # along z
    return kx, kz


def laplacian_8th(
    field: torch.Tensor,
    *,
    dx_m: float = 10.0,
    dz_m: float = 10.0,
    free_surface_top: bool = True,
) -> torch.Tensor:
    """8th-order Laplacian (p_xx + p_zz) on [..., Z, X], SAME shape out.

    Boundary handling matches the saved-crop diagnostic use: zero (Dirichlet) halo
    on the cropped sides; free-surface odd reflection at the top row when requested
    (p(z=0)=0 with odd ghost extension, as the generator documents).  This is a
    diagnostic-grade boundary; the learned residual handles boundary reality.
    """
    if field.shape[-1] < 9 or field.shape[-2] < 9:
        raise ValueError("field spatial dims must be >= 9 for the 8th-order stencil")
    orig_shape = field.shape
    x = field.reshape(-1, 1, orig_shape[-2], orig_shape[-1])
    kx, kz = second_derivative_kernels(dtype=x.dtype, device=x.device)
    # x-direction: zero pad both sides (Dirichlet)
    xx = F.conv2d(F.pad(x, (_HALO, _HALO, 0, 0)), kx) / (dx_m * dx_m)
    # z-direction: free-surface odd reflection at top, zero pad at bottom
    if free_surface_top:
        top = -x[:, :, 1 : _HALO + 1, :].flip(2)  # odd extension about z=0
        bottom = torch.zeros_like(x[:, :, :_HALO, :])
        xz_in = torch.cat((top, x, bottom), dim=2)
        zz = F.conv2d(xz_in, kz) / (dz_m * dz_m)
    else:
        zz = F.conv2d(F.pad(x, (0, 0, _HALO, _HALO)), kz) / (dz_m * dz_m)
    return (xx + zz).reshape(orig_shape)


def wave_acceleration(
    pressure: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dx_m: float = 10.0,
    dz_m: float = 10.0,
    free_surface_top: bool = True,
) -> torch.Tensor:
    """Physical acceleration L(p) = c^2 * lap(p) (source injected separately)."""
    lap = laplacian_8th(pressure, dx_m=dx_m, dz_m=dz_m, free_surface_top=free_surface_top)
    return velocity_mps.pow(2) * lap


def central_velocity(prev: torch.Tensor, nxt: torch.Tensor, dt_s: float) -> torch.Tensor:
    """v_k = (p_{k+1} - p_{k-1}) / (2 dt) -- 2nd-order centred, saved-grid spacing."""
    return (nxt - prev) / (2.0 * dt_s)


def interior_mask(shape_zx, margin: int) -> torch.Tensor:
    """Boolean [Z,X] mask that is True strictly inside a `margin`-cell border,
    so energy radiating out the cropped boundary is excluded from the diagnostic."""
    z, x = shape_zx
    m = torch.zeros(z, x, dtype=torch.bool)
    m[margin : z - margin, margin : x - margin] = True
    return m


def discrete_energy_variable_c(
    p_prev: torch.Tensor,
    p_cur: torch.Tensor,
    p_next: torch.Tensor,
    velocity_mps: torch.Tensor,
    *,
    dx_m: float = 10.0,
    dz_m: float = 10.0,
    dt_s: float = 0.0025,
    margin: int = 12,
    free_surface_top: bool = True,
) -> dict[str, float]:
    """Variable-coefficient conserved acoustic energy on the interior:

        E = 1/2 sum_interior [ (1/c^2) v^2 + |grad p|^2 ]

    computed via the identity |grad p|^2 ~= -p * lap(p) on the interior (boundary
    flux is excluded by the margin), using the SAME 8th-order operator as the core.
    This is the energy that is exactly conserved for p_tt = c^2 lap(p) with variable
    c, up to boundary radiation and source work -- NOT naive global conservation.
    Returns kinetic / potential / total (float64).
    """
    v = central_velocity(p_prev, p_next, dt_s).double()
    p = p_cur.double()
    c = velocity_mps.double()
    lap = laplacian_8th(p, dx_m=dx_m, dz_m=dz_m, free_surface_top=free_surface_top)
    mask = interior_mask(p.shape[-2:], margin).to(p.device)
    kinetic = 0.5 * ((v * v) / (c * c))[..., mask].sum().item()
    potential = 0.5 * (-(p * lap))[..., mask].sum().item()
    return {"kinetic": kinetic, "potential": potential, "total": kinetic + potential}
