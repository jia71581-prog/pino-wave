from __future__ import annotations

import hashlib

import numpy as np

from .grid import AcousticGrid, BoundaryConfig


def enforce_free_surface(field):
    out = np.asarray(field)
    out[..., 0, :] = 0.0
    return out


def cpml_damping_mask(grid: AcousticGrid, boundaries: BoundaryConfig) -> np.ndarray:
    nz_ext, nx_ext = boundaries.extended_shape(grid)
    mask = np.ones((nz_ext, nx_ext), dtype=np.float32)
    npml = int(boundaries.npml)
    order = int(boundaries.cpml_polynomial_order)
    target = float(boundaries.cpml_target_reflection)
    min_damp = max(0.0, min(0.999, target ** (1.0 / max(npml, 1))))
    for i in range(npml):
        depth = (npml - i) / npml
        value = 1.0 - (1.0 - min_damp) * depth**order
        mask[:, i] *= value
        mask[:, nx_ext - 1 - i] *= value
        mask[nz_ext - 1 - i, :] *= value
    mask[0, :] = 1.0
    return mask


def _cpml_axis_profile(
    n: int,
    *,
    left: bool,
    right: bool,
    npml: int,
    spacing_m: float,
    dt_s: float,
    c_ref_m_s: float,
    target_reflection: float,
    polynomial_order: int,
    kappa_max: float,
    alpha_max_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.zeros(int(n), dtype=np.float64)
    kappa = np.ones(int(n), dtype=np.float64)
    alpha = np.zeros(int(n), dtype=np.float64)
    thickness_m = max(float(npml) * float(spacing_m), float(spacing_m))
    target = min(max(float(target_reflection), 1.0e-12), 0.999999)
    order = int(polynomial_order)
    sigma_max = -float(order + 1) * float(c_ref_m_s) * np.log(target) / (2.0 * thickness_m)
    alpha_max = 2.0 * np.pi * max(float(alpha_max_hz), 0.0)

    def fill(idx: int, depth: float) -> None:
        depth = min(max(float(depth), 0.0), 1.0)
        sigma[idx] = sigma_max * depth**order
        kappa[idx] = 1.0 + (float(kappa_max) - 1.0) * depth**order
        alpha[idx] = alpha_max * (1.0 - depth)

    if left:
        for i in range(int(npml)):
            fill(i, (int(npml) - i) / float(npml))
    if right:
        start = int(n) - int(npml)
        for i in range(int(npml)):
            fill(start + i, (i + 1) / float(npml))

    decay = sigma / kappa + alpha
    b = np.exp(-decay * float(dt_s))
    a = np.zeros_like(b)
    denom = kappa * (sigma + kappa * alpha)
    active = denom > 0.0
    a[active] = sigma[active] * (b[active] - 1.0) / denom[active]
    return a.astype(np.float32), b.astype(np.float32), (1.0 / kappa).astype(np.float32)


def cpml_memory_coefficients(
    grid: AcousticGrid,
    boundaries: BoundaryConfig,
    *,
    dt_s: float,
    c_ref_m_s: float,
    kappa_max: float = 5.0,
    alpha_max_hz: float = 0.0,
) -> dict[str, np.ndarray]:
    nz_ext, nx_ext = boundaries.extended_shape(grid)
    npml = int(boundaries.npml)
    ax, bx, inv_kx = _cpml_axis_profile(
        nx_ext,
        left=True,
        right=True,
        npml=npml,
        spacing_m=float(grid.dx_m),
        dt_s=float(dt_s),
        c_ref_m_s=float(c_ref_m_s),
        target_reflection=float(boundaries.cpml_target_reflection),
        polynomial_order=int(boundaries.cpml_polynomial_order),
        kappa_max=float(kappa_max),
        alpha_max_hz=float(alpha_max_hz),
    )
    az, bz, inv_kz = _cpml_axis_profile(
        nz_ext,
        left=False,
        right=True,
        npml=npml,
        spacing_m=float(grid.dz_m),
        dt_s=float(dt_s),
        c_ref_m_s=float(c_ref_m_s),
        target_reflection=float(boundaries.cpml_target_reflection),
        polynomial_order=int(boundaries.cpml_polynomial_order),
        kappa_max=float(kappa_max),
        alpha_max_hz=float(alpha_max_hz),
    )
    return {
        "a_x": np.broadcast_to(ax[None, :], (nz_ext, nx_ext)).copy(),
        "b_x": np.broadcast_to(bx[None, :], (nz_ext, nx_ext)).copy(),
        "inv_kappa_x": np.broadcast_to(inv_kx[None, :], (nz_ext, nx_ext)).copy(),
        "a_z": np.broadcast_to(az[:, None], (nz_ext, nx_ext)).copy(),
        "b_z": np.broadcast_to(bz[:, None], (nz_ext, nx_ext)).copy(),
        "inv_kappa_z": np.broadcast_to(inv_kz[:, None], (nz_ext, nx_ext)).copy(),
    }


def restrict_fine_to_coarse_2x2(fine: np.ndarray) -> np.ndarray:
    arr = np.asarray(fine)
    if arr.shape[-2] % 2 or arr.shape[-1] % 2:
        raise ValueError("fine grid dimensions must be divisible by 2")
    reshaped = arr.reshape(*arr.shape[:-2], arr.shape[-2] // 2, 2, arr.shape[-1] // 2, 2)
    return reshaped.mean(axis=(-3, -1)).astype(arr.dtype, copy=False)


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
    digest.update(arr.tobytes())
    return digest.hexdigest()


def finite_wavefield_stats(wavefield: np.ndarray) -> dict[str, float | int | bool]:
    arr = np.asarray(wavefield)
    return {
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "max_abs_wavefield": float(np.max(np.abs(arr))) if arr.size else 0.0,
        "passed": bool(np.isfinite(arr).all()),
    }
