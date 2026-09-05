"""CPU builders for family-adaptive straight-ray/Eikonal travel caches."""
from __future__ import annotations

import numpy as np


def straight_ray_grid_numpy(
    velocity_mps: np.ndarray,
    *,
    source_x_m: np.ndarray,
    source_z_m: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    samples: int = 12,
) -> np.ndarray:
    """Match the operator's bilinear straight-ray feature on a complete grid."""

    velocity = np.asarray(velocity_mps, dtype=np.float32)
    x = np.asarray(x_m, dtype=np.float32)
    z = np.asarray(z_m, dtype=np.float32)
    source_x = np.asarray(source_x_m, dtype=np.float32)
    source_z = np.asarray(source_z_m, dtype=np.float32)
    if velocity.ndim != 2 or velocity.shape != (len(z), len(x)):
        raise ValueError("velocity and x/z grid shapes do not match")
    if x.ndim != 1 or z.ndim != 1 or len(x) < 2 or len(z) < 2:
        raise ValueError("straight-ray x/z coordinates must be nontrivial vectors")
    if source_x.ndim != 1 or source_z.shape != source_x.shape or not len(source_x):
        raise ValueError("straight-ray sources must be matching nonempty vectors")
    if int(samples) < 2:
        raise ValueError("straight-ray integration requires at least two samples")
    if (
        not np.isfinite(velocity).all()
        or np.any(velocity <= 0.0)
        or not np.isfinite(source_x).all()
        or not np.isfinite(source_z).all()
    ):
        raise ValueError("straight-ray inputs must be finite and velocity positive")
    if not (np.all(np.diff(x) > 0.0) and np.all(np.diff(z) > 0.0)):
        raise ValueError("straight-ray coordinate vectors must be increasing")

    zz, xx = np.meshgrid(z, x, indexing="ij")
    query_x = xx.reshape(-1)
    query_z = zz.reshape(-1)
    if bool(np.all(velocity == velocity.flat[0])):
        distance = np.sqrt(
            np.square(query_x[None] - source_x[:, None], dtype=np.float32)
            + np.square(query_z[None] - source_z[:, None], dtype=np.float32),
            dtype=np.float32,
        )
        return (distance / velocity.flat[0]).reshape(
            len(source_x), len(z), len(x)
        ).astype(np.float32, copy=False)
    alpha = np.linspace(0.0, 1.0, int(samples), dtype=np.float32)
    result = np.empty((len(source_x), len(z), len(x)), dtype=np.float32)
    for source_index, (sx, sz) in enumerate(zip(source_x, source_z, strict=True)):
        ray_x = sx + (query_x[:, None] - sx) * alpha[None]
        ray_z = sz + (query_z[:, None] - sz) * alpha[None]
        grid_x = (ray_x - x[0]) * np.float32((len(x) - 1) / float(x[-1] - x[0]))
        grid_z = (ray_z - z[0]) * np.float32((len(z) - 1) / float(z[-1] - z[0]))
        grid_x = np.clip(grid_x, 0.0, len(x) - 1)
        grid_z = np.clip(grid_z, 0.0, len(z) - 1)
        x0 = np.floor(grid_x).astype(np.int64)
        z0 = np.floor(grid_z).astype(np.int64)
        x1 = np.minimum(x0 + 1, len(x) - 1)
        z1 = np.minimum(z0 + 1, len(z) - 1)
        wx = (grid_x - x0).astype(np.float32, copy=False)
        wz = (grid_z - z0).astype(np.float32, copy=False)
        sampled = (
            velocity[z0, x0] * (1.0 - wx) * (1.0 - wz)
            + velocity[z0, x1] * wx * (1.0 - wz)
            + velocity[z1, x0] * (1.0 - wx) * wz
            + velocity[z1, x1] * wx * wz
        )
        mean_slowness = np.mean(np.reciprocal(sampled), axis=1, dtype=np.float32)
        distance = np.sqrt(
            np.square(query_x - sx, dtype=np.float32)
            + np.square(query_z - sz, dtype=np.float32),
            dtype=np.float32,
        )
        result[source_index] = (distance * mean_slowness).reshape(len(z), len(x))
    return result


__all__ = ["straight_ray_grid_numpy"]
