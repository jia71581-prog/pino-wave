from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .grid import AcousticGrid
from .model_marmousi import _load_velocity, _sha256_file


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def generate_uniform_velocity(
    grid: AcousticGrid,
    *,
    seed: int,
    velocity_range_mps: tuple[float, float] = (1800.0, 5800.0),
    excluded_window_mps: tuple[float, float] = (3950.0, 4050.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    low, high = (float(value) for value in velocity_range_mps)
    excluded_low, excluded_high = (float(value) for value in excluded_window_mps)
    for _ in range(10_000):
        value = float(rng.uniform(low, high))
        if not excluded_low <= value <= excluded_high:
            break
    else:
        raise ValueError("uniform velocity range is completely covered by its exclusion window")
    velocity = np.full((grid.nz, grid.nx), value, dtype=np.float32)
    return velocity, {"model_type": "uniform", "seed": int(seed), "velocity_mps": value}


def _layer_velocities(rng: np.random.Generator, count: int) -> np.ndarray:
    for _ in range(10_000):
        # The repaired production dataset uses the registered top-fast contract:
        # every deeper layer is slower than the layer above it.  Keep this
        # reconstruction rule identical to the generator used for those shards;
        # the frozen manifest stores the seed rather than every realised layer.
        values = np.sort(rng.uniform(1600.0, 6000.0, size=count))[::-1]
        if np.all(np.abs(np.diff(values)) >= 200.0):
            return values
    raise RuntimeError("failed to draw adjacent layer contrasts >=200 m/s")


def generate_layered_velocity(
    grid: AcousticGrid,
    *,
    seed: int,
    minimum_thickness_m: float = 150.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    n_layers = int(rng.integers(2, 6))
    minimum = float(minimum_thickness_m)
    extra = float(grid.lz_m) - n_layers * minimum
    if extra < 0.0:
        raise ValueError("grid is too shallow for requested layer count and minimum thickness")
    fractions = rng.dirichlet(np.ones(n_layers))
    thickness = minimum + extra * fractions
    interfaces_base = np.cumsum(thickness)[:-1]
    layer_values = _layer_velocities(rng, n_layers)
    style = "horizontal" if int(seed) % 2 == 0 else ("tilted" if int(seed) % 4 == 1 else "gently_wavy")
    x = grid.x_m
    shift = np.zeros_like(x)
    amplitude = 0.0
    if style != "horizontal":
        edge_clearance = min(float(thickness[0] - minimum), float(thickness[-1] - minimum))
        amplitude = max(0.0, min(50.0, 0.45 * edge_clearance))
        if style == "tilted":
            shift = amplitude * (2.0 * x / float(grid.lx_m) - 1.0)
        else:
            shift = amplitude * np.sin(2.0 * math.pi * x / float(grid.lx_m) + rng.uniform(0.0, 2.0 * math.pi))
    interfaces = interfaces_base[:, None] + shift[None, :]
    z = grid.z_m[:, None, None]
    layer_index = np.sum(z >= interfaces[None, :, :], axis=1)
    velocity = layer_values[layer_index].astype(np.float32)
    all_interfaces = np.concatenate(
        [np.zeros((1, grid.nx)), interfaces, np.full((1, grid.nx), grid.lz_m)], axis=0
    )
    realized_minimum = float(np.min(np.diff(all_interfaces, axis=0)))
    if realized_minimum + 1.0e-9 < minimum:
        raise RuntimeError("generated layer violates minimum thickness")
    return velocity, {
        "model_type": "layered",
        "seed": int(seed),
        "n_layers": n_layers,
        "layer_velocity_mps": [float(value) for value in layer_values],
        "base_interface_z_m": [float(value) for value in interfaces_base],
        "interface_style": style,
        "interface_shift_amplitude_m": amplitude,
        "minimum_realized_thickness_m": realized_minimum,
    }


def generate_anomaly_velocity(
    grid: AcousticGrid,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    background = float(rng.uniform(2000.0, 5000.0))
    count = int(rng.integers(1, 4))
    zz, xx = np.meshgrid(grid.z_m, grid.x_m, indexing="ij")
    velocity = np.full((grid.nz, grid.nx), background, dtype=np.float64)
    anomalies: list[dict[str, float | bool]] = []
    minimum_extent = min(float(grid.lx_m), float(grid.lz_m))
    axis_low = min(100.0, 0.10 * minimum_extent)
    axis_high = min(450.0, 0.22 * minimum_extent)
    for index in range(count):
        axis_x = float(rng.uniform(axis_low, axis_high))
        axis_z = float(rng.uniform(axis_low, axis_high))
        center_x = float(rng.uniform(axis_x, grid.lx_m - axis_x))
        center_z = float(rng.uniform(axis_z, grid.lz_m - axis_z))
        magnitude = float(rng.uniform(0.05, 0.35))
        contrast = magnitude if (index + int(seed)) % 2 == 0 else -magnitude
        smoothing_m = float(rng.uniform(0.0, 40.0))
        radius = np.sqrt(((xx - center_x) / axis_x) ** 2 + ((zz - center_z) / axis_z) ** 2)
        if smoothing_m >= min(grid.dx_m, grid.dz_m):
            normalized_width = smoothing_m / min(axis_x, axis_z)
            mask = 1.0 / (1.0 + np.exp(np.clip((radius - 1.0) / normalized_width, -60.0, 60.0)))
            smoothed = True
        else:
            mask = (radius <= 1.0).astype(np.float64)
            smoothed = False
        velocity *= 1.0 + contrast * mask
        anomalies.append(
            {
                "center_x_m": center_x,
                "center_z_m": center_z,
                "semi_axis_x_m": axis_x,
                "semi_axis_z_m": axis_z,
                "relative_contrast": contrast,
                "smoothing_m": smoothing_m,
                "smoothed": smoothed,
            }
        )
    if not np.isfinite(velocity).all() or float(velocity.min()) <= 0.0:
        raise RuntimeError("generated anomaly velocity is non-positive or non-finite")
    return velocity.astype(np.float32), {
        "model_type": "anomaly",
        "seed": int(seed),
        "background_velocity_mps": background,
        "anomaly_count": count,
        "anomalies": anomalies,
    }


def interpolate_marmousi_normalized_extent(
    source_velocity_mps: np.ndarray,
    *,
    target_shape: tuple[int, int],
    target_dx_m: float = 5.0,
    target_dz_m: float = 5.0,
    method: str = "linear",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stretch a 2D velocity image over a requested target extent by interpolation.

    This is explicitly a normalized-coordinate derived model, not a reconstruction of
    unobserved physical Marmousi geology.
    """

    source = np.asarray(source_velocity_mps, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 2:
        raise ValueError("source velocity must be a 2D array with at least 2 nodes per axis")
    target_nz, target_nx = (int(value) for value in target_shape)
    if target_nz < 2 or target_nx < 2:
        raise ValueError("target_shape must contain at least 2 nodes per axis")
    if not np.isfinite(source).all() or float(source.min()) <= 0.0:
        raise ValueError("source velocity must be positive and finite")
    if not np.isfinite([target_dx_m, target_dz_m]).all() or min(target_dx_m, target_dz_m) <= 0.0:
        raise ValueError("target spacing must be positive and finite")
    source_z = np.linspace(0.0, 1.0, source.shape[0], dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, source.shape[1], dtype=np.float64)
    target_z = np.linspace(0.0, 1.0, target_nz, dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_nx, dtype=np.float64)
    interpolator = RegularGridInterpolator(
        (source_z, source_x), source, method=method, bounds_error=True
    )
    zz, xx = np.meshgrid(target_z, target_x, indexing="ij")
    derived = interpolator(np.stack([zz.reshape(-1), xx.reshape(-1)], axis=-1))
    derived = derived.reshape(target_nz, target_nx).astype(np.float32)
    return derived, {
        "coordinate_transform": "normalized_extent_stretch",
        "interpolation": method,
        "source_shape": [int(value) for value in source.shape],
        "target_shape": [target_nz, target_nx],
        "target_dx_m": float(target_dx_m),
        "target_dz_m": float(target_dz_m),
        "target_physical_extent_m": [
            float((target_nx - 1) * target_dx_m),
            float((target_nz - 1) * target_dz_m),
        ],
        "source_velocity_range_mps": [float(source.min()), float(source.max())],
        "derived_velocity_range_mps": [float(derived.min()), float(derived.max())],
        "derived_array_sha256": _array_sha256(derived),
        "scientific_warning": (
            "Normalized-coordinate interpolation stretches existing pixels and does not add "
            "independent geological information."
        ),
    }


def load_marmousi_crop(
    path: str | Path,
    *,
    grid: AcousticGrid,
    source_dx_m: float,
    source_dz_m: float,
    source_unit: str,
    crop_x0_m: float,
    crop_z0_m: float,
    interpolation: str = "linear",
) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(path)
    source, format_evidence = _load_velocity(path)
    source = np.squeeze(np.asarray(source))
    if source.ndim != 2:
        raise ValueError(f"Marmousi velocity must be 2D, got {source.shape}")
    unit = str(source_unit).strip().lower()
    if unit in {"km/s", "km s-1", "km_per_s"}:
        source = source.astype(np.float64) * 1000.0
    elif unit in {"m/s", "m s-1", "m_per_s"}:
        source = source.astype(np.float64)
    else:
        raise ValueError(f"unsupported Marmousi velocity unit: {source_unit}")
    if not np.isfinite(source).all() or float(source.min()) <= 0.0:
        raise ValueError("Marmousi velocity must be positive and finite")
    source_x = np.arange(source.shape[1], dtype=np.float64) * float(source_dx_m)
    source_z = np.arange(source.shape[0], dtype=np.float64) * float(source_dz_m)
    target_x = float(crop_x0_m) + grid.x_m
    target_z = float(crop_z0_m) + grid.z_m
    if target_x[0] < source_x[0] or target_z[0] < source_z[0] or target_x[-1] > source_x[-1] or target_z[-1] > source_z[-1]:
        raise ValueError(
            f"requested Marmousi crop [{target_x[0]},{target_x[-1]}]x"
            f"[{target_z[0]},{target_z[-1]}] m is outside source extent "
            f"[0,{source_x[-1]}]x[0,{source_z[-1]}] m; tiling and mirroring are forbidden"
        )
    method = "linear" if interpolation in {"linear", "scipy_regular_grid_linear"} else interpolation
    interpolator = RegularGridInterpolator(
        (source_z, source_x), source, method=method, bounds_error=True
    )
    target_zz, target_xx = np.meshgrid(target_z, target_x, indexing="ij")
    points = np.stack([target_zz.reshape(-1), target_xx.reshape(-1)], axis=-1)
    crop = interpolator(points).reshape(grid.nz, grid.nx).astype(np.float32)
    return crop, {
        "source_path": str(path.resolve()),
        "source_sha256": _sha256_file(path),
        "source_format_evidence": format_evidence,
        "source_shape": [int(value) for value in source.shape],
        "source_dx_m": float(source_dx_m),
        "source_dz_m": float(source_dz_m),
        "source_unit_original": source_unit,
        "velocity_unit": "m/s",
        "interpolation": method,
        "crop_x0_m": float(crop_x0_m),
        "crop_z0_m": float(crop_z0_m),
        "crop_bbox_m": [
            float(crop_x0_m),
            float(crop_x0_m + grid.lx_m),
            float(crop_z0_m),
            float(crop_z0_m + grid.lz_m),
        ],
        "source_velocity_range_mps": [float(source.min()), float(source.max())],
        "crop_velocity_range_mps": [float(crop.min()), float(crop.max())],
        "crop_sha256": _array_sha256(crop),
        "clipped": False,
    }
