from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def h5_attrs(obj: h5py.Dataset | h5py.Group | h5py.File) -> dict[str, Any]:
    return {str(k): _attr_value(v) for k, v in obj.attrs.items()}


def inspect_hdf5(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    result: dict[str, Any] = {"path": str(path), "attrs": {}, "groups": {}, "datasets": {}}
    with h5py.File(path, "r") as h5:
        result["attrs"] = h5_attrs(h5)

        def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            full = "/" + name
            if isinstance(obj, h5py.Dataset):
                result["datasets"][full] = {
                    "path": full,
                    "parent": "/" + str(Path(name).parent) if str(Path(name).parent) != "." else "/",
                    "shape": list(obj.shape),
                    "rank": int(obj.ndim),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "compression_opts": obj.compression_opts,
                    "fillvalue": _attr_value(obj.fillvalue),
                    "attrs": h5_attrs(obj),
                }
            elif isinstance(obj, h5py.Group):
                result["groups"][full] = {"path": full, "attrs": h5_attrs(obj)}

        h5.visititems(visit)
    return result


def finite_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    if not np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1)
        preview = []
        for value in flat[: min(8, flat.size)].tolist():
            preview.append(_attr_value(value))
        try:
            unique_preview = [_attr_value(v) for v in np.unique(flat[: min(flat.size, 100)]).tolist()[:8]]
        except Exception:
            unique_preview = preview
        return {
            "dtype": str(arr.dtype),
            "size": int(arr.size),
            "preview": preview,
            "unique_preview": unique_preview,
        }
    finite = np.isfinite(arr)
    finite_count = int(finite.sum())
    if finite_count:
        finite_arr = arr[finite].astype(np.float64, copy=False)
        min_v = float(np.min(finite_arr))
        max_v = float(np.max(finite_arr))
        mean_v = float(np.mean(finite_arr))
        std_v = float(np.std(finite_arr))
        absmax_v = float(np.max(np.abs(finite_arr)))
        nonzero = float(np.count_nonzero(finite_arr) / finite_arr.size)
    else:
        min_v = max_v = mean_v = std_v = absmax_v = float("nan")
        nonzero = 0.0
    return {
        "finite_ratio": float(finite_count / arr.size) if arr.size else 0.0,
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "min": min_v,
        "max": max_v,
        "mean": mean_v,
        "std": std_v,
        "abs_max": absmax_v,
        "nonzero_ratio": nonzero,
    }


def dataset_limited_samples(dset: h5py.Dataset, max_elements: int = 2_000_000) -> dict[str, dict[str, Any]]:
    if dset.shape == ():
        return {"scalar": finite_stats(dset[()])}
    n = dset.shape[0]
    positions = {"first": 0, "middle": n // 2, "last": n - 1}
    out = {}
    for label, index in positions.items():
        index = max(0, min(index, n - 1))
        sample = dset[index]
        arr = np.asarray(sample)
        if arr.size > max_elements:
            stride = int(np.ceil((arr.size / max_elements) ** (1 / max(arr.ndim, 1))))
            slices = tuple(slice(None, None, stride) for _ in range(arr.ndim))
            arr = arr[slices]
        out[label] = finite_stats(arr)
    return out


def monotonic_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    diff = np.diff(arr)
    return {
        "monotonic_increasing": bool(np.all(diff > 0)) if diff.size else True,
        "min_step": float(diff.min()) if diff.size else None,
        "max_step": float(diff.max()) if diff.size else None,
        "mean_step": float(diff.mean()) if diff.size else None,
    }


def canonicalize_wavefield(array: np.ndarray, axes: list[str]) -> np.ndarray:
    axes_no_sample = [axis for axis in axes if axis != "sample"]
    aliases = {"h": "x", "w": "z", "y": "z", "t": "time"}
    axes_norm = [aliases.get(axis, axis) for axis in axes_no_sample]
    required = ["x", "z", "time"]
    if sorted(axes_norm) != sorted(required):
        raise ValueError(f"wavefield axes must contain {required}, got {axes}")
    perm = [axes_norm.index(axis) for axis in required]
    return np.transpose(array, perm)


def canonicalize_velocity(array: np.ndarray, axes: list[str]) -> np.ndarray:
    axes_no_sample = [axis for axis in axes if axis != "sample"]
    aliases = {"h": "x", "w": "z", "y": "z"}
    axes_norm = [aliases.get(axis, axis) for axis in axes_no_sample]
    required = ["x", "z"]
    if sorted(axes_norm) != sorted(required):
        raise ValueError(f"velocity axes must contain {required}, got {axes}")
    perm = [axes_norm.index(axis) for axis in required]
    return np.transpose(array, perm)


def semantic_candidates(schema: dict[str, Any]) -> dict[str, list[str]]:
    keywords = {
        "velocity": ["velocity", "vel", "vp", "model", "medium", "nu"],
        "wavefield": ["wavefield", "pressure", "solution", "field", "tensor"],
        "source": ["source", "src", "shot"],
        "time": ["time", "t-coordinate"],
        "coordinates": ["coordinate", "grid", "x", "z", "y"],
        "frequency": ["frequency", "f0", "fm"],
        "wavelet": ["wavelet", "ricker"],
        "amplitude": ["amplitude"],
    }
    out = {k: [] for k in keywords}
    for path in schema.get("datasets", {}):
        lname = path.lower()
        for semantic, words in keywords.items():
            if any(word in lname for word in words):
                out[semantic].append(path)
    return out


@dataclass(frozen=True)
class ShapeSummary:
    sample_count: int
    height: int
    width: int
    time_steps: int
