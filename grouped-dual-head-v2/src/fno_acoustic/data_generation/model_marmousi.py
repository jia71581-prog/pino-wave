from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io
import torch


@dataclass(frozen=True)
class MarmousiInventoryRow:
    path: str
    sha256: str
    size_bytes: int
    shape: list[int]
    dtype: str
    velocity_min_mps: float
    velocity_max_mps: float
    velocity_mean_mps: float
    dx_m: float | None
    dz_m: float | None
    shape_source: str
    extractable_2km_crop_count: int
    byte_order: str = "not_applicable_serialized_tensor"
    physical_width_m: float | None = None
    physical_depth_m: float | None = None
    velocity_unit: str = "m/s"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_array_from_hdf5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        candidates: list[np.ndarray] = []

        def visit(_name, obj) -> None:
            if isinstance(obj, h5py.Dataset) and np.issubdtype(obj.dtype, np.number) and obj.ndim >= 2:
                candidates.append(np.asarray(obj))

        h5.visititems(visit)
    if not candidates:
        raise ValueError(f"no numeric 2D+ dataset found in {path}")
    return np.asarray(candidates[0])


def _load_velocity(path: Path) -> tuple[np.ndarray, str]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path)), "npy"
    if suffix == ".npz":
        data = np.load(path)
        key = next(k for k in data.files if np.asarray(data[k]).ndim >= 2)
        return np.asarray(data[key]), f"npz:{key}"
    if suffix == ".mat":
        data = scipy.io.loadmat(path)
        key = next(k for k, v in data.items() if not k.startswith("__") and np.asarray(v).ndim >= 2)
        return np.asarray(data[key]), f"mat:{key}"
    if suffix in {".h5", ".hdf5"}:
        return _numeric_array_from_hdf5(path), "hdf5"
    if suffix == ".bin":
        obj = torch.load(path, map_location="cpu", weights_only=True)
        arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
        if arr.ndim < 2:
            raise ValueError(f"raw binary {path} is not a torch-saved array")
        return arr, "torch_load_repository_evidence"
    raise ValueError(f"unsupported Marmousi file format: {path}")


def inventory_marmousi(root: Path) -> list[MarmousiInventoryRow]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    rows: list[MarmousiInventoryRow] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".npy", ".npz", ".mat", ".h5", ".hdf5", ".bin"}:
            continue
        arr, source = _load_velocity(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Marmousi candidate {path} is not 2D after squeeze: {arr.shape}")
        if float(arr.min()) < 1200.0 or float(arr.max()) > 6500.0:
            raise ValueError(f"Marmousi velocity range outside hard limits for {path}: {arr.min()}..{arr.max()}")
        dx_m = dz_m = 5.0 if source == "torch_load_repository_evidence" else None
        crop_nx = int(np.floor(2000.0 / dx_m)) if dx_m else 0
        crop_nz = int(np.floor(2000.0 / dz_m)) if dz_m else 0
        crop_count = max(0, arr.shape[1] - crop_nx + 1) * max(0, arr.shape[0] - crop_nz + 1) if dx_m else 0
        rows.append(
            MarmousiInventoryRow(
                path=str(path),
                sha256=_sha256_file(path),
                size_bytes=int(path.stat().st_size),
                shape=[int(v) for v in arr.shape],
                dtype=str(arr.dtype),
                velocity_min_mps=float(arr.min()),
                velocity_max_mps=float(arr.max()),
                velocity_mean_mps=float(arr.mean()),
                dx_m=dx_m,
                dz_m=dz_m,
                shape_source=source,
                extractable_2km_crop_count=int(crop_count),
                physical_width_m=float((arr.shape[1] - 1) * dx_m) if dx_m else None,
                physical_depth_m=float((arr.shape[0] - 1) * dz_m) if dz_m else None,
            )
        )
    return rows


def select_full_marmousi_candidate(
    rows: list[MarmousiInventoryRow],
    config: dict[str, Any],
) -> MarmousiInventoryRow:
    configured_path = str(Path(config["velocity_file"]).resolve())
    matches = [row for row in rows if str(Path(row.path).resolve()) == configured_path]
    if len(matches) != 1:
        raise ValueError(f"configured Marmousi velocity file was not uniquely inventoried: {configured_path}")
    row = matches[0]
    if row.sha256 != str(config.get("sha256", "")):
        raise ValueError(f"Marmousi SHA-256 mismatch for {configured_path}")
    if row.dx_m is None or row.dz_m is None:
        provenance_path = Path(str(config.get("provenance_file", ""))).expanduser()
        expected_provenance_sha256 = str(config.get("provenance_sha256", ""))
        if not provenance_path.is_file() or not expected_provenance_sha256:
            raise ValueError(
                "Marmousi spacing is unavailable and no hash-bound provenance was configured"
            )
        if _sha256_file(provenance_path) != expected_provenance_sha256:
            raise ValueError("Marmousi provenance SHA-256 mismatch")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if str(provenance.get("prepared_output_sha256", "")) != row.sha256:
            raise ValueError("Marmousi provenance is not bound to the configured velocity file")
        if [int(value) for value in provenance.get("prepared_shape_zx", [])] != row.shape:
            raise ValueError("Marmousi provenance shape does not match the configured velocity")
        transform = str(provenance.get("transform", ""))
        expected_transform = str(config.get("coordinate_transform", ""))
        if transform != expected_transform:
            raise ValueError("Marmousi provenance coordinate transform mismatch")
        dx_m = float(provenance["dx_m"])
        dz_m = float(provenance["dz_m"])
        if not np.isclose(dx_m, float(config["source_dx_m"])) or not np.isclose(
            dz_m, float(config["source_dz_m"])
        ):
            raise ValueError("Marmousi provenance spacing does not match the configuration")
        crop_nx = int(np.floor(2000.0 / dx_m)) + 1
        crop_nz = int(np.floor(2000.0 / dz_m)) + 1
        row = replace(
            row,
            dx_m=dx_m,
            dz_m=dz_m,
            byte_order="not_applicable_numpy_npy",
            physical_width_m=float((row.shape[1] - 1) * dx_m),
            physical_depth_m=float((row.shape[0] - 1) * dz_m),
            extractable_2km_crop_count=(
                max(0, row.shape[1] - crop_nx + 1)
                * max(0, row.shape[0] - crop_nz + 1)
            ),
        )
    required_width_m = float(config.get("required_width_m", 2000.0))
    required_depth_m = float(config.get("required_depth_m", 2000.0))
    if (
        row.physical_width_m is None
        or row.physical_depth_m is None
        or row.physical_width_m < required_width_m
        or row.physical_depth_m < required_depth_m
    ):
        raise ValueError(
            f"configured Marmousi file {configured_path} spans "
            f"{row.physical_width_m}x{row.physical_depth_m} m and cannot provide a "
            f"{required_width_m}x{required_depth_m} m crop"
        )
    return row
