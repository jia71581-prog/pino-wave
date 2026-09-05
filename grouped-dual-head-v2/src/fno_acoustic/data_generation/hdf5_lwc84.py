from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "acoustic_lwc84_401_to_201_v1"
SAMPLE_FLOATS = (
    "source_x_m",
    "source_z_m",
    "source_f0_hz",
    "source_t0_s",
    "source_amplitude",
    "vmin_mps",
    "vmax_mps",
    "cfl_2d",
    "lwc_qmax",
    "dt_used_s",
    "crop_x0_m",
    "crop_z0_m",
    "qc_max_abs",
    "qc_final_energy_ratio",
)
SAMPLE_STRINGS = ("medium_type", "sample_id", "group_id", "qc_status")
VDS_DATASETS = (
    "velocity_mps",
    "wavefield",
    "source_map",
    "source_wavelet",
    *SAMPLE_FLOATS,
    *SAMPLE_STRINGS,
    "seed",
    "completed_mask",
    "sample_sha256",
)
SPLIT_IDS = {"train": 0, "validation": 1, "test_id": 2, "ood_canonical": 3}


def _file_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sample_sha256(velocity: np.ndarray, wavefield: np.ndarray, source_map: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (velocity, wavefield, source_map):
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sample_id_binding_sha256(sample_ids: Sequence[str]) -> str:
    payload = json.dumps([str(value) for value in sample_ids], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _required_attrs(attrs: dict[str, Any], *, split: str, x_m: np.ndarray, z_m: np.ndarray, time_s: np.ndarray) -> dict[str, Any]:
    dx = float(np.diff(x_m)[0]) if x_m.size > 1 else 0.0
    dz = float(np.diff(z_m)[0]) if z_m.size > 1 else 0.0
    dt_out = float(np.diff(time_s)[0]) if time_s.size > 1 else 0.0
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "axis_order": "NTZX",
        "split": split,
        "velocity_unit": "m/s",
        "space_unit": "m",
        "time_unit": "s",
        "dx_m": dx,
        "dz_m": dz,
        "saved_dx_m": dx,
        "saved_dz_m": dz,
        "solver_dx_m": dx / 2.0,
        "solver_dz_m": dz / 2.0,
        "restriction": "separable_binomial5_lowpass_then_nodal_decimate2",
        "dt_output_s": dt_out,
        "t_end_s": float(time_s[-1]),
        "lwc_version": "LWC-84: fourth-order time, eighth-order centred space",
        "second_derivative_coefficients": json.dumps(
            [-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0, -205.0 / 72.0,
             8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0]
        ),
        "free_surface": "node-centred p(z=0)=0 with eighth-order odd ghost extension",
        "cpml": "unsplit three-sided CFS-CPML/ADE; left,right,bottom; no top CPML",
        "lwc_update_formula": "p_np1=2p_n-p_nm1+dt^2*(L(p_n)+q_n)+dt^4/12*(L(L(p_n)+q_n)+q_tt_n)",
        "source_formula": "Ricker w=(1-2*pi^2*f0^2*(t-t0)^2)*exp(-pi^2*f0^2*(t-t0)^2)",
    }
    result.update(attrs)
    return result


class LWC84ShardWriter:
    """Single-worker resumable writer with atomic final-shard publication."""

    def __init__(
        self,
        path: str | Path,
        *,
        sample_count: int,
        split: str,
        time_s: np.ndarray,
        x_m: np.ndarray,
        z_m: np.ndarray,
        attrs: dict[str, Any],
        resume: bool = False,
        expected_sample_ids: Sequence[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.sha256_path = self.path.with_suffix(self.path.suffix + ".sha256")
        self.sample_count = int(sample_count)
        self.split = str(split)
        self.time_s = np.asarray(time_s, dtype=np.float64)
        self.x_m = np.asarray(x_m, dtype=np.float64)
        self.z_m = np.asarray(z_m, dtype=np.float64)
        self.expected_sample_ids = (
            tuple(str(value) for value in expected_sample_ids)
            if expected_sample_ids is not None
            else None
        )
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.expected_sample_ids is not None:
            if len(self.expected_sample_ids) != self.sample_count:
                raise ValueError("expected_sample_ids length must match sample_count")
            if len(set(self.expected_sample_ids)) != len(self.expected_sample_ids):
                raise ValueError("expected_sample_ids must be unique within a shard")
        for name, coordinate in (("time_s", self.time_s), ("x_m", self.x_m), ("z_m", self.z_m)):
            if coordinate.ndim != 1 or coordinate.size < 1 or not np.isfinite(coordinate).all():
                raise ValueError(f"{name} must be a finite one-dimensional coordinate")
            if coordinate.size > 1 and np.any(np.diff(coordinate) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")
        self.root_attrs = _required_attrs(
            attrs, split=self.split, x_m=self.x_m, z_m=self.z_m, time_s=self.time_s
        )
        if self.expected_sample_ids is not None:
            self.root_attrs["expected_sample_ids_sha256"] = _sample_id_binding_sha256(
                self.expected_sample_ids
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"finalized shard already exists: {self.path}")
        if self.partial_path.exists():
            if not resume:
                raise FileExistsError(
                    f"partial shard exists: {self.partial_path}; pass resume=True to continue"
                )
            self.h5 = h5py.File(self.partial_path, "r+")
            self._validate_resume_binding()
        else:
            self.h5 = h5py.File(self.partial_path, "w")
            for key, value in self.root_attrs.items():
                self.h5.attrs[key] = value
            self._create_datasets()
            self.h5.flush()

    def _create_datasets(self) -> None:
        n, nt, nz, nx = self.sample_count, self.time_s.size, self.z_m.size, self.x_m.size
        self.h5.create_dataset("time_s", data=self.time_s)
        self.h5.create_dataset("x_m", data=self.x_m)
        self.h5.create_dataset("z_m", data=self.z_m)
        self.h5.create_dataset("velocity_mps", (n, nz, nx), dtype="f4", chunks=(1, nz, nx), compression="lzf")
        self.h5.create_dataset("wavefield", (n, nt, nz, nx), dtype="f4", chunks=(1, 1, nz, nx), compression="lzf")
        self.h5.create_dataset("source_map", (n, nz, nx), dtype="f4", chunks=(1, nz, nx), compression="lzf")
        self.h5.create_dataset("source_wavelet", (n, nt), dtype="f4", chunks=(1, nt), compression="lzf")
        for name in SAMPLE_FLOATS:
            self.h5.create_dataset(name, (n,), dtype="f8")
        string_dtype = h5py.string_dtype("utf-8", 128)
        for name in SAMPLE_STRINGS:
            self.h5.create_dataset(name, (n,), dtype=string_dtype)
        self.h5.create_dataset("seed", (n,), dtype="u8")
        self.h5.create_dataset("completed_mask", (n,), dtype="bool")
        self.h5.create_dataset("sample_sha256", (n,), dtype=h5py.string_dtype("ascii", 64))

    def _validate_resume_binding(self) -> None:
        validate_lwc84_shard(self.partial_path, strict=False, expected_n=self.sample_count)
        if _decode(self.h5.attrs.get("split", "")) != self.split:
            raise ValueError("partial shard split does not match requested split")
        for key in ("config_sha256", "manifest_sha256"):
            if _decode(self.h5.attrs.get(key, "")) != _decode(self.root_attrs.get(key, "")):
                raise ValueError(f"partial shard {key} does not match current run")
        for name, expected in (("time_s", self.time_s), ("x_m", self.x_m), ("z_m", self.z_m)):
            if not np.array_equal(np.asarray(self.h5[name]), expected):
                raise ValueError(f"partial shard {name} does not match current run")
        if self.expected_sample_ids is not None:
            expected_binding = _sample_id_binding_sha256(self.expected_sample_ids)
            stored_binding = _decode(
                self.h5.attrs.get("expected_sample_ids_sha256", "")
            )
            if stored_binding and stored_binding != expected_binding:
                raise ValueError("partial shard expected sample-ID binding differs")
            completed = np.asarray(self.h5["completed_mask"][:], dtype=bool)
            actual_ids = [_decode(value) for value in self.h5["sample_id"][:]]
            mismatched = [
                (index, actual_ids[index], self.expected_sample_ids[index])
                for index in np.flatnonzero(completed)
                if actual_ids[index] != self.expected_sample_ids[index]
            ]
            if mismatched:
                raise ValueError(
                    "partial shard completed sample binding differs: "
                    f"{mismatched[:3]}"
                )
            self.h5.attrs["expected_sample_ids_sha256"] = expected_binding
            self.h5.flush()

    def is_complete(self, index: int) -> bool:
        return bool(self.h5["completed_mask"][int(index)])

    def write_sample(self, index: int, **sample: Any) -> None:
        i = int(index)
        if not 0 <= i < self.sample_count:
            raise IndexError(i)
        if self.is_complete(i):
            raise ValueError(f"sample {i} is already complete")
        if (
            self.expected_sample_ids is not None
            and str(sample["sample_id"]) != self.expected_sample_ids[i]
        ):
            raise ValueError(
                f"sample_id at index {i} differs from the bound production row"
            )
        velocity = np.asarray(sample["velocity_mps"], dtype=np.float32)
        wavefield = np.asarray(sample["wavefield"], dtype=np.float32)
        source_map = np.asarray(sample["source_map"], dtype=np.float32)
        source_wavelet = np.asarray(sample["source_wavelet"], dtype=np.float32)
        expected_static = (self.z_m.size, self.x_m.size)
        expected_wavefield = (self.time_s.size, *expected_static)
        if velocity.shape != expected_static or source_map.shape != expected_static:
            raise ValueError("velocity_mps/source_map saved-grid shape mismatch")
        if wavefield.shape != expected_wavefield or source_wavelet.shape != (self.time_s.size,):
            raise ValueError("wavefield/source_wavelet time or saved-grid shape mismatch")
        if not all(np.isfinite(value).all() for value in (velocity, wavefield, source_map, source_wavelet)):
            raise ValueError("sample arrays must be finite")
        if not np.isclose(float(source_map.sum(dtype=np.float64)), 1.0, rtol=0.0, atol=2.0e-6):
            raise ValueError("source_map weights must sum to one")
        self.h5["velocity_mps"][i] = velocity
        self.h5["wavefield"][i] = wavefield
        self.h5["source_map"][i] = source_map
        self.h5["source_wavelet"][i] = source_wavelet
        for name in SAMPLE_FLOATS:
            self.h5[name][i] = float(sample[name])
        for name in SAMPLE_STRINGS:
            self.h5[name][i] = str(sample[name])
        self.h5["seed"][i] = np.uint64(sample["seed"])
        self.h5["sample_sha256"][i] = _sample_sha256(velocity, wavefield, source_map)
        self.h5["completed_mask"][i] = True
        self.h5.flush()

    def close_partial(self) -> Path:
        self.h5.flush()
        self.h5.close()
        return self.partial_path

    def commit(self) -> Path:
        self.h5.flush()
        self.h5.close()
        validate_lwc84_shard(self.partial_path, strict=True, require_sidecar=False)
        os.replace(self.partial_path, self.path)
        digest = _file_sha256(self.path)
        sidecar_tmp = self.sha256_path.with_suffix(self.sha256_path.suffix + ".tmp")
        sidecar_tmp.write_text(digest + "\n", encoding="ascii")
        os.replace(sidecar_tmp, self.sha256_path)
        return self.path


def validate_lwc84_shard(
    path: str | Path,
    *,
    strict: bool = True,
    expected_n: int | None = None,
    require_sidecar: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    if strict and require_sidecar:
        sidecar = path.with_suffix(path.suffix + ".sha256")
        if not sidecar.is_file():
            raise ValueError(f"missing shard SHA-256 sidecar: {sidecar}")
        expected = sidecar.read_text(encoding="ascii").strip().split()[0]
        actual = _file_sha256(path)
        if expected != actual:
            raise ValueError(f"shard SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    required = ("time_s", "x_m", "z_m", *VDS_DATASETS)
    with h5py.File(path, "r") as h5:
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"missing LWC84 HDF5 datasets: {missing}")
        if _decode(h5.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError("unexpected LWC84 HDF5 schema_version")
        if _decode(h5.attrs.get("axis_order", "")) != "NTZX":
            raise ValueError("axis_order must be NTZX")
        required_attrs = (
            "velocity_unit", "space_unit", "time_unit", "solver_dx_m", "solver_dz_m",
            "saved_dx_m", "saved_dz_m", "dt_requested_s", "dt_used_s", "t_end_s",
            "snapshot_stride", "lwc_version", "second_derivative_coefficients", "free_surface",
            "cpml", "config_sha256", "manifest_sha256", "marmousi_sha256", "git_commit",
            "software_environment",
        )
        missing_attrs = [name for name in required_attrs if name not in h5.attrs]
        if missing_attrs:
            raise ValueError(f"missing required LWC84 root attributes: {missing_attrs}")
        n, nt, nz, nx = h5["wavefield"].shape
        if expected_n is not None and n != int(expected_n):
            raise ValueError(f"expected {expected_n} samples, got {n}")
        if (
            h5["time_s"].shape != (nt,)
            or h5["z_m"].shape != (nz,)
            or h5["x_m"].shape != (nx,)
        ):
            raise ValueError("coordinate shapes do not match the wavefield NTZX shape")
        time_s = np.asarray(h5["time_s"], dtype=np.float64)
        if time_s.size > 1 and np.any(np.diff(time_s) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        dt_used = float(h5.attrs["dt_used_s"])
        snapshot_stride = int(h5.attrs["snapshot_stride"])
        dt_output = float(h5.attrs.get("dt_output_s", np.nan))
        if not np.isclose(
            dt_used * snapshot_stride, dt_output, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError("dt_used_s and snapshot_stride do not align to dt_output_s")
        if time_s.size > 1 and not np.allclose(
            np.diff(time_s), dt_output, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("time_s spacing differs from dt_output_s")
        complete = np.asarray(h5["completed_mask"], dtype=bool)
        sample_dt = np.asarray(h5["dt_used_s"], dtype=np.float64)
        if not np.allclose(sample_dt[complete], dt_used, rtol=0.0, atol=1.0e-15):
            raise ValueError("per-sample dt_used_s differs from the root protocol")
        if h5["velocity_mps"].shape != (n, nz, nx) or h5["source_map"].shape != (n, nz, nx):
            raise ValueError("static field shape does not match wavefield NTZX shape")
        if h5["source_wavelet"].shape != (n, nt):
            raise ValueError("source_wavelet shape does not match wavefield time axis")
        if tuple(h5["wavefield"].chunks or ()) != (1, 1, nz, nx):
            raise ValueError("wavefield chunk must be (1,1,Z,X)")
        if h5["wavefield"].compression != "lzf":
            raise ValueError("wavefield compression must be lzf")
        if h5["wavefield"].dtype != np.float32 or h5["velocity_mps"].dtype != np.float32:
            raise ValueError("wavefield and velocity_mps must be float32")
        if strict and not bool(complete.all()):
            raise ValueError(f"shard is incomplete: {int(complete.sum())}/{n} samples")
        if strict:
            for index in range(n):
                velocity = np.asarray(h5["velocity_mps"][index])
                wavefield = np.asarray(h5["wavefield"][index])
                source_map = np.asarray(h5["source_map"][index])
                if not all(np.isfinite(value).all() for value in (velocity, wavefield, source_map)):
                    raise ValueError(f"sample {index} contains NaN or Inf")
                if not np.isclose(source_map.sum(dtype=np.float64), 1.0, atol=2.0e-6, rtol=0.0):
                    raise ValueError(f"sample {index} source_map does not sum to one")
                expected_hash = _decode(h5["sample_sha256"][index])
                if _sample_sha256(velocity, wavefield, source_map) != expected_hash:
                    raise ValueError(f"sample {index} checksum mismatch")
                if _decode(h5["qc_status"][index]) != "passed":
                    raise ValueError(f"sample {index} did not pass QC")
        return {
            "path": str(path.resolve()),
            "sample_count": int(n),
            "wavefield_shape": [int(v) for v in h5["wavefield"].shape],
            "split": _decode(h5.attrs["split"]),
            "schema_version": _decode(h5.attrs["schema_version"]),
        }


def validate_lwc84_dataset_vds(
    path: str | Path,
    *,
    strict: bool = True,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Validate a dataset-level VDS and every physical shard it references."""
    path = Path(path)
    required = ("time_s", "x_m", "z_m", "split", "split_id", *VDS_DATASETS)
    with h5py.File(path, "r") as h5:
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"missing LWC84 dataset VDS datasets: {missing}")
        if _decode(h5.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError("unexpected LWC84 HDF5 schema_version")
        if _decode(h5.attrs.get("axis_order", "")) != "NTZX":
            raise ValueError("axis_order must be NTZX")
        if _decode(h5.attrs.get("split", "")) != "all":
            raise ValueError("dataset VDS root split must be all")

        n, nt, nz, nx = h5["wavefield"].shape
        if expected_n is not None and n != int(expected_n):
            raise ValueError(f"expected {expected_n} samples, got {n}")
        if int(h5.attrs.get("vds_sample_count", -1)) != n:
            raise ValueError("dataset VDS sample count attribute does not match its shape")
        if h5["velocity_mps"].shape != (n, nz, nx) or h5["source_map"].shape != (n, nz, nx):
            raise ValueError("dataset VDS static field shape does not match wavefield NTZX shape")
        if h5["source_wavelet"].shape != (n, nt):
            raise ValueError("dataset VDS source_wavelet shape does not match wavefield time axis")
        if h5["time_s"].shape != (nt,) or h5["z_m"].shape != (nz,) or h5["x_m"].shape != (nx,):
            raise ValueError("dataset VDS coordinate shape does not match wavefield NTZX shape")
        for name in (*SAMPLE_FLOATS, *SAMPLE_STRINGS, "seed", "completed_mask", "sample_sha256"):
            if h5[name].shape != (n,):
                raise ValueError(f"dataset VDS per-sample dataset {name} has the wrong shape")
        if h5["split"].shape != (n,) or h5["split_id"].shape != (n,):
            raise ValueError("dataset VDS split metadata has the wrong shape")
        if h5["wavefield"].dtype != np.float32 or h5["velocity_mps"].dtype != np.float32:
            raise ValueError("dataset VDS wavefield and velocity_mps must be float32")
        nonvirtual = [name for name in VDS_DATASETS if not h5[name].is_virtual]
        if nonvirtual:
            raise ValueError(f"dataset VDS contains non-virtual sample datasets: {nonvirtual}")

        try:
            source_values = json.loads(_decode(h5.attrs["vds_source_shards"]))
            included_splits = json.loads(_decode(h5.attrs["included_splits"]))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("dataset VDS source or split metadata is invalid") from error
        if not isinstance(source_values, list) or not source_values:
            raise ValueError("dataset VDS must reference at least one physical shard")
        source_paths = [Path(str(value)).resolve() for value in source_values]
        if len(set(source_paths)) != len(source_paths):
            raise ValueError("dataset VDS source shard list contains duplicates")
        if not isinstance(included_splits, list) or not included_splits:
            raise ValueError("dataset VDS included_splits metadata is invalid")
        unknown_splits = sorted(set(str(value) for value in included_splits) - set(SPLIT_IDS))
        if unknown_splits:
            raise ValueError(f"dataset VDS includes unsupported splits: {unknown_splits}")

        for name in VDS_DATASETS:
            mapped_paths = [Path(_decode(value.file_name)).resolve() for value in h5[name].virtual_sources()]
            if mapped_paths != source_paths:
                raise ValueError(f"dataset VDS mapping for {name} differs from vds_source_shards")

        vds_ids = [_decode(value) for value in h5["sample_id"][:]]
        vds_splits = [_decode(value) for value in h5["split"][:]]
        vds_split_ids = np.asarray(h5["split_id"][:], dtype=np.uint8)
        completed = np.asarray(h5["completed_mask"][:], dtype=bool)
        coordinates = {
            name: np.asarray(h5[name]) for name in ("time_s", "x_m", "z_m")
        }
        provenance = {
            name: _decode(h5.attrs.get(name, ""))
            for name in ("config_sha256", "manifest_sha256", "marmousi_sha256")
        }

    expected_ids: list[str] = []
    expected_splits: list[str] = []
    source_splits: list[str] = []
    for source_path in source_paths:
        if not source_path.is_file():
            raise ValueError(f"dataset VDS source shard does not exist: {source_path}")
        summary = validate_lwc84_shard(source_path, strict=strict, require_sidecar=strict)
        source_split = str(summary["split"])
        source_splits.append(source_split)
        with h5py.File(source_path, "r") as source:
            for name, expected in provenance.items():
                if _decode(source.attrs.get(name, "")) != expected:
                    raise ValueError(f"dataset VDS source {name} differs in {source_path}")
            for name, expected in coordinates.items():
                if not np.array_equal(np.asarray(source[name]), expected):
                    raise ValueError(f"dataset VDS source coordinate {name} differs in {source_path}")
            source_ids = [_decode(value) for value in source["sample_id"][:]]
        expected_ids.extend(source_ids)
        expected_splits.extend([source_split] * len(source_ids))

    if len(expected_ids) != n:
        raise ValueError("dataset VDS source sample count does not match its shape")
    if vds_ids != expected_ids:
        raise ValueError("dataset VDS sample ordering differs from its source shards")
    if vds_splits != expected_splits:
        raise ValueError("dataset VDS split ordering differs from its source shards")
    expected_split_ids = np.asarray([SPLIT_IDS[value] for value in expected_splits], dtype=np.uint8)
    if not np.array_equal(vds_split_ids, expected_split_ids):
        raise ValueError("dataset VDS split_id values differ from its source shards")
    expected_included = sorted(set(source_splits), key=SPLIT_IDS.__getitem__)
    if [str(value) for value in included_splits] != expected_included:
        raise ValueError("dataset VDS included_splits differs from its source shards")
    if strict:
        if not bool(completed.all()):
            raise ValueError(f"dataset VDS is incomplete: {int(completed.sum())}/{n} samples")
        if any(not value for value in vds_ids):
            raise ValueError("dataset VDS contains an empty sample_id")
        if len(set(vds_ids)) != len(vds_ids):
            raise ValueError("dataset VDS sample_id values must be globally unique")

    return {
        "path": str(path.resolve()),
        "sample_count": int(n),
        "wavefield_shape": [int(n), int(nt), int(nz), int(nx)],
        "split": "all",
        "included_splits": expected_included,
        "source_shard_count": len(source_paths),
        "schema_version": SCHEMA_VERSION,
    }


def validate_lwc84_hybrid_vds(
    path: str | Path,
    *,
    strict: bool = True,
    expected_n: int | None = None,
) -> dict[str, Any]:
    """Validate a VDS that replaces train/Marmousi rows in an older dataset VDS."""
    path = Path(path)
    required = ("time_s", "x_m", "z_m", "split", "split_id", *VDS_DATASETS)
    with h5py.File(path, "r") as h5:
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"missing LWC84 hybrid VDS datasets: {missing}")
        if _decode(h5.attrs.get("schema_version", "")) != SCHEMA_VERSION:
            raise ValueError("unexpected LWC84 HDF5 schema_version")
        if _decode(h5.attrs.get("axis_order", "")) != "NTZX":
            raise ValueError("axis_order must be NTZX")
        if _decode(h5.attrs.get("split", "")) != "all":
            raise ValueError("hybrid VDS root split must be all")
        if _decode(h5.attrs.get("marmousi_replacement_scope", "")) != "train_only":
            raise ValueError("hybrid VDS replacement scope must be train_only")

        n, nt, nz, nx = h5["wavefield"].shape
        if expected_n is not None and n != int(expected_n):
            raise ValueError(f"expected {expected_n} samples, got {n}")
        if int(h5.attrs.get("vds_sample_count", -1)) != n:
            raise ValueError("hybrid VDS sample count attribute does not match its shape")
        if h5["velocity_mps"].shape != (n, nz, nx) or h5["source_map"].shape != (n, nz, nx):
            raise ValueError("hybrid VDS static field shape does not match wavefield NTZX shape")
        if h5["source_wavelet"].shape != (n, nt):
            raise ValueError("hybrid VDS source_wavelet shape does not match wavefield time axis")
        if h5["time_s"].shape != (nt,) or h5["z_m"].shape != (nz,) or h5["x_m"].shape != (nx,):
            raise ValueError("hybrid VDS coordinate shape does not match wavefield NTZX shape")
        if h5["split"].shape != (n,) or h5["split_id"].shape != (n,):
            raise ValueError("hybrid VDS split metadata has the wrong shape")
        nonvirtual = [name for name in VDS_DATASETS if not h5[name].is_virtual]
        if nonvirtual:
            raise ValueError(f"hybrid VDS contains non-virtual sample datasets: {nonvirtual}")

        try:
            source_values = json.loads(_decode(h5.attrs["vds_source_datasets"]))
            config_components = json.loads(_decode(h5.attrs["component_config_sha256"]))
            marmousi_components = json.loads(_decode(h5.attrs["component_marmousi_sha256"]))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("hybrid VDS source or component metadata is invalid") from error
        expected_source_names = {"old", "new_train_marmousi"}
        if not isinstance(source_values, dict) or set(source_values) != expected_source_names:
            raise ValueError("hybrid VDS must declare old and new_train_marmousi sources")
        source_paths = {
            name: Path(str(value)).resolve() for name, value in source_values.items()
        }
        if len(set(source_paths.values())) != len(source_paths):
            raise ValueError("hybrid VDS source datasets must be distinct")
        for source_path in source_paths.values():
            if not source_path.is_file():
                raise ValueError(f"hybrid VDS source dataset does not exist: {source_path}")

        mapping_paths: list[Path] | None = None
        for name in VDS_DATASETS:
            current = [
                Path(_decode(value.file_name)).resolve()
                for value in h5[name].virtual_sources()
            ]
            if mapping_paths is None:
                mapping_paths = current
            elif current != mapping_paths:
                raise ValueError(f"hybrid VDS mapping runs differ for {name}")
        if mapping_paths is None or set(mapping_paths) != set(source_paths.values()):
            raise ValueError("hybrid VDS mappings differ from vds_source_datasets")

        hybrid_ids = [_decode(value) for value in h5["sample_id"][:]]
        hybrid_hashes = [_decode(value) for value in h5["sample_sha256"][:]]
        hybrid_splits = [_decode(value) for value in h5["split"][:]]
        hybrid_media = [_decode(value) for value in h5["medium_type"][:]]
        hybrid_qc = [_decode(value) for value in h5["qc_status"][:]]
        hybrid_completed = np.asarray(h5["completed_mask"][:], dtype=bool)
        hybrid_config_sha256 = _decode(h5.attrs.get("config_sha256", ""))
        hybrid_marmousi_sha256 = _decode(h5.attrs.get("marmousi_sha256", ""))
        replacement_count = int(h5.attrs.get("replacement_sample_count", -1))

    old_path = source_paths["old"]
    new_path = source_paths["new_train_marmousi"]
    with h5py.File(old_path, "r") as old_h5, h5py.File(new_path, "r") as new_h5:
        expected_config_components = {
            "old_unreplaced_records": _decode(old_h5.attrs.get("config_sha256", "")),
            "new_train_marmousi_records": _decode(new_h5.attrs.get("config_sha256", "")),
        }
        expected_marmousi_components = {
            "old_unreplaced_records": _decode(old_h5.attrs.get("marmousi_sha256", "")),
            "new_train_marmousi_records": _decode(new_h5.attrs.get("marmousi_sha256", "")),
        }
        if config_components != expected_config_components:
            raise ValueError("hybrid VDS component config hashes differ from its sources")
        if marmousi_components != expected_marmousi_components:
            raise ValueError("hybrid VDS component Marmousi hashes differ from its sources")
        for components, actual, label in (
            (expected_config_components, hybrid_config_sha256, "config"),
            (expected_marmousi_components, hybrid_marmousi_sha256, "Marmousi"),
        ):
            encoded = json.dumps(components, sort_keys=True, separators=(",", ":"))
            expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if actual != expected:
                raise ValueError(f"hybrid VDS {label} aggregate hash is invalid")
        old_ids = [_decode(value) for value in old_h5["sample_id"][:]]
        old_hashes = [_decode(value) for value in old_h5["sample_sha256"][:]]
        old_completed = np.asarray(old_h5["completed_mask"][:], dtype=bool)
        new_ids = [_decode(value) for value in new_h5["sample_id"][:]]
        new_hashes = [_decode(value) for value in new_h5["sample_sha256"][:]]
        new_completed = np.asarray(new_h5["completed_mask"][:], dtype=bool)
        new_qc = [_decode(value) for value in new_h5["qc_status"][:]]
        for coordinate in ("time_s", "x_m", "z_m"):
            if not np.array_equal(np.asarray(old_h5[coordinate]), np.asarray(new_h5[coordinate])):
                raise ValueError(f"hybrid VDS source coordinate differs: {coordinate}")

    if len(old_ids) != n:
        raise ValueError("hybrid VDS old source length differs from its output length")
    if len(new_ids) != replacement_count or len(new_ids) != len(set(new_ids)):
        raise ValueError("hybrid VDS replacement source count or IDs are invalid")
    new_index = {sample_id: index for index, sample_id in enumerate(new_ids)}
    used_replacements: set[str] = set()
    for index, sample_id in enumerate(hybrid_ids):
        if sample_id in new_index:
            source_index = new_index[sample_id]
            used_replacements.add(sample_id)
            if hybrid_hashes[index] != new_hashes[source_index]:
                raise ValueError(f"hybrid VDS replacement hash differs for {sample_id}")
            if hybrid_splits[index] != "train" or hybrid_media[index] != "marmousi":
                raise ValueError(f"hybrid VDS replacement scope differs for {sample_id}")
            if strict and (not new_completed[source_index] or new_qc[source_index] != "passed"):
                raise ValueError(f"hybrid VDS replacement source is incomplete for {sample_id}")
        else:
            if old_ids[index] != sample_id or old_hashes[index] != hybrid_hashes[index]:
                raise ValueError(f"hybrid VDS unreplaced binding differs at index {index}")
            if strict and not old_completed[index]:
                raise ValueError(f"hybrid VDS unreplaced source is incomplete at index {index}")
    if used_replacements != set(new_ids):
        raise ValueError("hybrid VDS does not use every declared replacement sample")
    if strict:
        if any(not value for value in hybrid_ids):
            raise ValueError("hybrid VDS contains an empty sample_id")
        if len(set(hybrid_ids)) != len(hybrid_ids):
            raise ValueError("hybrid VDS sample_id values must be globally unique")
        if not bool(hybrid_completed.all()):
            raise ValueError(f"hybrid VDS is incomplete: {int(hybrid_completed.sum())}/{n} samples")
        if any(value != "passed" for value in hybrid_qc):
            raise ValueError("hybrid VDS contains a failed QC record")

    return {
        "path": str(path.resolve()),
        "sample_count": int(n),
        "wavefield_shape": [int(n), int(nt), int(nz), int(nx)],
        "split": "all",
        "replacement_sample_count": replacement_count,
        "virtual_mapping_run_count": len(mapping_paths),
        "source_datasets": {name: str(value) for name, value in source_paths.items()},
        "schema_version": SCHEMA_VERSION,
    }


def build_lwc84_vds(output_path: str | Path, shard_paths: Iterable[str | Path]) -> Path:
    output_path = Path(output_path)
    paths = [Path(path) for path in shard_paths]
    finalized = [path for path in paths if path.suffix != ".tmp"]
    if not finalized:
        raise ValueError("no finalized shards were provided; .tmp files are never added to a VDS")
    summaries = [validate_lwc84_shard(path, strict=True) for path in finalized]
    split = summaries[0]["split"]
    if any(summary["split"] != split for summary in summaries):
        raise ValueError("a VDS may only aggregate one split")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(finalized[0], "r") as first, h5py.File(tmp, "w", libver="latest") as output:
        for key, value in first.attrs.items():
            output.attrs[key] = value
        output.attrs["vds_source_shards"] = json.dumps([str(path.resolve()) for path in finalized])
        output.attrs["vds_sample_count"] = sum(summary["sample_count"] for summary in summaries)
        for coordinate in ("time_s", "x_m", "z_m"):
            output.create_dataset(coordinate, data=np.asarray(first[coordinate]))
        for name in VDS_DATASETS:
            source_dataset = first[name]
            total = sum(summary["sample_count"] for summary in summaries)
            layout = h5py.VirtualLayout(shape=(total, *source_dataset.shape[1:]), dtype=source_dataset.dtype)
            offset = 0
            for path, summary in zip(finalized, summaries, strict=True):
                with h5py.File(path, "r") as shard:
                    if shard[name].shape[1:] != source_dataset.shape[1:] or shard[name].dtype != source_dataset.dtype:
                        raise ValueError(f"incompatible VDS dataset {name} in {path}")
                count = summary["sample_count"]
                source = h5py.VirtualSource(str(path.resolve()), name, shape=(count, *source_dataset.shape[1:]))
                layout[offset : offset + count] = source
                offset += count
            output.create_virtual_dataset(name, layout)
    os.replace(tmp, output_path)
    return output_path


def build_lwc84_dataset_vds(
    output_path: str | Path,
    shard_paths: Iterable[str | Path],
    *,
    strict_validation: bool = True,
) -> Path:
    """Build one zero-copy dataset spanning every finalized split shard."""
    output_path = Path(output_path)
    paths = [Path(path) for path in shard_paths]
    finalized = [path for path in paths if path.suffix != ".tmp"]
    if not finalized:
        raise ValueError("no finalized shards were provided; .tmp files are never added to a VDS")
    summaries = [
        validate_lwc84_shard(path, strict=strict_validation) for path in finalized
    ]
    unknown = sorted({summary["split"] for summary in summaries} - set(SPLIT_IDS))
    if unknown:
        raise ValueError(f"unsupported split values for dataset VDS: {unknown}")
    ordered = sorted(
        zip(finalized, summaries, strict=True),
        key=lambda item: (SPLIT_IDS[item[1]["split"]], str(item[0].resolve())),
    )
    finalized = [item[0] for item in ordered]
    summaries = [item[1] for item in ordered]
    total = sum(summary["sample_count"] for summary in summaries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(finalized[0], "r") as first, h5py.File(tmp, "w", libver="latest") as output:
        for key, value in first.attrs.items():
            output.attrs[key] = value
        included_splits = sorted({summary["split"] for summary in summaries}, key=SPLIT_IDS.__getitem__)
        output.attrs["split"] = "all"
        output.attrs["included_splits"] = json.dumps(included_splits)
        output.attrs["vds_source_shards"] = json.dumps([str(path.resolve()) for path in finalized])
        output.attrs["vds_sample_count"] = total
        for coordinate in ("time_s", "x_m", "z_m"):
            output.create_dataset(coordinate, data=np.asarray(first[coordinate]))
        split_values: list[str] = []
        split_ids: list[int] = []
        for summary in summaries:
            count = int(summary["sample_count"])
            split_values.extend([summary["split"]] * count)
            split_ids.extend([SPLIT_IDS[summary["split"]]] * count)
        output.create_dataset("split", data=np.asarray(split_values, dtype=h5py.string_dtype("utf-8", 32)))
        output.create_dataset("split_id", data=np.asarray(split_ids, dtype=np.uint8))
        for name in VDS_DATASETS:
            source_dataset = first[name]
            layout = h5py.VirtualLayout(shape=(total, *source_dataset.shape[1:]), dtype=source_dataset.dtype)
            offset = 0
            for path, summary in zip(finalized, summaries, strict=True):
                with h5py.File(path, "r") as shard:
                    if shard[name].shape[1:] != source_dataset.shape[1:] or shard[name].dtype != source_dataset.dtype:
                        raise ValueError(f"incompatible VDS dataset {name} in {path}")
                    for coordinate in ("time_s", "x_m", "z_m"):
                        if not np.array_equal(np.asarray(shard[coordinate]), np.asarray(first[coordinate])):
                            raise ValueError(f"incompatible VDS coordinate {coordinate} in {path}")
                count = int(summary["sample_count"])
                source = h5py.VirtualSource(str(path.resolve()), name, shape=(count, *source_dataset.shape[1:]))
                layout[offset : offset + count] = source
                offset += count
            output.create_virtual_dataset(name, layout)
    os.replace(tmp, output_path)
    return output_path


class _Welford:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        count = int(flat.size)
        if count == 0:
            return
        mean = float(flat.mean())
        m2 = float(np.square(flat - mean).sum())
        if not np.isfinite(mean) or not np.isfinite(m2):
            raise ValueError("normalization statistics input contains NaN or Inf")
        delta = mean - self.mean
        total = self.count + count
        self.mean += delta * count / total
        self.m2 += m2 + delta * delta * self.count * count / total
        self.count = total

    def result(self) -> dict[str, float | int]:
        variance = self.m2 / (self.count - 1) if self.count > 1 else 0.0
        return {"count": self.count, "mean": self.mean, "std": float(np.sqrt(variance))}


def compute_train_only_stats(shard_paths: Iterable[str | Path]) -> dict[str, Any]:
    velocity, wavefield = _Welford(), _Welford()
    sample_count = 0
    paths = [Path(path) for path in shard_paths]
    if not paths:
        raise ValueError("at least one train shard is required")
    for path in paths:
        summary = validate_lwc84_shard(path, strict=True)
        if summary["split"] != "train":
            raise ValueError(f"train-only statistics reject split={summary['split']} in {path}")
        with h5py.File(path, "r") as h5:
            for index in range(summary["sample_count"]):
                velocity.update(h5["velocity_mps"][index])
                wavefield.update(h5["wavefield"][index])
        sample_count += summary["sample_count"]
    return {
        "schema_version": 1,
        "computed_from_split": "train",
        "train_sample_count": sample_count,
        "velocity": velocity.result(),
        "wavefield": wavefield.result(),
    }


def compute_train_only_dataset_stats(
    dataset_path: str | Path,
    *,
    time_chunk_size: int = 16,
) -> dict[str, Any]:
    """Stream exact train-only moments from physical shards behind a dataset VDS."""
    dataset_path = Path(dataset_path)
    if int(time_chunk_size) <= 0:
        raise ValueError("time_chunk_size must be positive")
    with h5py.File(dataset_path, "r") as h5:
        is_hybrid = "vds_source_datasets" in h5.attrs
    if is_hybrid:
        summary = validate_lwc84_hybrid_vds(dataset_path, strict=True)
    else:
        summary = validate_lwc84_dataset_vds(dataset_path, strict=True)

    with h5py.File(dataset_path, "r") as h5:
        dataset_ids = [_decode(value) for value in h5["sample_id"][:]]
        split_values = [_decode(value) for value in h5["split"][:]]
        train_ids = {
            sample_id
            for sample_id, split in zip(dataset_ids, split_values, strict=True)
            if split == "train"
        }
        if not train_ids:
            raise ValueError("dataset VDS contains no train records")
        _, nt, nz, nx = h5["wavefield"].shape
        bindings = {
            name: _decode(h5.attrs.get(name, ""))
            for name in ("config_sha256", "manifest_sha256", "marmousi_sha256")
        }
        if is_hybrid:
            source_datasets = json.loads(_decode(h5.attrs["vds_source_datasets"]))
        else:
            source_datasets = None

    source_entries: list[tuple[str, Path]] = []
    replacement_ids: set[str] = set()
    if source_datasets is not None:
        new_dataset = Path(str(source_datasets["new_train_marmousi"])).resolve()
        old_dataset = Path(str(source_datasets["old"])).resolve()
        with h5py.File(new_dataset, "r") as new_h5:
            replacement_ids = {_decode(value) for value in new_h5["sample_id"][:]}
            new_paths = [
                Path(str(value)).resolve()
                for value in json.loads(_decode(new_h5.attrs["vds_source_shards"]))
            ]
        with h5py.File(old_dataset, "r") as old_h5:
            old_paths = [
                Path(str(value)).resolve()
                for value in json.loads(_decode(old_h5.attrs["vds_source_shards"]))
            ]
        source_entries.extend(("new", path) for path in new_paths)
        source_entries.extend(("old", path) for path in old_paths)
    else:
        with h5py.File(dataset_path, "r") as h5:
            source_paths = [
                Path(str(value)).resolve()
                for value in json.loads(_decode(h5.attrs["vds_source_shards"]))
            ]
        source_entries.extend(("dataset", path) for path in source_paths)

    source_entries = [
        (kind, path) for kind, path in source_entries if path.parent.name == "train"
    ]
    if not source_entries:
        raise ValueError("dataset VDS contains no physical train shard references")

    def release_file_cache(path: Path) -> bool:
        if not path.is_file() or not hasattr(os, "posix_fadvise"):
            return False
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(
                descriptor,
                0,
                0,
                getattr(os, "POSIX_FADV_DONTNEED", 4),
            )
        except OSError:
            return False
        finally:
            os.close(descriptor)
        return True

    initial_evictions = sum(
        release_file_cache(path) for _, path in source_entries if path.is_file()
    )
    velocity, wavefield = _Welford(), _Welford()
    seen_ids: set[str] = set()
    processed_shards = 0
    final_evictions = 0
    for source_kind, source_path in source_entries:
        if not source_path.is_file():
            continue
        selected_in_shard = 0
        with h5py.File(source_path, "r") as source:
            if _decode(source.attrs.get("split", "")) != "train":
                continue
            source_ids = [_decode(value) for value in source["sample_id"][:]]
            for index, sample_id in enumerate(source_ids):
                if sample_id not in train_ids:
                    continue
                if sample_id in seen_ids:
                    if source_kind == "old" and sample_id in replacement_ids:
                        continue
                    raise ValueError(f"duplicate train sample across source shards: {sample_id}")
                seen_ids.add(sample_id)
                selected_in_shard += 1
                velocity.update(source["velocity_mps"][index])
                for start in range(0, int(nt), int(time_chunk_size)):
                    wavefield.update(
                        source[
                            "wavefield"
                        ][index, start : min(start + int(time_chunk_size), int(nt))]
                    )
        if selected_in_shard:
            processed_shards += 1
        final_evictions += int(release_file_cache(source_path))

    if seen_ids != train_ids:
        missing = sorted(train_ids - seen_ids)
        extra = sorted(seen_ids - train_ids)
        raise ValueError(
            "physical train shard coverage differs from the hybrid VDS: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    expected_velocity_count = len(train_ids) * int(nz) * int(nx)
    expected_wavefield_count = expected_velocity_count * int(nt)
    if velocity.count != expected_velocity_count or wavefield.count != expected_wavefield_count:
        raise ValueError("normalization statistics element count does not match train geometry")
    return {
        "schema_version": 1,
        "computed_from_split": "train",
        "train_sample_count": len(train_ids),
        "dataset_sample_count": int(summary["sample_count"]),
        "dataset": str(dataset_path.resolve()),
        **bindings,
        "physical_train_shards_processed": processed_shards,
        "initial_cache_evictions": initial_evictions,
        "final_cache_evictions": final_evictions,
        "velocity": velocity.result(),
        "wavefield": wavefield.result(),
    }
