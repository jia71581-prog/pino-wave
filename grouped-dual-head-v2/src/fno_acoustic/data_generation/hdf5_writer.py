from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .grid import AcousticGrid, OutputTimeGrid
from .quality import sha256_array
from .source import FORMULA
from .split_manifest import CATEGORY_IDS, FREQUENCY_ROLE_IDS, SPLIT_IDS, TRUTH_KIND_IDS


def _root_attrs(grid: AcousticGrid, time: OutputTimeGrid, root_attrs: dict[str, Any] | None = None) -> dict[str, Any]:
    attrs = {
        "schema_version": "acoustic_2km_400x400_v3_gpu",
        "layout": "N,Z,X,T",
        "units": "SI",
        "pde_convention": "p_tt=c^2*(p_xx+p_zz)+c^2*f(t,x,z)",
        "source_formula": FORMULA,
        "source_spatial_delta": "delta(x-x0)*delta(z-z0)",
        "required_anchor_frequencies_hz": json.dumps([10.0, 25.0]),
        "frequency_role_fractions": json.dumps([0.20, 0.20, 0.60]),
        "boundary_top": "free_surface_dirichlet",
        "boundary_left": "cpml",
        "boundary_right": "cpml",
        "boundary_bottom": "cpml",
        "nx": int(grid.nx),
        "nz": int(grid.nz),
        "dx_m": float(grid.dx_m),
        "dz_m": float(grid.dz_m),
        "nt_out": int(time.nt_out),
        "dt_out_s": float(time.dt_out_s),
        "category_ids": json.dumps(CATEGORY_IDS, sort_keys=True),
        "split_ids": json.dumps(SPLIT_IDS, sort_keys=True),
        "frequency_role_ids": json.dumps(FREQUENCY_ROLE_IDS, sort_keys=True),
        "truth_kind_ids": json.dumps(TRUTH_KIND_IDS, sort_keys=True),
    }
    attrs.update(root_attrs or {})
    return attrs


class AtomicShardWriter:
    def __init__(
        self,
        path: str | Path,
        *,
        sample_count: int,
        grid: AcousticGrid,
        time: OutputTimeGrid,
        root_attrs: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.sample_count = int(sample_count)
        self.grid = grid
        self.time = time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.h5 = h5py.File(self.partial_path, "w")
        for key, value in _root_attrs(grid, time, root_attrs).items():
            self.h5.attrs[key] = value
        self._create_datasets()

    def _create_datasets(self) -> None:
        h5 = self.h5
        n = self.sample_count
        z, x, t = self.grid.nz, self.grid.nx, self.time.nt_out
        h5.create_group("coordinates")
        h5["coordinates"].create_dataset("x_m", data=self.grid.x_m.astype(np.float64))
        h5["coordinates"].create_dataset("z_m", data=self.grid.z_m.astype(np.float64))
        h5["coordinates"].create_dataset("t_s", data=self.time.t_s.astype(np.float64))
        samples = h5.create_group("samples")
        samples.create_dataset("velocity_mps", shape=(n, z, x), dtype="float32")
        samples.create_dataset("wavefield", shape=(n, z, x, t), dtype="float32")
        samples.create_dataset("source_map", shape=(n, z, x), dtype="float32")
        samples.create_dataset("source_xy_m", shape=(n, 2), dtype="float64")
        samples.create_dataset("source_indices", shape=(n, 4, 2), dtype="int32")
        samples.create_dataset("source_weights", shape=(n, 4), dtype="float64")
        samples.create_dataset("source_wavelet_out", shape=(n, t), dtype="float32")
        samples.create_dataset("f0_hz", shape=(n,), dtype="float32")
        samples.create_dataset("frequency_role", shape=(n,), dtype="uint8")
        samples.create_dataset("frequency_pair_id", shape=(n,), dtype="int64")
        samples.create_dataset("category_id", shape=(n,), dtype="uint8")
        samples.create_dataset("split_id", shape=(n,), dtype="uint8")
        samples.create_dataset("base_model_id", shape=(n,), dtype="int64")
        samples.create_dataset("sample_seed", shape=(n,), dtype="uint64")
        quality = h5.create_group("quality")
        for name, dtype in {
            "c_min_mps": "float32",
            "c_max_mps": "float32",
            "cfl_axis": "float64",
            "ppw_60db": "float32",
            "truth_internal_dx_m": "float32",
            "truth_internal_dt_s": "float64",
            "truth_internal_n_substeps": "int32",
            "ppw_exception_validated": "bool",
            "max_abs_wavefield": "float32",
            "truth_kind": "uint8",
            "gpu_solver_elapsed_s": "float32",
            "gpu_pack_d2h_elapsed_s": "float32",
            "hdf5_write_elapsed_s": "float32",
            "gpu_batch_size": "uint16",
            "passed": "bool",
        }.items():
            quality.create_dataset(name, shape=(n,), dtype=dtype)
        h5.create_dataset("completed_mask", shape=(n,), dtype="bool")
        h5.create_dataset("sample_sha256", shape=(n,), dtype=h5py.string_dtype("ascii", 64))

    def write_sample(
        self,
        index: int,
        *,
        row: dict[str, Any],
        velocity_mps: np.ndarray,
        wavefield: np.ndarray,
        source,
        source_wavelet_out: np.ndarray,
        quality: dict[str, Any],
    ) -> None:
        i = int(index)
        velocity = np.asarray(velocity_mps, dtype=np.float32)
        wf = np.asarray(wavefield, dtype=np.float32)
        self.h5["samples/velocity_mps"][i] = velocity
        self.h5["samples/wavefield"][i] = wf
        self.h5["samples/source_map"][i] = np.asarray(source.source_map, dtype=np.float32)
        self.h5["samples/source_xy_m"][i] = np.asarray(row.get("source_xy_m", [source.x0_m, source.z0_m]), dtype=np.float64)
        self.h5["samples/source_indices"][i] = np.asarray(source.indices, dtype=np.int32)
        self.h5["samples/source_weights"][i] = np.asarray(source.weights, dtype=np.float64)
        self.h5["samples/source_wavelet_out"][i] = np.asarray(source_wavelet_out, dtype=np.float32)
        self.h5["samples/f0_hz"][i] = np.float32(row["f0_hz"])
        self.h5["samples/frequency_role"][i] = np.uint8(row.get("frequency_role_id", FREQUENCY_ROLE_IDS[row["frequency_role"]]))
        self.h5["samples/frequency_pair_id"][i] = np.int64(row["frequency_pair_id"])
        self.h5["samples/category_id"][i] = np.uint8(row.get("category_id", CATEGORY_IDS[row["category"]]))
        self.h5["samples/split_id"][i] = np.uint8(row.get("split_id", SPLIT_IDS[row["split"]]))
        self.h5["samples/base_model_id"][i] = np.int64(row["base_model_id"])
        self.h5["samples/sample_seed"][i] = np.uint64(row.get("sample_seed", 0))
        q = self.h5["quality"]
        cmin = float(quality.get("c_min_mps", np.min(velocity)))
        cmax = float(quality.get("c_max_mps", np.max(velocity)))
        q["c_min_mps"][i] = cmin
        q["c_max_mps"][i] = cmax
        q["cfl_axis"][i] = float(quality.get("cfl_axis", 0.0))
        q["ppw_60db"][i] = float(quality.get("ppw_60db", 8.0))
        q["truth_internal_dx_m"][i] = float(quality.get("truth_internal_dx_m", self.grid.dx_m))
        q["truth_internal_dt_s"][i] = float(quality.get("truth_internal_dt_s", self.time.dt_out_s))
        q["truth_internal_n_substeps"][i] = int(quality.get("truth_internal_n_substeps", 1))
        q["ppw_exception_validated"][i] = bool(quality.get("ppw_exception_validated", False))
        q["max_abs_wavefield"][i] = float(quality.get("max_abs_wavefield", np.max(np.abs(wf))))
        q["truth_kind"][i] = np.uint8(row.get("truth_kind_id", TRUTH_KIND_IDS.get(row.get("truth_kind", "gpu_lwc84"), 1)))
        q["gpu_solver_elapsed_s"][i] = float(quality.get("gpu_solver_elapsed_s", 0.0))
        q["gpu_pack_d2h_elapsed_s"][i] = float(quality.get("gpu_pack_d2h_elapsed_s", 0.0))
        q["hdf5_write_elapsed_s"][i] = float(quality.get("hdf5_write_elapsed_s", 0.0))
        q["gpu_batch_size"][i] = int(quality.get("gpu_batch_size", 1))
        q["passed"][i] = bool(quality.get("passed", True))
        self.h5["completed_mask"][i] = True
        self.h5["sample_sha256"][i] = sha256_array(wf)

    def close_and_commit(self) -> Path:
        self.h5.flush()
        self.h5.close()
        os.replace(self.partial_path, self.path)
        return self.path


class IncrementalHDF5Writer(AtomicShardWriter):
    def __init__(
        self,
        path: str | Path,
        *,
        sample_count: int,
        grid: AcousticGrid,
        time: OutputTimeGrid,
        root_attrs: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.partial_path = self.path.with_suffix(self.path.suffix + ".partial")
        self.sample_count = int(sample_count)
        self.grid = grid
        self.time = time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not resume:
                raise FileExistsError(f"{self.path} already exists; pass resume=True to continue writing it")
            self.h5 = h5py.File(self.path, "r+")
            self._validate_existing_file()
        else:
            self.h5 = h5py.File(self.path, "w")
            for key, value in _root_attrs(grid, time, root_attrs).items():
                self.h5.attrs[key] = value
            self._create_datasets()
            self.h5.flush()

    def _validate_existing_file(self) -> None:
        validate_v3_hdf5_schema(self.h5, expected_n=self.sample_count, strict=False)
        expected_shape = (self.sample_count, self.grid.nz, self.grid.nx, self.time.nt_out)
        actual_shape = tuple(int(v) for v in self.h5["/samples/wavefield"].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"existing HDF5 wavefield shape {actual_shape} does not match expected {expected_shape}")
        if "/completed_mask" not in self.h5:
            raise ValueError("existing HDF5 is missing /completed_mask and cannot be resumed")

    def is_complete(self, index: int) -> bool:
        return bool(self.h5["/completed_mask"][int(index)])

    def completed_count(self) -> int:
        return int(np.asarray(self.h5["/completed_mask"], dtype=bool).sum())

    def write_sample(self, *args, **kwargs) -> None:
        super().write_sample(*args, **kwargs)
        self.h5.flush()

    def close(self) -> Path:
        self.h5.flush()
        self.h5.close()
        return self.path

    def close_and_commit(self) -> Path:
        return self.close()


def validate_v3_hdf5_schema(h5: h5py.File, *, expected_n: int | None = None, strict: bool = True) -> dict[str, Any]:
    required = [
        "/coordinates/x_m",
        "/coordinates/z_m",
        "/coordinates/t_s",
        "/samples/velocity_mps",
        "/samples/wavefield",
        "/samples/source_map",
        "/samples/source_xy_m",
        "/samples/source_indices",
        "/samples/source_weights",
        "/samples/source_wavelet_out",
        "/samples/f0_hz",
        "/quality/cfl_axis",
        "/quality/passed",
    ]
    missing = [path for path in required if path not in h5]
    if missing:
        raise ValueError(f"missing required v3 HDF5 paths: {missing}")
    wavefield = h5["/samples/wavefield"]
    n = int(wavefield.shape[0])
    if expected_n is not None and n != int(expected_n):
        raise ValueError(f"expected {expected_n} samples, got {n}")
    if wavefield.dtype != np.float32:
        raise ValueError("wavefield dtype must be float32")
    if strict:
        if h5.attrs.get("production_device") != "cuda":
            raise ValueError("strict v3 validation requires production_device=cuda")
        if not bool(np.asarray(h5["/quality/passed"]).all()):
            raise ValueError("not all samples passed quality gates")
    return {"sample_count": n, "wavefield_shape": list(wavefield.shape), "schema_version": h5.attrs.get("schema_version")}
