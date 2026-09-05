#!/usr/bin/env python3
"""Build one resumable GPU shard of the exact 401-to-201 LWC-84 teacher cache."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data_generation.config import (
    boundaries_from_config,
    grid_from_config,
    load_config,
    time_from_config,
)
from fno_acoustic.data_generation.velocity_models_lwc84 import (
    generate_anomaly_velocity,
    generate_layered_velocity,
    load_marmousi_crop,
)
from fno_acoustic.data_generation.restriction import restrict_nodal_2x
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES
from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices


SHARD_SCHEMA = "lwc84_multifidelity_teacher_shard_401solver_v2"
EXPECTED_TRAIN_FAMILY_COUNTS = {
    "uniform": 420,
    "layered": 1120,
    "marmousi": 700,
}


@dataclass(frozen=True)
class TrainingRecord:
    source_index: int
    sample_id: str
    medium_type: str


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_production_velocity(
    config: Mapping[str, Any], grid, row: Mapping[str, Any]
) -> np.ndarray:
    """Mirror the frozen production generator without importing unrelated CLI code."""

    medium = str(row["medium_type"])
    parameters = row["medium_parameters"]
    if medium == "uniform":
        return np.full(
            (grid.nz, grid.nx), float(parameters["velocity_mps"]), dtype=np.float32
        )
    if medium == "layered":
        fixed_keys = {"upper_velocity_mps", "lower_velocity_mps", "interface_z_m"}
        if fixed_keys.issubset(parameters):
            velocity_column = np.where(
                grid.z_m[:, None] < float(parameters["interface_z_m"]),
                float(parameters["upper_velocity_mps"]),
                float(parameters["lower_velocity_mps"]),
            )
            return np.broadcast_to(velocity_column, (grid.nz, grid.nx)).astype(
                np.float32, copy=True
            )
        return generate_layered_velocity(grid, seed=int(parameters["seed"]))[0]
    if medium == "anomaly":
        return generate_anomaly_velocity(grid, seed=int(parameters["seed"]))[0]
    if medium == "marmousi":
        marmousi = config["marmousi"]
        return load_marmousi_crop(
            marmousi["velocity_file"],
            grid=grid,
            source_dx_m=float(marmousi["source_dx_m"]),
            source_dz_m=float(marmousi["source_dz_m"]),
            source_unit=str(marmousi["velocity_unit"]),
            crop_x0_m=float(row["crop_x0_m"]),
            crop_z0_m=float(row["crop_z0_m"]),
            interpolation=str(marmousi["interpolation"]),
        )[0]
    raise ValueError(f"unsupported production medium_type={medium}")


def _decode_text(values: np.ndarray) -> tuple[str, ...]:
    return tuple(
        value.decode("utf8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    )


def select_training_records(
    source_h5: str | Path,
    *,
    expected_family_counts: Mapping[str, int] = EXPECTED_TRAIN_FAMILY_COUNTS,
) -> tuple[TrainingRecord, ...]:
    """Return the complete allowed training census in source-index order."""

    with h5py.File(Path(source_h5), "r", swmr=True) as handle:
        split = _decode_text(np.asarray(handle["split"][:]))
        family = _decode_text(np.asarray(handle["medium_type"][:]))
        sample_id = _decode_text(np.asarray(handle["sample_id"][:]))
    if not (len(split) == len(family) == len(sample_id)):
        raise ValueError("source index datasets have inconsistent lengths")
    records = tuple(
        TrainingRecord(index, sample_id[index], family[index])
        for index in range(len(split))
        if split[index] == "train" and family[index] in ALLOWED_MEDIUM_TYPES
    )
    counts = {
        name: sum(record.medium_type == name for record in records)
        for name in ALLOWED_MEDIUM_TYPES
    }
    expected = {str(key): int(value) for key, value in expected_family_counts.items()}
    if counts != expected or len({record.sample_id for record in records}) != len(records):
        raise ValueError(f"eligible training census mismatch: {counts} != {expected}")
    return records


def shard_training_records(
    records: Sequence[TrainingRecord], *, shard_index: int, shard_count: int
) -> tuple[TrainingRecord, ...]:
    count = int(shard_count)
    index = int(shard_index)
    if count < 1 or index < 0 or index >= count:
        raise ValueError("shard index must lie in [0, shard_count)")
    return tuple(records[index::count])


def source_identity(
    source_h5: str | Path,
    *,
    manifest_path: str | Path | None = None,
    dataset_config: str | Path | None = None,
) -> dict[str, object]:
    """Bind a cache to the HDF5, frozen manifest, and frozen config bytes."""

    path = Path(source_h5).expanduser().resolve()
    manifest = (
        path.parent / "manifest.jsonl"
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    config_path = (
        path.parent / "frozen_config.yaml"
        if dataset_config is None
        else Path(dataset_config).expanduser().resolve()
    )
    if not path.is_file() or not manifest.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"teacher identity requires source={path}, manifest={manifest}, config={config_path}"
        )
    manifest_hash = sha256_file(manifest)
    config_file_hash = sha256_file(config_path)
    config = load_config(config_path)
    with h5py.File(path, "r", swmr=True) as handle:
        registered_manifest_hash = str(handle.attrs.get("manifest_sha256", ""))
        registered_config_hash = str(handle.attrs.get("config_sha256", ""))
    if not registered_manifest_hash or manifest_hash != registered_manifest_hash:
        raise ValueError("source HDF5 and manifest.jsonl SHA-256 do not match")
    if not registered_config_hash or str(config["config_sha256"]) != registered_config_hash:
        raise ValueError("source HDF5 and frozen_config.yaml identity do not match")
    return {
        "path": str(path),
        "byte_count": int(path.stat().st_size),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_hash,
        "dataset_config": str(config_path),
        "dataset_config_file_sha256": config_file_hash,
        "config_sha256": registered_config_hash,
    }


def load_exact_training_numerical_contract(
    dataset_config: str | Path,
) -> dict[str, object]:
    """Load the exact fine-grid solver/restriction contract used by the dataset."""

    path = Path(dataset_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = load_config(path)
    if not isinstance(payload, Mapping):
        raise ValueError("training dataset config must be a mapping")
    grid = payload.get("grid")
    storage = payload.get("storage_grid")
    boundaries = payload.get("boundaries")
    time_config = payload.get("time")
    if not all(
        isinstance(value, Mapping)
        for value in (grid, storage, boundaries, time_config)
    ):
        raise ValueError("training dataset config lacks grid or boundary mappings")
    assert isinstance(grid, Mapping)
    assert isinstance(storage, Mapping)
    assert isinstance(boundaries, Mapping)
    assert isinstance(time_config, Mapping)
    if (
        str(boundaries.get("top")) != "free_surface_dirichlet"
        or any(str(boundaries.get(side)) != "cpml" for side in ("left", "right", "bottom"))
        or not bool(boundaries.get("cpml_outside_physical_domain"))
        or str(boundaries.get("alpha_max_definition"))
        != "pi_times_minimum_frequency"
    ):
        raise ValueError("training CPML boundary layout is incompatible")
    solver_nx = int(grid["nx"])
    solver_nz = int(grid["nz"])
    saved_nx = int(storage["nx"])
    saved_nz = int(storage["nz"])
    solver_dx = float(grid["dx_m"])
    solver_dz = float(grid["dz_m"])
    saved_dx = float(storage["dx_m"])
    saved_dz = float(storage["dz_m"])
    restriction = str(storage.get("restriction", ""))
    if (
        str(grid.get("centering")) != "node"
        or str(storage.get("centering")) != "node"
        or (solver_nz, solver_nx) != (401, 401)
        or (saved_nz, saved_nx) != (201, 201)
        or min(solver_dx, solver_dz, saved_dx, saved_dz) <= 0.0
        or not math.isclose(saved_dx, 2.0 * solver_dx, abs_tol=1.0e-9)
        or not math.isclose(saved_dz, 2.0 * solver_dz, abs_tol=1.0e-9)
        or restriction != "binomial5_lowpass_then_decimate2"
    ):
        raise ValueError("teacher must reproduce the registered 401-to-201 grid restriction")
    npml = int(boundaries["npml"])
    target_reflection = float(boundaries["cpml_target_reflection"])
    polynomial_order = int(boundaries["cpml_polynomial_order"])
    kappa_max = float(boundaries["kappa_max"])
    minimum_frequency = float(boundaries["minimum_frequency_hz"])
    training_internal_dt_s = float(time_config["dt_used_s"])
    if (
        npml < 1
        or not 0.0 < target_reflection < 1.0
        or polynomial_order < 1
        or kappa_max < 1.0
        or minimum_frequency <= 0.0
        or training_internal_dt_s <= 0.0
    ):
        raise ValueError("training CPML profile parameters are invalid")
    return {
        "dataset_config": str(path),
        "dataset_config_sha256": str(payload["config_sha256"]),
        "solver_grid_shape": [solver_nz, solver_nx],
        "saved_grid_shape": [saved_nz, saved_nx],
        "solver_dx_m": solver_dx,
        "solver_dz_m": solver_dz,
        "saved_dx_m": saved_dx,
        "saved_dz_m": saved_dz,
        "npml": npml,
        "cpml_physical_thickness_m": npml * solver_dx,
        "cpml_target_reflection": target_reflection,
        "cpml_polynomial_order": polynomial_order,
        "kappa_max": kappa_max,
        "minimum_frequency_hz": minimum_frequency,
        "internal_dt_s": training_internal_dt_s,
        "output_time_count": int(time_config["nt_out"]),
        "output_dt_s": float(time_config["dt_out_s"]),
        "alpha_max_definition": "pi_times_minimum_frequency",
        "top_boundary": "free_surface_dirichlet",
        "other_boundaries": "cpml",
        "cpml_outside_physical_domain": True,
        "output_restriction_factor": 2,
        "restriction": "binomial5_lowpass_then_decimate2",
    }


def initialize_or_validate_shard(
    path: str | Path,
    *,
    records: Sequence[TrainingRecord],
    source_identity: Mapping[str, object],
    time_indices: Sequence[int],
    time_s: Sequence[float],
    spatial_shape: tuple[int, int],
    numerical_contract: Mapping[str, object],
) -> tuple[int, ...]:
    """Create a cache shard or return its still-incomplete local positions."""

    destination = Path(path)
    rows = tuple(records)
    indices = tuple(int(value) for value in time_indices)
    times = tuple(float(value) for value in time_s)
    height, width = (int(value) for value in spatial_shape)
    if (
        not rows
        or len(indices) < 2
        or len(indices) != len(times)
        or indices[0] != 0
        or any(left >= right for left, right in zip(indices, indices[1:]))
        or height < 2
        or width < 2
    ):
        raise ValueError("cache shard dimensions are invalid")
    identity_json = _canonical_json(source_identity)
    numerics_json = _canonical_json(numerical_contract)
    expected_samples = tuple(record.sample_id for record in rows)
    expected_sources = np.asarray(
        [record.source_index for record in rows], dtype=np.int64
    )
    expected_families = tuple(record.medium_type for record in rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with h5py.File(destination, "w", libver="latest") as handle:
            handle.attrs["schema"] = SHARD_SCHEMA
            handle.attrs["status"] = "partial"
            handle.attrs["source_identity_json"] = identity_json
            handle.attrs["numerical_contract_json"] = numerics_json
            text_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "sample_id", data=np.asarray(expected_samples, dtype=object), dtype=text_dtype
            )
            handle.create_dataset("source_index", data=expected_sources)
            handle.create_dataset(
                "medium_type",
                data=np.asarray(expected_families, dtype=object),
                dtype=text_dtype,
            )
            handle.create_dataset("time_indices", data=np.asarray(indices, dtype=np.int64))
            handle.create_dataset("time_s", data=np.asarray(times, dtype=np.float64))
            handle.create_dataset(
                "wavefield",
                shape=(len(rows), len(indices), height, width),
                dtype=np.float32,
                chunks=(1, 1, height, width),
                fillvalue=np.nan,
            )
            handle.create_dataset(
                "completed", data=np.zeros(len(rows), dtype=np.bool_)
            )
            handle.create_dataset(
                "lwc_qmax", data=np.full(len(rows), np.nan, dtype=np.float64)
            )
            handle.flush()
    with h5py.File(destination, "r+", libver="latest") as handle:
        if (
            str(handle.attrs.get("schema", "")) != SHARD_SCHEMA
            or str(handle.attrs.get("source_identity_json", "")) != identity_json
            or str(handle.attrs.get("numerical_contract_json", "")) != numerics_json
        ):
            raise ValueError("cache shard identity mismatch")
        actual_samples = tuple(handle["sample_id"].asstr()[:].tolist())
        actual_families = tuple(handle["medium_type"].asstr()[:].tolist())
        actual_sources = np.asarray(handle["source_index"][:], dtype=np.int64)
        actual_indices = tuple(int(value) for value in handle["time_indices"][:])
        actual_times = tuple(float(value) for value in handle["time_s"][:])
        expected_shape = (len(rows), len(indices), height, width)
        if (
            actual_samples != expected_samples
            or actual_families != expected_families
            or not np.array_equal(actual_sources, expected_sources)
            or actual_indices != indices
            or not np.array_equal(np.asarray(actual_times), np.asarray(times))
            or tuple(handle["wavefield"].shape) != expected_shape
            or handle["wavefield"].dtype != np.dtype(np.float32)
            or handle["completed"].shape != (len(rows),)
        ):
            raise ValueError("cache shard dataset identity mismatch")
        completed = np.asarray(handle["completed"][:], dtype=bool)
        if bool(completed.all()):
            handle.attrs["status"] = "complete"
        elif str(handle.attrs.get("status", "")) == "complete":
            raise ValueError("cache shard status contradicts completion mask")
        handle.flush()
    return tuple(int(value) for value in np.flatnonzero(~completed))


def write_completed_batch(
    path: str | Path,
    *,
    positions: Sequence[int],
    wavefield: np.ndarray,
    lwc_qmax: Sequence[float] | np.ndarray,
) -> None:
    """Persist data and QC before atomically exposing completion bits."""

    selected = tuple(int(value) for value in positions)
    values = np.asarray(wavefield, dtype=np.float32)
    qmax = np.asarray(lwc_qmax, dtype=np.float64)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or tuple(sorted(selected)) != selected
        or values.ndim != 4
        or values.shape[0] != len(selected)
        or qmax.shape != (len(selected),)
        or not np.isfinite(values).all()
        or not np.isfinite(qmax).all()
        or bool(np.any((qmax < 0.0) | (qmax >= 1.0)))
        or float(np.max(np.abs(values[:, :, 0, :]))) != 0.0
    ):
        raise ValueError("completed cache batch is invalid")
    with h5py.File(Path(path), "r+", libver="latest") as handle:
        completed = handle["completed"]
        if selected[-1] >= len(completed) or bool(np.asarray(completed[list(selected)]).any()):
            raise ValueError("cache batch position is invalid or already complete")
        if tuple(values.shape[1:]) != tuple(handle["wavefield"].shape[1:]):
            raise ValueError("cache batch wavefield shape mismatch")
        handle["wavefield"][list(selected)] = values
        handle["lwc_qmax"][list(selected)] = qmax
        handle.flush()
        completed[list(selected)] = True
        done = int(np.count_nonzero(completed[:]))
        handle.attrs["completed_count"] = done
        handle.attrs["status"] = "complete" if done == len(completed) else "partial"
        handle.flush()


def _read_axes(source_h5: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(source_h5, "r", swmr=True) as handle:
        return (
            np.asarray(handle["time_s"][:], dtype=np.float64),
            np.asarray(handle["x_m"][:], dtype=np.float64),
            np.asarray(handle["z_m"][:], dtype=np.float64),
        )


def _uniform_step(values: np.ndarray, name: str) -> float:
    differences = np.diff(values)
    if len(values) < 2 or not np.allclose(
        differences, differences[0], rtol=0.0, atol=1.0e-10
    ):
        raise ValueError(f"{name} must be a uniform axis")
    return float(differences[0])


def load_manifest_index(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load a frozen production manifest with unique sample identifiers."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256):
        raise ValueError("manifest SHA-256 differs from the source HDF5 identity")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not str(row.get("sample_id", "")):
            raise ValueError(f"invalid manifest row at line {line_number}")
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate manifest sample_id={sample_id}")
        rows[sample_id] = row
    if not rows:
        raise ValueError("production manifest is empty")
    return rows


def validate_source_dataset_contract(
    source_h5: str | Path,
    *,
    config: Mapping[str, Any],
    manifest_rows: Mapping[str, Mapping[str, Any]],
    numerical_contract: Mapping[str, object],
) -> None:
    """Reject any grid, axis, record, source, or completion drift before solving."""

    source_path = Path(source_h5).expanduser().resolve()
    grid = grid_from_config(dict(config))
    expected_time = time_from_config(dict(config)).t_s
    expected_x = grid.x_m[::2]
    expected_z = grid.z_m[::2]
    with h5py.File(source_path, "r", swmr=True) as handle:
        sample_ids = tuple(handle["sample_id"].asstr()[:].tolist())
        if (
            len(sample_ids) != len(manifest_rows)
            or len(set(sample_ids)) != len(sample_ids)
            or set(sample_ids) != set(manifest_rows)
        ):
            raise ValueError("source HDF5 sample census differs from the frozen manifest")
        split = tuple(handle["split"].asstr()[:].tolist())
        medium = tuple(handle["medium_type"].asstr()[:].tolist())
        expected_split = tuple(str(manifest_rows[sample]["split"]) for sample in sample_ids)
        expected_medium = tuple(
            str(manifest_rows[sample]["medium_type"]) for sample in sample_ids
        )
        if split != expected_split or medium != expected_medium:
            raise ValueError("source HDF5 split/medium metadata differs from the manifest")
        for name in (
            "source_x_m",
            "source_z_m",
            "source_f0_hz",
            "source_t0_s",
            "source_amplitude",
        ):
            actual = np.asarray(handle[name][:], dtype=np.float64)
            expected = np.asarray(
                [float(manifest_rows[sample][name]) for sample in sample_ids],
                dtype=np.float64,
            )
            if not np.array_equal(actual, expected):
                raise ValueError(f"source HDF5 {name} differs from the manifest")
        if not bool(np.asarray(handle["completed_mask"][:], dtype=bool).all()):
            raise ValueError("source HDF5 contains incomplete records")
        if (
            tuple(handle["velocity_mps"].shape[1:])
            != tuple(int(value) for value in numerical_contract["saved_grid_shape"])
            or tuple(handle["wavefield"].shape[1:])
            != (
                int(numerical_contract["output_time_count"]),
                *tuple(int(value) for value in numerical_contract["saved_grid_shape"]),
            )
            or json.loads(str(handle.attrs.get("solver_grid_shape", "[]")))
            != list(numerical_contract["solver_grid_shape"])
            or json.loads(str(handle.attrs.get("saved_grid_shape", "[]")))
            != list(numerical_contract["saved_grid_shape"])
            or not math.isclose(
                float(handle.attrs.get("solver_dx_m", math.nan)),
                float(numerical_contract["solver_dx_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(handle.attrs.get("solver_dz_m", math.nan)),
                float(numerical_contract["solver_dz_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(handle.attrs.get("dt_used_s", math.nan)),
                float(numerical_contract["internal_dt_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise ValueError("source HDF5 numerical grid contract differs from frozen config")
        actual_time = np.asarray(handle["time_s"][:], dtype=np.float64)
        actual_x = np.asarray(handle["x_m"][:], dtype=np.float64)
        actual_z = np.asarray(handle["z_m"][:], dtype=np.float64)
    if (
        not np.allclose(actual_time, expected_time, rtol=0.0, atol=1.0e-12)
        or not np.array_equal(actual_x, expected_x)
        or not np.array_equal(actual_z, expected_z)
    ):
        raise ValueError("source HDF5 saved axes differ from the 401-to-201 config")


def reconstruct_generation_inputs(
    source_h5: str | Path,
    records: Sequence[TrainingRecord],
    *,
    config: Mapping[str, Any],
    manifest_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Reconstruct exact 401 velocities and prove their 201 restriction matches HDF5."""

    grid = grid_from_config(dict(config))
    rows: list[Mapping[str, Any]] = []
    for record in records:
        row = manifest_rows.get(record.sample_id)
        if row is None:
            raise ValueError(f"sample {record.sample_id} is absent from the frozen manifest")
        if (
            str(row.get("medium_type")) != record.medium_type
            or str(row.get("split")) != "train"
        ):
            raise ValueError(f"manifest metadata differs for sample {record.sample_id}")
        rows.append(row)
    velocity = np.stack(
        [reconstruct_production_velocity(config, grid, row) for row in rows]
    ).astype(np.float32, copy=False)
    source = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in (
            "source_x_m",
            "source_z_m",
            "source_f0_hz",
            "source_t0_s",
            "source_amplitude",
        )
    }
    with h5py.File(source_h5, "r", swmr=True) as handle:
        expected_velocity_saved = np.stack(
            [
                np.asarray(handle["velocity_mps"][record.source_index], dtype=np.float32)
                for record in records
            ]
        )
        expected_source_map_saved = np.stack(
            [
                np.asarray(handle["source_map"][record.source_index], dtype=np.float32)
                for record in records
            ]
        )
    reconstructed_saved = np.asarray(restrict_nodal_2x(velocity), dtype=np.float32)
    if not np.array_equal(reconstructed_saved, expected_velocity_saved):
        different = np.flatnonzero(
            np.any(reconstructed_saved != expected_velocity_saved, axis=(1, 2))
        )
        first = records[int(different[0])]
        maximum = float(
            np.max(
                np.abs(
                    reconstructed_saved[int(different[0])]
                    - expected_velocity_saved[int(different[0])]
                )
            )
        )
        raise ValueError(
            "reconstructed 401 velocity does not reproduce saved 201 velocity for "
            f"{first.sample_id}; max_abs={maximum}"
        )
    return velocity, source, expected_velocity_saved, expected_source_map_saved


def run_shard(
    *,
    source_h5: str | Path,
    output_h5: str | Path,
    shard_index: int,
    shard_count: int = 4,
    solver_batch_size: int = 120,
    device: str = "cuda",
    time_count: int = 64,
    dataset_config: str | Path | None = None,
    manifest_path: str | Path | None = None,
    c_ref_mps: float = 6750.0,
    max_batches: int | None = None,
) -> dict[str, object]:
    source_path = Path(source_h5).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    if int(solver_batch_size) < 1:
        raise ValueError("solver batch must be positive")
    if max_batches is not None and int(max_batches) < 1:
        raise ValueError("max_batches must be positive when provided")
    config_path = (
        source_path.parent / "frozen_config.yaml"
        if dataset_config is None
        else Path(dataset_config).expanduser().resolve()
    )
    manifest = (
        source_path.parent / "manifest.jsonl"
        if manifest_path is None
        else Path(manifest_path).expanduser().resolve()
    )
    identity = source_identity(
        source_path,
        manifest_path=manifest,
        dataset_config=config_path,
    )
    config = load_config(config_path)
    numerical_contract = load_exact_training_numerical_contract(config_path)
    manifest_rows = load_manifest_index(
        manifest, expected_sha256=str(identity["manifest_sha256"])
    )
    validate_source_dataset_contract(
        source_path,
        config=config,
        manifest_rows=manifest_rows,
        numerical_contract=numerical_contract,
    )
    all_records = select_training_records(source_path)
    records = shard_training_records(
        all_records, shard_index=int(shard_index), shard_count=int(shard_count)
    )
    time_axis, x_m, z_m = _read_axes(source_path)
    saved_dx_m = _uniform_step(x_m, "x_m")
    saved_dz_m = _uniform_step(z_m, "z_m")
    if (
        not math.isclose(
            saved_dx_m,
            float(numerical_contract["saved_dx_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            saved_dz_m,
            float(numerical_contract["saved_dz_m"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("saved axes do not match the exact numerical contract")
    internal_dt_s = float(numerical_contract["internal_dt_s"])
    time_indices = fixed_teacher_time_indices(
        stored_time_count=len(time_axis), count=int(time_count)
    )
    output_times = time_axis[np.asarray(time_indices, dtype=np.int64)]
    ratios = output_times / float(internal_dt_s)
    if not np.allclose(ratios, np.rint(ratios), rtol=0.0, atol=1.0e-9):
        raise ValueError("teacher times do not align to the internal step")
    numerics = {
        **dict(numerical_contract),
        "c_ref_mps": float(c_ref_mps),
        "dtype": "float32",
        "teacher_semantics": "exact_training_solver_replay",
    }
    pending = initialize_or_validate_shard(
        output_h5,
        records=records,
        source_identity=identity,
        time_indices=time_indices,
        time_s=output_times,
        spatial_shape=(len(z_m), len(x_m)),
        numerical_contract=numerics,
    )
    if not pending:
        return {
            "status": "complete",
            "shard_index": int(shard_index),
            "record_count": len(records),
            "reused": True,
        }
    if requested_device.type == "cuda":
        torch.cuda.set_device(requested_device)
        torch.cuda.reset_peak_memory_stats(requested_device)
    solver = LWC84CPMLSolver(
        grid=grid_from_config(config),
        boundaries=boundaries_from_config(config),
        dt_s=float(internal_dt_s),
        output_times_s=output_times,
        c_ref_mps=float(c_ref_mps),
        device=requested_device,
        dtype=torch.float32,
        kappa_max=float(numerical_contract["kappa_max"]),
        minimum_frequency_hz=float(numerical_contract["minimum_frequency_hz"]),
        output_restriction_factor=2,
    )
    started = time.perf_counter()
    solver_seconds = 0.0
    processed_batches = 0
    for start in range(0, len(pending), int(solver_batch_size)):
        positions = pending[start : start + int(solver_batch_size)]
        batch_records = tuple(records[position] for position in positions)
        velocity, source, expected_velocity_saved, expected_source_map_saved = (
            reconstruct_generation_inputs(
                source_path,
                batch_records,
                config=config,
                manifest_rows=manifest_rows,
            )
        )
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
        solve_started = time.perf_counter()
        result = solver.simulate(
            velocity,
            source_x_m=source["source_x_m"],
            source_z_m=source["source_z_m"],
            source_f0_hz=source["source_f0_hz"],
            source_t0_s=source["source_t0_s"],
            source_amplitude=source["source_amplitude"],
        )
        if requested_device.type == "cuda":
            torch.cuda.synchronize(requested_device)
        elapsed = time.perf_counter() - solve_started
        solver_seconds += elapsed
        if not np.array_equal(result.velocity_saved_mps, expected_velocity_saved):
            raise ValueError("solver restriction no longer matches saved HDF5 velocity")
        if not np.array_equal(result.source_map_saved, expected_source_map_saved):
            raise ValueError("solver source injection no longer matches saved HDF5 source map")
        qmax = np.asarray(
            [float(row["lwc_qmax"]) for row in result.metrics], dtype=np.float64
        )
        write_completed_batch(
            output_h5,
            positions=positions,
            wavefield=result.wavefield,
            lwc_qmax=qmax,
        )
        print(
            json.dumps(
                {
                    "event": "cache_batch_complete",
                    "shard_index": int(shard_index),
                    "completed": min(start + len(positions), len(pending)),
                    "pending_at_start": len(pending),
                    "batch_records": len(positions),
                    "solver_seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        processed_batches += 1
        if max_batches is not None and processed_batches >= int(max_batches):
            break
    with h5py.File(Path(output_h5), "r", swmr=True) as handle:
        completed_count = int(np.count_nonzero(handle["completed"][:]))
    complete = completed_count == len(records)
    report = {
        "status": "complete" if complete else "partial",
        "shard_index": int(shard_index),
        "record_count": len(records),
        "completed_count": completed_count,
        "time_count": len(time_indices),
        "solver_seconds": solver_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(requested_device))
            if requested_device.type == "cuda"
            else 0
        ),
        "output_h5": str(Path(output_h5).resolve()),
    }
    print(
        json.dumps(
            {
                "event": (
                    "cache_shard_complete" if complete else "cache_shard_partial"
                ),
                **report,
            },
            sort_keys=True,
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--solver-batch-size", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--time-count", type=int, default=64)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--c-ref-mps", type=float, default=6750.0)
    parser.add_argument("--max-batches", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_shard(
        source_h5=args.source_h5,
        output_h5=args.output_h5,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        solver_batch_size=args.solver_batch_size,
        device=args.device,
        time_count=args.time_count,
        dataset_config=args.dataset_config,
        manifest_path=args.manifest,
        c_ref_mps=args.c_ref_mps,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
