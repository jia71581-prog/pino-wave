#!/usr/bin/env python3
"""Build a train-only coarse-LWC cache for the R25 residual neural operator.

The script has two modes:

* ``manifest`` freezes group-disjoint fit/holdout records from the train split.
* ``cache`` runs the already-audited 201x201 LWC-84 solver, then stores
  normalized coarse fields, train truth targets, and deployment-available
  static features.  Truth is read only after an entire solver batch has been
  materialized; validation and test_id are never selected.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from fno_acoustic.data_generation.solver_lwc84 import LWC84CPMLSolver
from saved_time_phase_operator_v4.coarse_lwc84 import CoarseRecordIndex
from scripts.evaluate_coarse_lwc84_201_shard import (
    _read_axes,
    _read_generation_inputs,
    _validate_axes,
    _validate_solver_batch,
    _warm_up_cuda,
)


FAMILIES = ("uniform", "layered", "marmousi")
INTERNAL_DT_S = 1.25e-4
NPML = 20
C_REF_MPS = 6750.0
GRID_SIZE = 201
STORED_TIME_COUNT = 401
SCHEMA_MANIFEST = "r25_coarse_residual_manifest_v1"
SCHEMA_CACHE = "r25_coarse_residual_cache_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def text_array(handle: h5py.File, name: str) -> np.ndarray:
    return np.asarray(handle[name].asstr()[:], dtype=str)


def record_row(
    index: int,
    *,
    sample_ids: np.ndarray,
    group_ids: np.ndarray,
    sample_hashes: np.ndarray,
    families: np.ndarray,
) -> dict[str, Any]:
    return {
        "source_index": int(index),
        "sample_id": str(sample_ids[index]),
        "group_id": str(group_ids[index]),
        "sample_sha256": str(sample_hashes[index]),
        "family": str(families[index]),
        "split": "train",
    }


def take_group_exclusive_records(
    group_order: Sequence[str],
    group_to_indices: Mapping[str, Sequence[int]],
    *,
    count: int,
) -> tuple[list[int], list[str], int]:
    chosen: list[int] = []
    used_groups: list[str] = []
    consumed = 0
    for group in group_order:
        if len(chosen) >= int(count):
            break
        indices = list(group_to_indices[group])
        remaining = int(count) - len(chosen)
        chosen.extend(indices[:remaining])
        used_groups.append(str(group))
        consumed += 1
    if len(chosen) != int(count):
        raise RuntimeError(f"could select only {len(chosen)} of {count} records")
    return chosen, used_groups, consumed


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_h5 = args.source_h5.expanduser().resolve()
    if not source_h5.is_file():
        raise FileNotFoundError(source_h5)
    with h5py.File(source_h5, "r", swmr=True) as handle:
        split = text_array(handle, "split")
        families = text_array(handle, "medium_type")
        sample_ids = text_array(handle, "sample_id")
        group_ids = text_array(handle, "group_id")
        sample_hashes = text_array(handle, "sample_sha256")
        source_manifest_sha256 = str(handle.attrs.get("manifest_sha256", ""))
        source_config_sha256 = str(handle.attrs.get("config_sha256", ""))
    lengths = {
        len(value)
        for value in (split, families, sample_ids, group_ids, sample_hashes)
    }
    if len(lengths) != 1:
        raise RuntimeError("source index datasets have inconsistent lengths")

    fit_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    family_group_counts: dict[str, dict[str, int]] = {}
    for family_index, family in enumerate(FAMILIES):
        candidates = np.flatnonzero((split == "train") & (families == family))
        if candidates.size == 0:
            raise RuntimeError(f"train family is empty: {family}")
        group_to_indices: dict[str, list[int]] = {}
        for raw_index in candidates:
            index = int(raw_index)
            group_to_indices.setdefault(str(group_ids[index]), []).append(index)
        groups = sorted(group_to_indices)
        rng = np.random.default_rng(int(args.seed) + 1009 * family_index)
        rng.shuffle(groups)

        held, held_groups, consumed = take_group_exclusive_records(
            groups,
            group_to_indices,
            count=int(args.holdout_per_family),
        )
        remaining_groups = groups[consumed:]
        fit, fit_groups, _ = take_group_exclusive_records(
            remaining_groups,
            group_to_indices,
            count=int(args.fit_per_family),
        )
        if set(held_groups) & set(fit_groups):
            raise RuntimeError(f"group leakage within {family}")
        holdout_rows.extend(
            record_row(
                index,
                sample_ids=sample_ids,
                group_ids=group_ids,
                sample_hashes=sample_hashes,
                families=families,
            )
            for index in held
        )
        fit_rows.extend(
            record_row(
                index,
                sample_ids=sample_ids,
                group_ids=group_ids,
                sample_hashes=sample_hashes,
                families=families,
            )
            for index in fit
        )
        family_group_counts[family] = {
            "fit_records": len(fit),
            "fit_groups": len(set(fit_groups)),
            "holdout_records": len(held),
            "holdout_groups": len(set(held_groups)),
        }

    fit_groups_all = {row["group_id"] for row in fit_rows}
    holdout_groups_all = {row["group_id"] for row in holdout_rows}
    if fit_groups_all & holdout_groups_all:
        raise RuntimeError("fit/holdout group leakage")
    if len({row["sample_id"] for row in fit_rows + holdout_rows}) != len(
        fit_rows + holdout_rows
    ):
        raise RuntimeError("selected sample IDs are not unique")

    fit_times = np.rint(
        np.linspace(0, STORED_TIME_COUNT - 1, int(args.fit_time_count))
    ).astype(np.int64)
    if len(np.unique(fit_times)) != int(args.fit_time_count):
        raise RuntimeError("fit time selection is not unique")
    payload: dict[str, Any] = {
        "schema": SCHEMA_MANIFEST,
        "status": "frozen_before_cache_generation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "source_h5": str(source_h5),
        "source_h5_byte_count": source_h5.stat().st_size,
        "source_manifest_sha256": source_manifest_sha256,
        "source_config_sha256": source_config_sha256,
        "split_policy": {
            "fit": "train_only",
            "holdout": "train_only_group_disjoint_from_fit",
            "validation_opened": False,
            "test_id_opened": False,
        },
        "numerical_contract": {
            "method": "LWC-84",
            "grid": [GRID_SIZE, GRID_SIZE],
            "spacing_m": 10.0,
            "internal_dt_s": INTERNAL_DT_S,
            "stored_dt_s": 0.0025,
            "stored_time_count": STORED_TIME_COUNT,
            "npml": NPML,
            "c_ref_mps": C_REF_MPS,
            "dtype": "float32",
        },
        "fit_time_indices": fit_times.tolist(),
        "holdout_time_indices": list(range(STORED_TIME_COUNT)),
        "family_group_counts": family_group_counts,
        "fit_records": fit_rows,
        "holdout_records": holdout_rows,
        "deployment_input_contract": [
            "coarse_lwc84_wavefield",
            "velocity_model",
            "source_location",
            "source_frequency",
            "source_onset",
            "query_time",
        ],
        "truth_policy": "high_fidelity wavefield is train-only supervision and is never a deployment input",
        "pilot_gate": {
            "primary": "group-disjoint train holdout record-relative L2",
            "scale_if_mean_relative_improvement_gte": 0.10,
            "scale_if_max_relative_improvement_gte": 0.10,
            "absolute_goal_mean_lte": 0.05,
            "absolute_goal_max_lte": 0.05,
        },
    }
    payload["selection_sha256"] = canonical_json_sha256(payload)
    atomic_json(payload, args.output)
    return payload


def rows_to_records(rows: Sequence[Mapping[str, Any]]) -> tuple[CoarseRecordIndex, ...]:
    records = tuple(
        CoarseRecordIndex(
            source_index=int(row["source_index"]),
            sample_id=str(row["sample_id"]),
            group_id=str(row["group_id"]),
            sample_sha256=str(row["sample_sha256"]),
            split=str(row["split"]),
            medium_type=str(row["family"]),
        )
        for row in rows
    )
    if any(record.split != "train" for record in records):
        raise RuntimeError("cache selection contains a non-train record")
    return records


def static_features(
    velocity: np.ndarray,
    *,
    x_m: np.ndarray,
    z_m: np.ndarray,
    source_x_m: float,
    source_z_m: float,
) -> np.ndarray:
    value = np.asarray(velocity, dtype=np.float32)
    logv = np.log(np.maximum(value, 1.0))
    grad_z, grad_x = np.gradient(logv)
    grad_x = np.clip(20.0 * grad_x, -4.0, 4.0)
    grad_z = np.clip(20.0 * grad_z, -4.0, 4.0)
    xx, zz = np.meshgrid(x_m.astype(np.float32), z_m.astype(np.float32))
    distance = np.sqrt((xx - float(source_x_m)) ** 2 + (zz - float(source_z_m)) ** 2)
    source_map = np.exp(-0.5 * (distance / 20.0) ** 2)
    source_map /= max(float(source_map.max()), 1.0e-12)
    travel = distance / np.maximum(value, 1.0)
    x_norm = 2.0 * xx / max(float(x_m[-1]), 1.0) - 1.0
    z_norm = 2.0 * zz / max(float(z_m[-1]), 1.0) - 1.0
    return np.stack(
        [
            np.clip((value - 4500.0) / 2500.0, -2.0, 2.0),
            grad_x,
            grad_z,
            source_map.astype(np.float32),
            np.clip(travel, 0.0, 1.5).astype(np.float32),
            x_norm.astype(np.float32),
            z_norm.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def relative_l2_squares(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    truth = np.asarray(target, dtype=np.float64)
    return float(np.square(difference).sum()), float(np.square(truth).sum())


def create_cache(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_MANIFEST:
        raise RuntimeError("unexpected R25 manifest schema")
    expected_selection_sha = str(manifest.get("selection_sha256", ""))
    verification_payload = dict(manifest)
    verification_payload.pop("selection_sha256", None)
    if canonical_json_sha256(verification_payload) != expected_selection_sha:
        raise RuntimeError("R25 manifest selection digest mismatch")
    source_h5 = Path(str(manifest["source_h5"])).resolve()
    if source_h5.stat().st_size != int(manifest["source_h5_byte_count"]):
        raise RuntimeError("source HDF5 byte count changed after manifest freeze")
    with h5py.File(source_h5, "r", swmr=True) as source_identity_handle:
        if str(source_identity_handle.attrs.get("manifest_sha256", "")) != str(
            manifest["source_manifest_sha256"]
        ):
            raise RuntimeError("source HDF5 manifest identity changed after freeze")
        if str(source_identity_handle.attrs.get("config_sha256", "")) != str(
            manifest["source_config_sha256"]
        ):
            raise RuntimeError("source HDF5 config identity changed after freeze")
    subset_key = f"{args.subset}_records"
    raw_rows = list(manifest[subset_key])
    records_all = rows_to_records(raw_rows)
    records = records_all[int(args.shard_index) :: int(args.shard_count)]
    if not records:
        raise RuntimeError("cache shard is empty")
    time_indices = np.asarray(
        manifest[f"{args.subset}_time_indices"], dtype=np.int64
    )
    if np.any(time_indices < 0) or np.any(time_indices >= STORED_TIME_COUNT):
        raise RuntimeError("cache time index is outside the stored axis")

    output = args.output.expanduser().resolve()
    if output.exists():
        with h5py.File(output, "r", swmr=True) as existing:
            if (
                str(existing.attrs.get("schema", "")) == SCHEMA_CACHE
                and str(existing.attrs.get("status", "")) == "complete"
                and str(existing.attrs.get("selection_sha256", ""))
                == expected_selection_sha
                and str(existing.attrs.get("subset", "")) == args.subset
                and int(existing.attrs.get("shard_index", -1))
                == int(args.shard_index)
                and int(existing.attrs.get("shard_count", -1))
                == int(args.shard_count)
            ):
                return {
                    "status": "already_complete",
                    "output": str(output),
                    "sha256": sha256_file(output),
                }
        raise FileExistsError(f"incompatible cache already exists: {output}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R25 cache generation requires CUDA")
    torch.cuda.set_device(device)
    time_s, x_m, z_m = _read_axes(source_h5)
    stored_dt_s, dx_m, dz_m = _validate_axes(
        time_s, x_m, z_m, internal_dt_s=INTERNAL_DT_S
    )
    if (
        len(time_s) != STORED_TIME_COUNT
        or len(x_m) != GRID_SIZE
        or len(z_m) != GRID_SIZE
    ):
        raise RuntimeError("R25 cache requires the production 401x201x201 axes")
    _warm_up_cuda(
        dx_m=dx_m,
        dz_m=dz_m,
        internal_dt_s=INTERNAL_DT_S,
        npml=NPML,
        c_ref_mps=C_REF_MPS,
        device=device,
    )
    solver = LWC84CPMLSolver(
        grid=AcousticGrid(
            nx=GRID_SIZE,
            nz=GRID_SIZE,
            dx_m=dx_m,
            dz_m=dz_m,
            lx_m=float(x_m[-1] - x_m[0]),
            lz_m=float(z_m[-1] - z_m[0]),
            centering="node",
        ),
        boundaries=BoundaryConfig(npml=NPML),
        dt_s=INTERNAL_DT_S,
        output_times_s=time_s,
        c_ref_mps=C_REF_MPS,
        device=device,
        dtype=torch.float32,
        output_restriction_factor=1,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    text_dtype = h5py.string_dtype(encoding="utf-8")
    started = time.perf_counter()
    solver_seconds = 0.0
    error_squares: list[float] = []
    target_squares: list[float] = []
    family_errors: dict[str, list[float]] = {name: [] for name in FAMILIES}
    try:
        with h5py.File(temporary, "x") as cache:
            cache.attrs["schema"] = SCHEMA_CACHE
            cache.attrs["status"] = "building"
            cache.attrs["selection_sha256"] = expected_selection_sha
            cache.attrs["manifest_sha256"] = sha256_file(manifest_path)
            cache.attrs["source_h5_byte_count"] = int(
                manifest["source_h5_byte_count"]
            )
            cache.attrs["source_manifest_sha256"] = str(
                manifest["source_manifest_sha256"]
            )
            cache.attrs["subset"] = args.subset
            cache.attrs["shard_index"] = int(args.shard_index)
            cache.attrs["shard_count"] = int(args.shard_count)
            cache.attrs["internal_dt_s"] = INTERNAL_DT_S
            cache.attrs["stored_dt_s"] = stored_dt_s
            cache.attrs["truth_policy"] = "train_only_supervision_not_deployment_input"
            count = len(records)
            frames = len(time_indices)
            cache.create_dataset(
                "source_index",
                data=np.asarray([record.source_index for record in records], dtype=np.int64),
            )
            cache.create_dataset(
                "sample_id",
                data=np.asarray([record.sample_id for record in records], dtype=object),
                dtype=text_dtype,
            )
            cache.create_dataset(
                "group_id",
                data=np.asarray([record.group_id for record in records], dtype=object),
                dtype=text_dtype,
            )
            cache.create_dataset(
                "family",
                data=np.asarray([record.medium_type for record in records], dtype=object),
                dtype=text_dtype,
            )
            cache.create_dataset("time_indices", data=time_indices)
            cache.create_dataset("time_s", data=time_s[time_indices])
            scale_ds = cache.create_dataset("field_scale", shape=(count,), dtype=np.float32)
            f0_ds = cache.create_dataset("source_f0_hz", shape=(count,), dtype=np.float32)
            t0_ds = cache.create_dataset("source_t0_s", shape=(count,), dtype=np.float32)
            coarse_ds = cache.create_dataset(
                "coarse_norm",
                shape=(count, frames, GRID_SIZE, GRID_SIZE),
                dtype=np.float16,
                chunks=(1, 1, GRID_SIZE, GRID_SIZE),
                compression="lzf",
                shuffle=True,
            )
            truth_ds = cache.create_dataset(
                "truth_norm",
                shape=(count, frames, GRID_SIZE, GRID_SIZE),
                dtype=np.float16,
                chunks=(1, 1, GRID_SIZE, GRID_SIZE),
                compression="lzf",
                shuffle=True,
            )
            static_ds = cache.create_dataset(
                "static_features",
                shape=(count, 7, GRID_SIZE, GRID_SIZE),
                dtype=np.float16,
                chunks=(1, 7, GRID_SIZE, GRID_SIZE),
                compression="lzf",
                shuffle=True,
            )
            max_energy_ds = cache.create_dataset(
                "truth_frame_energy_max_norm", shape=(count,), dtype=np.float32
            )
            mean_energy_ds = cache.create_dataset(
                "truth_frame_energy_mean_norm", shape=(count,), dtype=np.float32
            )
            baseline_error_ds = cache.create_dataset(
                "baseline_error_square_norm", shape=(count,), dtype=np.float64
            )
            target_error_ds = cache.create_dataset(
                "target_square_norm", shape=(count,), dtype=np.float64
            )

            batch_size = int(args.solver_batch_size)
            for batch_start in range(0, count, batch_size):
                batch_records = records[batch_start : batch_start + batch_size]
                velocity, source = _read_generation_inputs(source_h5, batch_records)
                torch.cuda.synchronize(device)
                solve_started = time.perf_counter()
                result = solver.simulate(
                    velocity,
                    source_x_m=source["source_x_m"],
                    source_z_m=source["source_z_m"],
                    source_f0_hz=source["source_f0_hz"],
                    source_t0_s=source["source_t0_s"],
                    source_amplitude=source["source_amplitude"],
                )
                torch.cuda.synchronize(device)
                solver_seconds += time.perf_counter() - solve_started
                _validate_solver_batch(
                    result,
                    velocity,
                    records=batch_records,
                    stored_time_count=STORED_TIME_COUNT,
                )
                prediction_batch = np.asarray(result.wavefield, dtype=np.float32)
                prediction_hashes = [
                    hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
                    for value in prediction_batch
                ]
                predicted_before_truth_ns = time.monotonic_ns()
                indices = [record.source_index for record in batch_records]
                with h5py.File(source_h5, "r", swmr=True) as source_handle:
                    truth_batch = np.stack(
                        [
                            np.asarray(
                                source_handle["wavefield"][index, time_indices],
                                dtype=np.float32,
                            )
                            for index in indices
                        ],
                        axis=0,
                    )
                if time.monotonic_ns() <= predicted_before_truth_ns:
                    raise RuntimeError("predict-before-truth monotonic audit failed")

                for local, record in enumerate(batch_records):
                    row = batch_start + local
                    full_prediction = prediction_batch[local]
                    selected_prediction = full_prediction[time_indices]
                    truth = truth_batch[local]
                    scale = max(float(np.max(np.abs(full_prediction))), 1.0e-12)
                    coarse_norm = selected_prediction / scale
                    truth_norm = truth / scale
                    error_square, target_square = relative_l2_squares(
                        coarse_norm, truth_norm
                    )
                    relative = math.sqrt(error_square / max(target_square, 1.0e-30))
                    error_squares.append(error_square)
                    target_squares.append(target_square)
                    family_errors[record.medium_type].append(relative)
                    scale_ds[row] = scale
                    f0_ds[row] = float(source["source_f0_hz"][local])
                    t0_ds[row] = float(source["source_t0_s"][local])
                    coarse_ds[row] = coarse_norm.astype(np.float16)
                    truth_ds[row] = truth_norm.astype(np.float16)
                    static_ds[row] = static_features(
                        velocity[local],
                        x_m=x_m,
                        z_m=z_m,
                        source_x_m=float(source["source_x_m"][local]),
                        source_z_m=float(source["source_z_m"][local]),
                    ).astype(np.float16)
                    frame_energies = np.square(truth_norm.astype(np.float64)).mean(
                        axis=(1, 2)
                    )
                    max_energy_ds[row] = float(frame_energies.max())
                    mean_energy_ds[row] = float(frame_energies.mean())
                    baseline_error_ds[row] = error_square
                    target_error_ds[row] = target_square
                    print(
                        json.dumps(
                            {
                                "event": "cached_record",
                                "subset": args.subset,
                                "shard": int(args.shard_index),
                                "row": row,
                                "count": count,
                                "sample_id": record.sample_id,
                                "family": record.medium_type,
                                "baseline_rel_l2_selected": relative,
                                "prediction_sha256": prediction_hashes[local],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                cache.flush()
                del result, prediction_batch, truth_batch
                torch.cuda.empty_cache()

            record_rel = [
                math.sqrt(error / max(target, 1.0e-30))
                for error, target in zip(error_squares, target_squares, strict=True)
            ]
            cache.attrs["baseline_record_rel_l2_mean"] = float(np.mean(record_rel))
            cache.attrs["baseline_record_rel_l2_max"] = float(np.max(record_rel))
            cache.attrs["solver_seconds"] = float(solver_seconds)
            cache.attrs["elapsed_seconds"] = float(time.perf_counter() - started)
            cache.attrs["status"] = "complete"
            cache.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "schema": "r25_coarse_residual_cache_summary_v1",
        "status": "complete",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "selection_sha256": expected_selection_sha,
        "manifest_sha256": sha256_file(manifest_path),
        "subset": args.subset,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "record_count": len(records),
        "frame_count_per_record": len(time_indices),
        "baseline_record_rel_l2_mean": float(
            np.mean(
                [
                    math.sqrt(error / max(target, 1.0e-30))
                    for error, target in zip(
                        error_squares, target_squares, strict=True
                    )
                ]
            )
        ),
        "baseline_record_rel_l2_max": float(
            np.max(
                [
                    math.sqrt(error / max(target, 1.0e-30))
                    for error, target in zip(
                        error_squares, target_squares, strict=True
                    )
                ]
            )
        ),
        "family_record_rel_l2": {
            family: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "max": float(np.max(values)) if values else None,
            }
            for family, values in family_errors.items()
        },
        "solver_seconds": float(solver_seconds),
        "elapsed_seconds": float(time.perf_counter() - started),
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(summary, output.with_suffix(output.suffix + ".summary.json"))
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    subparsers = value.add_subparsers(dest="mode", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--source-h5", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--seed", type=int, default=250827)
    manifest.add_argument("--fit-per-family", type=int, default=64)
    manifest.add_argument("--holdout-per-family", type=int, default=8)
    manifest.add_argument("--fit-time-count", type=int, default=64)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--subset", choices=("fit", "holdout"), required=True)
    cache.add_argument("--shard-index", type=int, required=True)
    cache.add_argument("--shard-count", type=int, default=4)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--solver-batch-size", type=int, default=8)
    cache.add_argument("--device", default="cuda:0")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.mode == "manifest":
        result = build_manifest(args)
    else:
        result = create_cache(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
