#!/usr/bin/env python3
"""Validate LWC-84 teacher shards and expose them as one zero-copy HDF5 VDS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from saved_time_phase_operator_v4.multifidelity import TEACHER_CACHE_SCHEMA
from scripts.build_lwc84_multifidelity_cache import (
    SHARD_SCHEMA,
    TrainingRecord,
    load_exact_training_numerical_contract,
    load_manifest_index,
    reconstruct_generation_inputs,
    select_training_records,
    shard_training_records,
    source_identity as build_source_identity,
    validate_source_dataset_contract,
)
from fno_acoustic.data_generation.config import load_config
from saved_time_phase_operator_v4.multifidelity import fixed_teacher_time_indices


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _contiguous_record_runs(
    records: Sequence[TrainingRecord],
) -> tuple[tuple[int, int, int, int], ...]:
    """Return `(out_start,out_stop,source_start,source_stop)` VDS runs."""

    rows = tuple(records)
    if not rows:
        raise ValueError("direct teacher VDS requires records")
    runs: list[tuple[int, int, int, int]] = []
    output_start = 0
    source_start = rows[0].source_index
    previous = source_start
    for output_index, record in enumerate(rows[1:], start=1):
        current = int(record.source_index)
        if current != previous + 1:
            runs.append((output_start, output_index, source_start, previous + 1))
            output_start = output_index
            source_start = current
        previous = current
    runs.append((output_start, len(rows), source_start, previous + 1))
    if any(
        output_stop - output_start != source_stop - source_start
        for output_start, output_stop, source_start, source_stop in runs
    ):
        raise RuntimeError("direct teacher VDS run construction failed")
    return tuple(runs)


def build_source_teacher_vds(
    source_h5: str | Path,
    output_h5: str | Path,
    *,
    expected_records: Sequence[TrainingRecord],
    source_identity: Mapping[str, object],
    numerical_contract: Mapping[str, object],
    time_indices: Sequence[int],
) -> dict[str, object]:
    """Expose exact 401-to-201 training solutions without copying their fields."""

    source_path = Path(source_h5).expanduser().resolve()
    destination = Path(output_h5).expanduser().resolve()
    records = tuple(expected_records)
    indices = tuple(int(value) for value in time_indices)
    if (
        not source_path.is_file()
        or not records
        or not indices
        or indices[0] != 0
        or any(left >= right for left, right in zip(indices, indices[1:]))
    ):
        raise ValueError("direct teacher VDS inputs are invalid")
    with h5py.File(source_path, "r", swmr=True) as source_handle:
        source_shape = tuple(int(value) for value in source_handle["wavefield"].shape)
        source_dtype = source_handle["wavefield"].dtype
        source_times = np.asarray(source_handle["time_s"][:], dtype=np.float64)
    spatial_shape = tuple(int(value) for value in numerical_contract["saved_grid_shape"])
    if (
        len(source_shape) != 4
        or source_shape[1] != int(numerical_contract["output_time_count"])
        or source_shape[-2:] != spatial_shape
        or source_dtype != np.dtype(np.float32)
        or indices[-1] >= source_shape[1]
        or max(record.source_index for record in records) >= source_shape[0]
    ):
        raise ValueError("source wavefield cannot satisfy the direct teacher layout")
    runs = _contiguous_record_runs(records)
    layout = h5py.VirtualLayout(
        shape=(len(records), len(indices), *spatial_shape), dtype=np.float32
    )
    source = h5py.VirtualSource(source_path, "wavefield", shape=source_shape)
    for teacher_time, source_time in enumerate(indices):
        for output_start, output_stop, source_start, source_stop in runs:
            layout[output_start:output_stop, teacher_time, :, :] = source[
                source_start:source_stop, source_time, :, :
            ]
    identity_json = _canonical_json(source_identity)
    numerics = {
        **dict(numerical_contract),
        "teacher_semantics": "zero_copy_exact_training_solver_output",
    }
    numerics_json = _canonical_json(numerics)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        with h5py.File(partial, "w", libver="latest") as handle:
            handle.attrs["schema"] = TEACHER_CACHE_SCHEMA
            handle.attrs["status"] = "complete"
            handle.attrs["source_manifest_sha256"] = str(
                source_identity.get("manifest_sha256", "")
            )
            handle.attrs["source_identity_json"] = identity_json
            handle.attrs["numerical_contract_json"] = numerics_json
            handle.attrs["zero_copy_source_vds"] = True
            text_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "sample_id",
                data=np.asarray([record.sample_id for record in records], dtype=object),
                dtype=text_dtype,
            )
            handle.create_dataset(
                "source_index",
                data=np.asarray([record.source_index for record in records], dtype=np.int64),
            )
            handle.create_dataset(
                "medium_type",
                data=np.asarray([record.medium_type for record in records], dtype=object),
                dtype=text_dtype,
            )
            handle.create_dataset("time_indices", data=np.asarray(indices, dtype=np.int64))
            handle.create_dataset(
                "time_s", data=source_times[np.asarray(indices, dtype=np.int64)]
            )
            handle.create_virtual_dataset("wavefield", layout, fillvalue=np.nan)
            handle.flush()
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    report = {
        "schema": TEACHER_CACHE_SCHEMA,
        "status": "complete",
        "record_count": len(records),
        "time_count": len(indices),
        "spatial_shape": list(spatial_shape),
        "source_manifest_sha256": str(source_identity.get("manifest_sha256", "")),
        "output_h5": str(destination),
        "vds_byte_count": int(destination.stat().st_size),
        "backing_source_h5": str(source_path),
        "zero_copy": True,
        "mapping_run_count": len(runs),
    }
    _write_json_atomic(destination.with_suffix(".summary.json"), report)
    return report


def merge_cache_shards(
    shard_paths: Sequence[str | Path],
    output_h5: str | Path,
    *,
    expected_records: Sequence[TrainingRecord],
    source_identity: Mapping[str, object],
    numerical_contract: Mapping[str, object],
) -> dict[str, object]:
    """Create a complete global-order VDS only after every shard validates."""

    paths = tuple(Path(path).expanduser().resolve() for path in shard_paths)
    records = tuple(expected_records)
    if not paths or not records or any(not path.is_file() for path in paths):
        raise ValueError("cache merge requires all shard files and expected records")
    shard_count = len(paths)
    identity_json = _canonical_json(source_identity)
    numerics_json = _canonical_json(numerical_contract)
    time_indices: tuple[int, ...] | None = None
    time_s: tuple[float, ...] | None = None
    spatial_shape: tuple[int, int] | None = None
    shard_shapes: list[tuple[int, ...]] = []
    for shard_index, path in enumerate(paths):
        expected_shard = shard_training_records(
            records, shard_index=shard_index, shard_count=shard_count
        )
        with h5py.File(path, "r", swmr=True) as handle:
            if (
                str(handle.attrs.get("schema", "")) != SHARD_SCHEMA
                or str(handle.attrs.get("status", "")) != "complete"
                or str(handle.attrs.get("source_identity_json", "")) != identity_json
                or str(handle.attrs.get("numerical_contract_json", "")) != numerics_json
                or not bool(np.asarray(handle["completed"][:], dtype=bool).all())
            ):
                raise ValueError(f"cache shard identity or completion mismatch: {path}")
            samples = tuple(handle["sample_id"].asstr()[:].tolist())
            sources = tuple(int(value) for value in handle["source_index"][:])
            families = tuple(handle["medium_type"].asstr()[:].tolist())
            expected_samples = tuple(record.sample_id for record in expected_shard)
            expected_sources = tuple(record.source_index for record in expected_shard)
            expected_families = tuple(record.medium_type for record in expected_shard)
            if (
                samples != expected_samples
                or sources != expected_sources
                or families != expected_families
                or handle["wavefield"].dtype != np.dtype(np.float32)
            ):
                raise ValueError(f"cache shard record mapping mismatch: {path}")
            current_indices = tuple(int(value) for value in handle["time_indices"][:])
            current_times = tuple(float(value) for value in handle["time_s"][:])
            current_shape = tuple(int(value) for value in handle["wavefield"].shape)
            if current_shape[:2] != (len(expected_shard), len(current_indices)):
                raise ValueError(f"cache shard wavefield shape mismatch: {path}")
            if time_indices is None:
                time_indices = current_indices
                time_s = current_times
                spatial_shape = current_shape[-2:]
            elif (
                current_indices != time_indices
                or current_times != time_s
                or current_shape[-2:] != spatial_shape
            ):
                raise ValueError("cache shards use different time or spatial axes")
            shard_shapes.append(current_shape)
    assert time_indices is not None and time_s is not None and spatial_shape is not None
    destination = Path(output_h5).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    layout = h5py.VirtualLayout(
        shape=(len(records), len(time_indices), *spatial_shape), dtype=np.float32
    )
    for shard_index, (path, shape) in enumerate(zip(paths, shard_shapes, strict=True)):
        source = h5py.VirtualSource(str(path), "wavefield", shape=shape)
        for local_index in range(shape[0]):
            global_index = shard_index + local_index * shard_count
            layout[global_index] = source[local_index]
    try:
        with h5py.File(partial, "w", libver="latest") as handle:
            handle.attrs["schema"] = TEACHER_CACHE_SCHEMA
            handle.attrs["status"] = "complete"
            handle.attrs["source_manifest_sha256"] = str(
                source_identity.get("manifest_sha256", "")
            )
            handle.attrs["source_identity_json"] = identity_json
            handle.attrs["numerical_contract_json"] = numerics_json
            handle.attrs["shard_count"] = shard_count
            text_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset(
                "sample_id",
                data=np.asarray([record.sample_id for record in records], dtype=object),
                dtype=text_dtype,
            )
            handle.create_dataset(
                "source_index",
                data=np.asarray([record.source_index for record in records], dtype=np.int64),
            )
            handle.create_dataset(
                "medium_type",
                data=np.asarray([record.medium_type for record in records], dtype=object),
                dtype=text_dtype,
            )
            handle.create_dataset(
                "time_indices", data=np.asarray(time_indices, dtype=np.int64)
            )
            handle.create_dataset("time_s", data=np.asarray(time_s, dtype=np.float64))
            handle.create_dataset(
                "shard_path",
                data=np.asarray([str(path) for path in paths], dtype=object),
                dtype=text_dtype,
            )
            handle.create_virtual_dataset("wavefield", layout, fillvalue=np.nan)
            handle.flush()
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    report = {
        "schema": TEACHER_CACHE_SCHEMA,
        "status": "complete",
        "record_count": len(records),
        "time_count": len(time_indices),
        "spatial_shape": list(spatial_shape),
        "shard_count": shard_count,
        "source_manifest_sha256": str(source_identity.get("manifest_sha256", "")),
        "output_h5": str(destination),
        "vds_byte_count": int(destination.stat().st_size),
        "backing_byte_count": int(sum(path.stat().st_size for path in paths)),
    }
    _write_json_atomic(destination.with_suffix(".summary.json"), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--source-direct", action="store_true")
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--time-count", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = select_training_records(args.source_h5)
    if args.source_direct:
        config_path = (
            args.source_h5.parent / "frozen_config.yaml"
            if args.dataset_config is None
            else args.dataset_config
        )
        manifest_path = (
            args.source_h5.parent / "manifest.jsonl"
            if args.manifest is None
            else args.manifest
        )
        identity = build_source_identity(
            args.source_h5,
            manifest_path=manifest_path,
            dataset_config=config_path,
        )
        config = load_config(config_path)
        numerical_contract = load_exact_training_numerical_contract(config_path)
        manifest_rows = load_manifest_index(
            manifest_path, expected_sha256=str(identity["manifest_sha256"])
        )
        validate_source_dataset_contract(
            args.source_h5,
            config=config,
            manifest_rows=manifest_rows,
            numerical_contract=numerical_contract,
        )
        # Reconstruct one sample per permitted family before publishing the view.
        selected = tuple(
            next(record for record in records if record.medium_type == family)
            for family in ("uniform", "layered", "marmousi")
        )
        reconstruct_generation_inputs(
            args.source_h5,
            selected,
            config=config,
            manifest_rows=manifest_rows,
        )
        report = build_source_teacher_vds(
            args.source_h5,
            args.output_h5,
            expected_records=records,
            source_identity=identity,
            numerical_contract=numerical_contract,
            time_indices=fixed_teacher_time_indices(
                stored_time_count=int(numerical_contract["output_time_count"]),
                count=int(args.time_count),
            ),
        )
        print(json.dumps(report, sort_keys=True), flush=True)
        return
    if args.shard_dir is None:
        raise ValueError("--shard-dir is required unless --source-direct is used")
    paths = tuple(
        args.shard_dir / f"shard_{index:02d}.h5"
        for index in range(int(args.shard_count))
    )
    if not paths:
        raise ValueError("shard count must be positive")
    with h5py.File(paths[0], "r", swmr=True) as handle:
        numerical_contract = json.loads(
            str(handle.attrs["numerical_contract_json"])
        )
    report = merge_cache_shards(
        paths,
        args.output_h5,
        expected_records=records,
        source_identity=build_source_identity(args.source_h5),
        numerical_contract=numerical_contract,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
