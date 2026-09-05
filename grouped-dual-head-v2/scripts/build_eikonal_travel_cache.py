#!/usr/bin/env python
"""Build a multiprocessing source-dependent travel-time cache from the HDF5 VDS."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time


_WORKER_H5: h5py.File | None = None


def _decode(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values)


def _init_worker(source_h5: str) -> None:
    global _WORKER_H5
    _WORKER_H5 = h5py.File(source_h5, "r")


def _solve_group(task: tuple[str, tuple[int, ...]]) -> tuple[tuple[int, ...], np.ndarray]:
    if _WORKER_H5 is None:
        raise RuntimeError("travel-cache worker HDF5 was not initialized")
    _, indices = task
    first = int(indices[0])
    velocity = _WORKER_H5["velocity_mps"][first]
    x_m = _WORKER_H5["x_m"][:]
    z_m = _WORKER_H5["z_m"][:]
    source_x = _WORKER_H5["source_x_m"][list(indices)]
    source_z = _WORKER_H5["source_z_m"][list(indices)]
    source_indices = tuple(
        (
            int(np.argmin(np.abs(z_m - float(source_z_value)))),
            int(np.argmin(np.abs(x_m - float(source_x_value)))),
        )
        for source_x_value, source_z_value in zip(source_x, source_z, strict=True)
    )
    travel = grid_eikonal_travel_time(
        velocity,
        source_indices=source_indices,
        dx_m=float(np.median(np.diff(x_m))),
        dz_m=float(np.median(np.diff(z_m))),
    )
    return indices, travel


def _atomic_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--families", default="uniform,layered,marmousi")
    parser.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--limit-groups", type=int)
    args = parser.parse_args(argv)

    source = Path(args.source_h5).resolve()
    output = Path(args.output).resolve()
    if not source.exists() or args.workers <= 0:
        raise ValueError("source HDF5 must exist and workers must be positive")
    selected_splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    if not selected_splits:
        raise ValueError("at least one split must be selected")
    selected_families = {
        item.strip() for item in str(args.families).split(",") if item.strip()
    }
    if not selected_families:
        raise ValueError("at least one medium family must be selected")

    with h5py.File(source, "r") as handle:
        completed = handle["completed_mask"][:].astype(bool)
        split = _decode(handle["split"][:])
        group_id = _decode(handle["group_id"][:])
        sample_id = _decode(handle["sample_id"][:])
        medium_type = _decode(handle["medium_type"][:])
        selected = (
            completed
            & np.isin(split, tuple(sorted(selected_splits)))
            & np.isin(medium_type, tuple(sorted(selected_families)))
        )
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for index in np.flatnonzero(selected):
            grouped[str(group_id[index])].append(int(index))
        tasks = tuple(
            (name, tuple(indices)) for name, indices in sorted(grouped.items())
        )
        if args.limit_groups is not None:
            if int(args.limit_groups) <= 0:
                raise ValueError("limit-groups must be positive")
            tasks = tasks[: int(args.limit_groups)]
        selected_indices = tuple(index for _, indices in tasks for index in indices)
        if not selected_indices:
            raise ValueError("selected splits contain no completed records")
        x_m = handle["x_m"][:]
        z_m = handle["z_m"][:]
        rows = len(selected_indices)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    started = time.monotonic()
    row = 0
    digest = hashlib.sha256()
    with h5py.File(temporary, "w") as target:
        target.attrs["schema"] = "source_eikonal_grid_graph_v1"
        target.attrs["source_h5"] = str(source)
        target.attrs["splits"] = ",".join(sorted(selected_splits))
        target.attrs["families"] = ",".join(sorted(selected_families))
        target.attrs["group_count"] = len(tasks)
        target.attrs["record_count"] = rows
        target.create_dataset("x_m", data=x_m)
        target.create_dataset("z_m", data=z_m)
        target.create_dataset("source_index", shape=(rows,), dtype=np.int64)
        target.create_dataset("sample_id", shape=(rows,), dtype="S128")
        target.create_dataset("group_id", shape=(rows,), dtype="S128")
        target.create_dataset("medium_type", shape=(rows,), dtype="S32")
        target.create_dataset("split", shape=(rows,), dtype="S32")
        travel_dataset = target.create_dataset(
            "travel_time_s",
            shape=(rows, len(z_m), len(x_m)),
            dtype=np.float32,
            chunks=(1, len(z_m), len(x_m)),
            compression="lzf",
        )
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(str(source),),
        ) as executor:
            for indices, travel in executor.map(_solve_group, tasks, chunksize=1):
                count = len(indices)
                selection = slice(row, row + count)
                index_array = np.asarray(indices, dtype=np.int64)
                target["source_index"][selection] = index_array
                target["sample_id"][selection] = np.char.encode(sample_id[index_array])
                target["group_id"][selection] = np.char.encode(group_id[index_array])
                target["medium_type"][selection] = np.char.encode(medium_type[index_array])
                target["split"][selection] = np.char.encode(split[index_array])
                travel_dataset[selection] = travel
                digest.update(index_array.tobytes())
                digest.update(np.asarray(travel, dtype=np.float32).tobytes())
                row += count
        target.attrs["content_sha256"] = digest.hexdigest()
        target.attrs["elapsed_seconds"] = time.monotonic() - started
        target.flush()
    if row != rows:
        raise RuntimeError("travel cache wrote an unexpected record count")
    _atomic_replace(temporary, output)
    report = {
        "status": "complete",
        "output": str(output),
        "records": rows,
        "groups": len(tasks),
        "workers": int(args.workers),
        "elapsed_seconds": time.monotonic() - started,
        "content_sha256": digest.hexdigest(),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
