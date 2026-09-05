#!/usr/bin/env python
"""Build a derived cache using Eikonal for layered media and rays otherwise."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saved_time_phase_operator_v4.hybrid_travel import straight_ray_grid_numpy


_SOURCE: h5py.File | None = None


def _decode(values: np.ndarray) -> np.ndarray:
    return np.char.decode(values)


def _init_worker(source_h5: str) -> None:
    global _SOURCE
    _SOURCE = h5py.File(source_h5, "r", swmr=True)


def _ray_group(task: tuple[tuple[int, ...], tuple[int, ...]]):
    if _SOURCE is None:
        raise RuntimeError("hybrid travel worker was not initialized")
    cache_rows, source_indices = task
    indices = np.asarray(source_indices, dtype=np.int64)
    velocity = np.asarray(_SOURCE["velocity_mps"][int(indices[0])], dtype=np.float32)
    travel = straight_ray_grid_numpy(
        velocity,
        source_x_m=np.asarray(_SOURCE["source_x_m"][indices], dtype=np.float32),
        source_z_m=np.asarray(_SOURCE["source_z_m"][indices], dtype=np.float32),
        x_m=np.asarray(_SOURCE["x_m"][:], dtype=np.float32),
        z_m=np.asarray(_SOURCE["z_m"][:], dtype=np.float32),
        samples=12,
    )
    return cache_rows, travel


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eikonal-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    args = parser.parse_args(argv)

    source_cache = Path(args.eikonal_cache).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source_cache.is_file() or int(args.workers) <= 0:
        raise ValueError("Eikonal cache must exist and workers must be positive")
    if output.exists():
        raise FileExistsError(f"hybrid cache already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_cache, "r", swmr=True) as handle:
        source_h5 = str(Path(str(handle.attrs["source_h5"])).resolve())
        base_digest = str(handle.attrs["content_sha256"])
        family = _decode(handle["medium_type"][:])
        groups = _decode(handle["group_id"][:])
        source_indices = np.asarray(handle["source_index"][:], dtype=np.int64)
        if set(family) != {"uniform", "layered", "marmousi"}:
            raise ValueError("hybrid cache requires exactly the three registered families")
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for row in np.flatnonzero(family != "layered"):
            grouped[str(groups[row])].append(int(row))
        tasks = tuple(
            (
                tuple(rows),
                tuple(int(source_indices[row]) for row in rows),
            )
            for _, rows in sorted(grouped.items())
        )

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    started = time.monotonic()
    try:
        shutil.copyfile(source_cache, temporary)
        with h5py.File(temporary, "r+") as target:
            dataset = target["travel_time_s"]
            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=_init_worker,
                initargs=(source_h5,),
            ) as executor:
                for cache_rows, travel in executor.map(_ray_group, tasks, chunksize=1):
                    for local, row in enumerate(cache_rows):
                        dataset[int(row)] = travel[local]
            digest = hashlib.sha256()
            source_rows = np.asarray(target["source_index"][:], dtype=np.int64)
            for row in range(len(source_rows)):
                digest.update(source_rows[row : row + 1].tobytes())
                digest.update(np.asarray(dataset[row], dtype=np.float32).tobytes())
            target.attrs["schema"] = "source_family_adaptive_travel_v1"
            target.attrs["base_eikonal_content_sha256"] = base_digest
            target.attrs["travel_rule"] = "layered=eikonal;uniform,marmousi=straight_ray12"
            target.attrs["content_sha256"] = digest.hexdigest()
            target.attrs["elapsed_seconds"] = time.monotonic() - started
            target.flush()
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    report = {
        "status": "complete",
        "output": str(output),
        "records": int(len(family)),
        "ray_groups": int(len(tasks)),
        "workers": int(args.workers),
        "elapsed_seconds": time.monotonic() - started,
        "content_sha256": digest.hexdigest(),
    }
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
