#!/usr/bin/env python
"""Materialize compact single-source query supervision from the LWC-84 VDS.

The cache deliberately stores only the frames and points used by grouped
operator training.  It never alters the source VDS and is atomically promoted
only after every record is written and validated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

import h5py
import numpy as np


_HANDLE: h5py.File | None = None
_FRAMES = 32
_BLOCKS = 4
_RECEIVERS = 512
_SEED = 17


def _worker_init(path: str, frames: int, blocks: int, receivers: int, seed: int) -> None:
    global _HANDLE, _FRAMES, _BLOCKS, _RECEIVERS, _SEED
    _HANDLE = h5py.File(path, "r", swmr=True)
    _FRAMES, _BLOCKS, _RECEIVERS, _SEED = frames, blocks, receivers, seed


def _sample(index: int):
    assert _HANDLE is not None
    h5 = _HANDLE
    wavefield = h5["wavefield"]
    nt, nz, nx = wavefield.shape[1:]
    time_index = np.rint(np.linspace(0, nt - 1, min(_FRAMES, nt))).astype(np.int64)
    # One indexed VDS read for exactly the saved frames required by training.
    frames = np.asarray(wavefield[index, time_index, :, :], dtype=np.float32)
    rng = np.random.default_rng(_SEED + int(index) * 1009)
    coords = np.empty((len(time_index), _BLOCKS, _RECEIVERS, 3), dtype=np.float32)
    targets = np.empty((len(time_index), _BLOCKS, _RECEIVERS), dtype=np.float32)
    x = np.asarray(h5["x_m"], dtype=np.float32)
    z = np.asarray(h5["z_m"], dtype=np.float32)
    t = np.asarray(h5["time_s"], dtype=np.float32)
    edges = np.linspace(0, nz * nx, _BLOCKS + 1, dtype=np.int64)
    for block, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        pool = np.arange(lo, max(lo + 1, hi), dtype=np.int64)
        chosen = rng.choice(pool, _RECEIVERS, replace=pool.size < _RECEIVERS)
        zz, xx = np.divmod(chosen, nx)
        for frame_pos, frame_index in enumerate(time_index):
            coords[frame_pos, block, :, 0] = x[xx]
            coords[frame_pos, block, :, 1] = z[zz]
            coords[frame_pos, block, :, 2] = t[frame_index]
            targets[frame_pos, block] = frames[frame_pos, zz, xx]
    source = np.asarray([
        h5["source_x_m"][index], h5["source_z_m"][index],
        h5["source_f0_hz"][index], h5["source_t0_s"][index],
        h5["source_amplitude"][index],
    ], dtype=np.float32)
    return (
        index,
        np.asarray(h5["velocity_mps"][index], dtype=np.float32),
        np.asarray(h5["source_map"][index], dtype=np.float32),
        source,
        coords,
        targets,
    )


def _text(values):
    return np.asarray([v.decode("utf8") if isinstance(v, bytes) else str(v) for v in values], dtype=h5py.string_dtype("utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5")
    parser.add_argument("--output", default="/home/jiayh/Data/data/processed/grouped_sparse_query_cache_train_v1.h5")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--receivers", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if min(args.workers, args.frames, args.blocks, args.receivers) <= 0:
        raise ValueError("workers, frames, blocks, and receivers must be positive")
    source = Path(args.dataset).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + f".partial.{os.getpid()}")
    with h5py.File(source, "r", swmr=True) as h5:
        split_values = np.asarray(h5["split"].asstr()[:], dtype=str)
        indices = np.flatnonzero(split_values == args.split).astype(np.int64)
        if indices.size == 0:
            raise ValueError(f"split {args.split!r} is empty")
        nz, nx = h5["velocity_mps"].shape[1:]
        metadata = {
            "sample_id": _text(h5["sample_id"][indices]),
            "group_id": _text(h5["group_id"][indices]),
            "medium_type": _text(h5["medium_type"][indices]),
            "global_index": indices,
            "x_m": np.asarray(h5["x_m"], dtype=np.float32),
            "z_m": np.asarray(h5["z_m"], dtype=np.float32),
            "time_s": np.asarray(h5["time_s"], dtype=np.float32),
        }
    started = time.monotonic()
    with h5py.File(partial, "w", libver="latest") as cache:
        n = indices.size
        cache.attrs.update({
            "schema": "grouped_sparse_query_cache_v1", "source_dataset": str(source),
            "split": args.split, "frames": args.frames, "blocks": args.blocks,
            "receivers": args.receivers, "seed": args.seed,
        })
        cache.create_dataset("velocity_mps", (n, nz, nx), dtype="f4", chunks=(1, nz, nx), compression="lzf")
        cache.create_dataset("source_map", (n, nz, nx), dtype="f4", chunks=(1, nz, nx), compression="lzf")
        cache.create_dataset("source_parameters", (n, 5), dtype="f4")
        cache.create_dataset("query_coords", (n, args.frames, args.blocks, args.receivers, 3), dtype="f4", chunks=(1, args.frames, args.blocks, args.receivers, 3), compression="lzf")
        cache.create_dataset("targets", (n, args.frames, args.blocks, args.receivers), dtype="f4", chunks=(1, args.frames, args.blocks, args.receivers), compression="lzf")
        cache.create_dataset("completed", (n,), dtype="?", data=np.zeros(n, dtype=bool))
        for key, value in metadata.items():
            cache.create_dataset(key, data=value)
        position = {int(index): row for row, index in enumerate(indices.tolist())}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init,
                                 initargs=(str(source), args.frames, args.blocks, args.receivers, args.seed)) as pool:
            for count, result in enumerate(pool.map(_sample, indices.tolist(), chunksize=1), start=1):
                index, velocity, source_map, parameters, coords, targets = result
                row = position[int(index)]
                cache["velocity_mps"][row] = velocity
                cache["source_map"][row] = source_map
                cache["source_parameters"][row] = parameters
                cache["query_coords"][row] = coords
                cache["targets"][row] = targets
                cache["completed"][row] = True
                if count % 50 == 0 or count == n:
                    cache.flush()
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(f"cached={count}/{n} rate={count / elapsed:.2f} samples/s", flush=True)
        if not bool(np.asarray(cache["completed"], dtype=bool).all()):
            raise RuntimeError("cache generation ended with incomplete records")
        cache.flush()
    os.replace(partial, output)
    print(f"complete path={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
