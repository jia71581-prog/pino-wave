#!/usr/bin/env python
"""Build atomic structured full-frame/query/receiver caches for dual-head V2."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v2.data.cache import select_dense_times


_H5 = None
_QUERY_COUNT = 8192
_UNIFORM_FLOOR = 0.2
_SEED = 17


def mixture_probabilities(energy, uniform_floor=0.2):
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    if not 0.2 <= uniform_floor <= 1:
        raise ValueError("uniform_floor must be in [0.2,1]")
    weighted = np.maximum(energy, 0) + np.finfo(np.float64).eps
    weighted /= weighted.sum()
    return ((1 - uniform_floor) * weighted + uniform_floor / energy.size).astype(np.float64)


def _init(path, query_count, uniform_floor, seed):
    global _H5, _QUERY_COUNT, _UNIFORM_FLOOR, _SEED
    _H5 = h5py.File(path, "r", swmr=True); _QUERY_COUNT = query_count; _UNIFORM_FLOOR = uniform_floor; _SEED = seed


def _record(index):
    h5 = _H5; assert h5 is not None
    time_s = np.asarray(h5["time_s"], np.float32); x = np.asarray(h5["x_m"], np.float32); z = np.asarray(h5["z_m"], np.float32)
    wave = np.asarray(h5["wavefield"][index], np.float32)
    dense_indices = select_dense_times(time_s, float(h5["source_t0_s"][index]), 16)
    dense = wave[dense_indices]
    receiver_x = np.rint(np.linspace(.05, .95, 8) * (wave.shape[-1] - 1)).astype(np.int64)
    receiver_z = np.full(8, int(round(.2 * (wave.shape[-2] - 1))), np.int64)
    receivers = wave[:, receiver_z, receiver_x].T
    energy = np.abs(dense).reshape(-1)
    probability = mixture_probabilities(energy, _UNIFORM_FLOOR)
    rng = np.random.default_rng(_SEED + int(index) * 1009)
    chosen = rng.choice(probability.size, _QUERY_COUNT, replace=probability.size < _QUERY_COUNT, p=probability)
    frame, remainder = np.divmod(chosen, wave.shape[-2] * wave.shape[-1]); zz, xx = np.divmod(remainder, wave.shape[-1])
    coords = np.stack((x[xx], z[zz], time_s[dense_indices[frame]]), axis=-1).astype(np.float32)
    source = np.asarray([h5[name][index] for name in ("source_x_m", "source_z_m", "source_f0_hz", "source_t0_s", "source_amplitude")], np.float32)
    return (int(index), np.asarray(h5["velocity_mps"][index], np.float32), source,
            np.asarray(h5["source_map"][index], np.float32), dense_indices, dense,
            np.stack((receiver_z, receiver_x), -1), receivers, coords,
            dense[frame, zz, xx], probability[chosen].astype(np.float32))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True); parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--workers", type=int, default=8); parser.add_argument("--query-count", type=int, default=8192)
    parser.add_argument("--uniform-floor", type=float, default=.2); parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    normalization_bytes = Path(args.normalization).read_bytes(); normalization = json.loads(normalization_bytes)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    partial = output.with_suffix(output.suffix + f".partial.{os.getpid()}")
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        indices = np.flatnonzero(h5["split"].asstr()[:] == args.split)
        if indices.size == 0: raise ValueError("requested split is empty")
        nz, nx = h5["velocity_mps"].shape[1:]; nt = len(h5["time_s"])
        text = {name: np.asarray(h5[name].asstr()[indices], dtype=h5py.string_dtype()) for name in ("sample_id", "group_id", "medium_type")}
        time_s = np.asarray(h5["time_s"], np.float32)
    with h5py.File(partial, "w", libver="latest") as cache:
        n = len(indices); cache.attrs.update(schema="grouped_dual_head_v2", split=args.split, complete=False,
            source_dataset_sha256=normalization["train_manifest_sha256"], normalization_sha256=hashlib.sha256(normalization_bytes).hexdigest())
        cache.create_dataset("velocity_mps", (n,nz,nx), "f4", chunks=(1,nz,nx), compression="lzf")
        cache.create_dataset("source_parameters", (n,5), "f4"); cache.create_dataset("source_map", (n,nz,nx), "f4", chunks=(1,nz,nx), compression="lzf")
        cache.create_dataset("dense_time_indices", (n,16), "i8"); cache.create_dataset("dense_target", (n,16,nz,nx), "f4", chunks=(1,1,nz,nx), compression="lzf")
        cache.create_dataset("receiver_zx_indices", (n,8,2), "i8"); cache.create_dataset("receiver_target", (n,8,nt), "f4", compression="lzf")
        cache.create_dataset("query_coords", (n,args.query_count,3), "f4", compression="lzf"); cache.create_dataset("query_target", (n,args.query_count), "f4", compression="lzf"); cache.create_dataset("sample_probability", (n,args.query_count), "f4", compression="lzf")
        cache.create_dataset("time_s", data=time_s); [cache.create_dataset(name, data=value) for name,value in text.items()]
        positions = {int(index): row for row,index in enumerate(indices)}
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(args.dataset,args.query_count,args.uniform_floor,args.seed)) as pool:
            for count, values in enumerate(pool.map(_record, indices, chunksize=1), 1):
                index,*arrays=values; row=positions[index]
                for name,value in zip(("velocity_mps","source_parameters","source_map","dense_time_indices","dense_target","receiver_zx_indices","receiver_target","query_coords","query_target","sample_probability"),arrays): cache[name][row]=value
                if count % 25 == 0: cache.flush(); print(f"cached={count}/{n}", flush=True)
        cache.attrs["complete"] = True; cache.flush()
    os.replace(partial, output); print(output)


if __name__ == "__main__":
    main()
