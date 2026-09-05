#!/usr/bin/env python
"""Fit deterministic train-only physical scales for grouped V2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v2.data.cache import select_dense_times


def robust_scales(velocity, pressure, *, pressure_percentile=99.9):
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    pressure = np.abs(np.asarray(pressure, dtype=np.float64).reshape(-1))
    pressure = pressure[pressure > 0]
    if velocity.size == 0 or pressure.size == 0:
        raise ValueError("nonempty velocity and nonzero pressure samples are required")
    center = float(np.median(velocity))
    q25, q75 = np.percentile(velocity, [25, 75])
    velocity_scale = float(max(q75 - q25, np.std(velocity), 1.0))
    pressure_scale = float(np.percentile(pressure, pressure_percentile))
    return center, velocity_scale, max(pressure_scale, np.finfo(np.float32).tiny)


def manifest_digest(h5, indices):
    digest = hashlib.sha256()
    for name in ("sample_id", "group_id", "split", "sample_sha256"):
        if name in h5:
            for value in h5[name][indices]:
                digest.update(value if isinstance(value, bytes) else str(value).encode())
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pressure-percentile", type=float, default=99.9)
    parser.add_argument("--per-record-samples", type=int, default=2048)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(20260717)
    velocity_samples, pressure_samples = [], []
    with h5py.File(args.dataset, "r", swmr=True) as h5:
        splits = h5["split"].asstr()[:]
        indices = np.flatnonzero(splits == "train")
        if indices.size == 0:
            raise ValueError("train split is empty")
        time_s = np.asarray(h5["time_s"], dtype=np.float64)
        digest = manifest_digest(h5, indices)
        for index in indices:
            velocity_samples.append(np.asarray(h5["velocity_mps"][index, ::8, ::8], np.float32).reshape(-1))
            frames = select_dense_times(time_s, float(h5["source_t0_s"][index]), 16)
            values = np.abs(np.asarray(h5["wavefield"][index, frames, ::4, ::4], np.float32).reshape(-1))
            values = values[values > 0]
            if values.size > args.per_record_samples:
                values = values[rng.choice(values.size, args.per_record_samples, replace=False)]
            pressure_samples.append(values)
    center, velocity_scale, pressure_scale = robust_scales(
        np.concatenate(velocity_samples), np.concatenate(pressure_samples),
        pressure_percentile=args.pressure_percentile,
    )
    payload = {
        "velocity_center_mps": center, "velocity_scale_mps": velocity_scale,
        "pressure_scale_pa": pressure_scale, "source_scales": [2000.0, 2000.0, 50.0, 1.2, 1.0],
        "train_manifest_sha256": digest, "pressure_percentile": args.pressure_percentile,
        "split": "train", "record_count": int(indices.size), "algorithm": "deterministic_record_reservoir_v1",
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf8"); os.replace(temporary, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
