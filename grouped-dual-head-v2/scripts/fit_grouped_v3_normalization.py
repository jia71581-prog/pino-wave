#!/usr/bin/env python
"""Fit train-only normalization after excluding anomaly media."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grouped_ufno_mionet_v3.config import ALLOWED_MEDIUM_TYPES, V3Config
from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from grouped_ufno_mionet_v3.normalization import fit_scale_metadata


def fit_from_filtered_cache(
    cache_path: str | Path,
    *,
    manifest_digest: str,
    pressure_percentile: float,
    pressure_samples_per_record: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    velocity_samples: list[np.ndarray] = []
    pressure_samples: list[np.ndarray] = []
    source_samples: list[np.ndarray] = []
    counts = {family: 0 for family in ALLOWED_MEDIUM_TYPES}
    with h5py.File(cache_path, "r", swmr=True) as h5:
        if h5.attrs.get("schema", "") != "grouped_dual_head_v2" or h5.attrs.get("split", "") != "train":
            raise ValueError("normalization requires the complete V2 train cache")
        families = h5["medium_type"].asstr()[:]
        allowed_rows = np.flatnonzero(np.isin(families, np.asarray(ALLOWED_MEDIUM_TYPES)))
        for row in allowed_rows:
            family = str(families[row])
            counts[family] += 1
            velocity_samples.append(np.asarray(h5["velocity_mps"][row, ::8, ::8], np.float32).reshape(-1))
            source = np.asarray(h5["source_parameters"][row], np.float32)
            source_samples.append(source)
            pressure = np.abs(
                np.asarray(h5["dense_target"][row, :, ::4, ::4], np.float32).reshape(-1)
            )
            pressure = pressure[pressure > 0] / max(float(source[4]), np.finfo(np.float32).tiny)
            if pressure.size > pressure_samples_per_record:
                pressure = pressure[
                    rng.choice(pressure.size, pressure_samples_per_record, replace=False)
                ]
            pressure_samples.append(pressure)
    expected_counts = {"uniform": 420, "layered": 1120, "marmousi": 700}
    if counts != expected_counts:
        raise ValueError(f"filtered normalization census mismatch: {counts}")
    metadata = fit_scale_metadata(
        torch.from_numpy(np.concatenate(velocity_samples)),
        torch.from_numpy(np.concatenate(pressure_samples)),
        torch.from_numpy(np.stack(source_samples)),
        train_manifest_sha256=manifest_digest,
        record_count=sum(counts.values()),
        pressure_percentile=pressure_percentile,
    )
    payload = metadata.to_dict()
    payload.update(
        pressure_percentile=float(pressure_percentile),
        counts_by_family=counts,
        pressure_samples_per_record=int(pressure_samples_per_record),
        seed=int(seed),
    )
    return payload


def _write_atomic(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--pressure-percentile", type=float, default=99.9)
    parser.add_argument("--pressure-samples-per-record", type=int, default=512)
    args = parser.parse_args(argv)
    config = V3Config.from_yaml(args.config)
    manifest = build_manifest(config.data.source_h5)
    validate_expected_counts(
        manifest,
        {
            "train": config.data.expected_train_records,
            "validation": config.data.expected_validation_records,
        },
    )
    payload = fit_from_filtered_cache(
        config.data.train_cache,
        manifest_digest=manifest.digest,
        pressure_percentile=args.pressure_percentile,
        pressure_samples_per_record=args.pressure_samples_per_record,
        seed=config.train.seed,
    )
    output = Path(args.output or config.data.normalization_json)
    _write_atomic(payload, output)
    print(json.dumps(payload, sort_keys=True))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
