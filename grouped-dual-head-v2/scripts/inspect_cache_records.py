#!/usr/bin/env python3
"""Read scalar metadata for selected records from audited cache shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def scalar_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, nargs="+", required=True)
    parser.add_argument("--sample-id", nargs="+", required=True)
    args = parser.parse_args()
    wanted = set(args.sample_id)
    rows: list[dict[str, Any]] = []
    all_keys: set[str] = set()
    for path in args.cache:
        with h5py.File(path, "r", swmr=True) as handle:
            all_keys.update(handle.keys())
            sample_ids = handle["sample_id"].asstr()[:]
            for local_index, sample_id in enumerate(sample_ids):
                if str(sample_id) not in wanted:
                    continue
                row: dict[str, Any] = {
                    "sample_id": str(sample_id),
                    "cache": str(path),
                    "local_index": int(local_index),
                }
                record_count = len(sample_ids)
                for key, dataset in handle.items():
                    if not isinstance(dataset, h5py.Dataset):
                        continue
                    if dataset.ndim == 1 and dataset.shape[0] == record_count:
                        row[key] = scalar_value(dataset[local_index])
                if "static_features" in handle:
                    static = np.asarray(handle["static_features"][local_index], dtype=np.float64)
                    row["static_features_summary"] = {
                        "shape": list(static.shape),
                        "channel_min": static.min(axis=(1, 2)).tolist(),
                        "channel_max": static.max(axis=(1, 2)).tolist(),
                        "channel_mean": static.mean(axis=(1, 2)).tolist(),
                    }
                rows.append(row)
    missing = sorted(wanted - {row["sample_id"] for row in rows})
    print(
        json.dumps(
            {
                "records": rows,
                "missing_sample_ids": missing,
                "cache_dataset_keys": sorted(all_keys),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
