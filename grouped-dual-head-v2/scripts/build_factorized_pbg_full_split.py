#!/usr/bin/env python3
"""Create a deterministic 80/10/10 split over the train-only Marmousi P_bg VDS."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def _text(values) -> tuple[str, ...]:
    return tuple(
        value.decode("utf8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=372)
    args = parser.parse_args()
    vds = Path(args.vds).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    with h5py.File(vds, "r", swmr=True) as handle:
        sample_ids = _text(handle["sample_id"][:])
        project_splits = set(_text(handle["split"][:]))
        families = set(_text(handle["medium_type"][:]))
    if project_splits != {"train"} or families != {"marmousi"}:
        raise ValueError(
            f"expected only project train/Marmousi, got {project_splits}/{families}"
        )
    if len(sample_ids) != 700 or len(set(sample_ids)) != 700:
        raise ValueError("full P_bg split requires exactly 700 unique records")
    rng = np.random.default_rng(int(args.seed))
    permutation = rng.permutation(len(sample_ids)).tolist()
    train = sorted(int(value) for value in permutation[:560])
    val = sorted(int(value) for value in permutation[560:630])
    test = sorted(int(value) for value in permutation[630:])
    payload = {
        "schema": "factorized_pbg_full_train_only_split_v1",
        "vds": str(vds),
        "seed": int(args.seed),
        "strategy": "fixed_seed_random_complete_records",
        "project_data_scope": "train_only",
        "train": train,
        "val": val,
        "test": test,
        "sample_ids": {
            "train": [sample_ids[index] for index in train],
            "val": [sample_ids[index] for index in val],
            "test": [sample_ids[index] for index in test],
        },
        "sample_counts": {
            "train": 560,
            "val": 70,
            "test": 70,
            "total_used": 700,
        },
        "validation_opened": False,
        "test_id_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps(payload["sample_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
