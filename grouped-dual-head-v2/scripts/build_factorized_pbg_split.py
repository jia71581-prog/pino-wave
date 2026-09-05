#!/usr/bin/env python3
"""Bind a Factorized-FNO split to an existing WKB train-only panel identity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py


def _text(values) -> tuple[str, ...]:
    return tuple(
        value.decode("utf8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", required=True)
    parser.add_argument("--vds", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    identity_path = Path(args.identity).resolve()
    vds_path = Path(args.vds).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    identity = json.loads(identity_path.read_text())
    with h5py.File(vds_path, "r", swmr=True) as handle:
        sample_ids = _text(handle["sample_id"][:])
        splits = set(_text(handle["split"][:]))
    if splits != {"train"}:
        raise ValueError(f"P_bg VDS must contain only project train records: {splits}")
    by_sample = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    requested = {
        "train": tuple(identity["fit_sample_ids"]),
        "val": tuple(identity["calibration_sample_ids"]),
        "test": tuple(identity["confirm_sample_ids"]),
    }
    missing = sorted(
        sample_id
        for ids in requested.values()
        for sample_id in ids
        if sample_id not in by_sample
    )
    if missing:
        raise ValueError(f"split samples missing from P_bg VDS: {missing[:3]}")
    sets = {name: set(ids) for name, ids in requested.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError("identity panels overlap")
    payload = {
        "schema": "factorized_pbg_train_only_split_v1",
        "source_identity": str(identity_path),
        "source_identity_run_digest": identity["run_digest"],
        "vds": str(vds_path),
        "project_data_scope": "train_only",
        "train": [by_sample[value] for value in requested["train"]],
        "val": [by_sample[value] for value in requested["val"]],
        "test": [by_sample[value] for value in requested["test"]],
        "sample_ids": {name: list(values) for name, values in requested.items()},
        "sample_counts": {
            "train": len(requested["train"]),
            "val": len(requested["val"]),
            "test": len(requested["test"]),
            "total_used": sum(len(values) for values in requested.values()),
        },
        "validation_opened": False,
        "test_id_opened": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.name}.partial.{os.getpid()}")
    try:
        partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps(payload["sample_counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
