#!/usr/bin/env python3
"""Audit held-out source-position distance from same-family training support."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def source_position_distances(
    train_xy: np.ndarray,
    query_xy: np.ndarray,
    *,
    domain_x_m: float,
    domain_z_m: float,
) -> dict[str, np.ndarray]:
    """Return nearest physical and domain-normalized distances for [x,z]."""

    train = np.asarray(train_xy, dtype=np.float64)
    query = np.asarray(query_xy, dtype=np.float64)
    if (
        train.ndim != 2
        or query.ndim != 2
        or train.shape[1:] != (2,)
        or query.shape[1:] != (2,)
        or len(train) == 0
        or len(query) == 0
        or not np.isfinite(train).all()
        or not np.isfinite(query).all()
    ):
        raise ValueError("source arrays must be finite nonempty [record,2]")
    if domain_x_m <= 0 or domain_z_m <= 0:
        raise ValueError("domain extents must be positive")
    diagonal = math.hypot(domain_x_m, domain_z_m)
    delta = query[:, None, :] - train[None, :, :]
    spatial = np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2) / diagonal
    return {
        "nearest_spatial_distance_m": spatial.min(axis=1) * diagonal,
        "nearest_spatial_distance_domain_diagonal": spatial.min(axis=1),
    }


def audit(source_h5: Path, *, evaluation_split: str, medium: str) -> dict[str, Any]:
    """Read only index/source-coordinate fields and return a source-support audit."""

    accessed = {
        "completed_mask",
        "group_id",
        "medium_type",
        "qc_status",
        "sample_id",
        "source_x_m",
        "source_z_m",
        "split",
        "x_m",
        "z_m",
    }
    with h5py.File(source_h5, "r", swmr=True) as handle:
        text = lambda name: np.asarray(handle[name].asstr()[:], dtype=str)
        split = text("split")
        family = text("medium_type")
        sample_id = text("sample_id")
        group_id = text("group_id")
        qc_status = np.char.lower(text("qc_status"))
        completed = np.asarray(handle["completed_mask"][:], dtype=bool)
        source_xy = np.column_stack(
            (
                np.asarray(handle["source_x_m"][:], dtype=np.float64),
                np.asarray(handle["source_z_m"][:], dtype=np.float64),
            )
        )
        eligible = completed & np.isin(
            qc_status, ["pass", "passed", "ok", "complete", "completed"]
        )
        family_mask = np.char.lower(family.astype(str)) == medium.lower()
        train_mask = eligible & family_mask & (split == "train")
        query_mask = eligible & family_mask & (split == evaluation_split)
        if not np.any(train_mask) or not np.any(query_mask):
            raise ValueError("train or evaluation source support is empty")
        domain_x_m = float(handle["x_m"][-1] - handle["x_m"][0])
        domain_z_m = float(handle["z_m"][-1] - handle["z_m"][0])
        distances = source_position_distances(
            source_xy[train_mask],
            source_xy[query_mask],
            domain_x_m=domain_x_m,
            domain_z_m=domain_z_m,
        )
        query_indices = np.flatnonzero(query_mask)

    rows = []
    for local_index, h5_index in enumerate(query_indices):
        rows.append(
            {
                "h5_index": int(h5_index),
                "sample_id": sample_id[h5_index],
                "group_id": group_id[h5_index],
                "source_x_m": float(source_xy[h5_index, 0]),
                "source_z_m": float(source_xy[h5_index, 1]),
                **{name: float(values[local_index]) for name, values in distances.items()},
            }
        )
    return {
        "schema": "source_position_support_audit_v1",
        "source_h5": str(source_h5.resolve()),
        "evaluation_split": evaluation_split,
        "medium": medium,
        "train_source_count": int(train_mask.sum()),
        "evaluation_source_count": int(query_mask.sum()),
        "distance_reference": "same-family training source positions only; source frequency is intentionally excluded",
        "normalization": {
            "spatial": "Euclidean x-z distance in metres and divided by domain diagonal"
        },
        "accessed_datasets": sorted(accessed),
        "target_wavefield_access": False,
        "records": rows,
    }


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--evaluation-split", default="validation")
    parser.add_argument("--medium", default="marmousi")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = audit(
        args.source_h5,
        evaluation_split=args.evaluation_split,
        medium=args.medium,
    )
    if args.output is not None:
        _atomic_json(payload, args.output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["audit", "source_position_distances"]
