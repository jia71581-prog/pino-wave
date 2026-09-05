#!/usr/bin/env python3
"""Select a qualitative multi-source medium using input-only HDF5 fields.

The selector deliberately never opens the target ``wavefield`` dataset.  It
ranks medium groups by velocity-map first-difference energy and reports the
lower median group, so qualitative examples can be locked before predictions
or target wavefields are inspected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ALLOWED_DATASETS = {
    "completed_mask",
    "group_id",
    "medium_type",
    "qc_status",
    "sample_id",
    "source_f0_hz",
    "source_x_m",
    "source_z_m",
    "split",
    "velocity_mps",
}


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]
    )


def _velocity_roughness(velocity: np.ndarray) -> float:
    velocity64 = np.asarray(velocity, dtype=np.float64)
    return float(
        np.mean(np.diff(velocity64, axis=0) ** 2)
        + np.mean(np.diff(velocity64, axis=1) ** 2)
    )


def select_group(source_h5: Path, split_name: str, medium_name: str) -> dict[str, Any]:
    accessed: set[str] = set()

    def read(handle: h5py.File, name: str) -> np.ndarray:
        if name not in ALLOWED_DATASETS:
            raise RuntimeError(f"input-only contract forbids dataset: {name}")
        accessed.add(name)
        return handle[name][:]

    with h5py.File(source_h5, "r") as handle:
        split = _decode(read(handle, "split"))
        medium = _decode(read(handle, "medium_type"))
        group_id = _decode(read(handle, "group_id"))
        sample_id = _decode(read(handle, "sample_id"))
        completed = np.asarray(read(handle, "completed_mask"), dtype=bool)
        qc_status = np.char.lower(_decode(read(handle, "qc_status")).astype(str))
        qc_ok = np.isin(qc_status, ["pass", "passed", "ok", "complete", "completed"])
        target = (
            (split == split_name)
            & (np.char.lower(medium.astype(str)) == medium_name.lower())
            & completed
            & qc_ok
        )
        if not np.any(target):
            raise ValueError(f"no completed {split_name}/{medium_name} records found")

        groups: dict[str, list[int]] = defaultdict(list)
        for index in np.flatnonzero(target):
            groups[group_id[index]].append(int(index))

        ranked: list[tuple[float, str, list[int]]] = []
        velocity_dataset = handle["velocity_mps"]
        accessed.add("velocity_mps")
        for group, indices in groups.items():
            reference_velocity = velocity_dataset[indices[0]]
            if any(not np.array_equal(reference_velocity, velocity_dataset[index]) for index in indices[1:]):
                raise ValueError(f"group {group} does not share one identical velocity map")
            ranked.append((_velocity_roughness(reference_velocity), group, indices))
        ranked.sort(key=lambda row: (row[0], row[1]))

        rank_zero_based = (len(ranked) - 1) // 2
        roughness, selected_group, selected_indices = ranked[rank_zero_based]
        source_x = read(handle, "source_x_m")
        source_z = read(handle, "source_z_m")
        source_f0 = read(handle, "source_f0_hz")
        selected_indices.sort(key=lambda i: (float(source_x[i]), float(source_z[i]), sample_id[i]))
        samples = [
            {
                "h5_index": index,
                "sample_id": sample_id[index],
                "source_x_m": float(source_x[index]),
                "source_z_m": float(source_z[index]),
                "source_f0_hz": float(source_f0[index]),
            }
            for index in selected_indices
        ]

    if "wavefield" in accessed:
        raise AssertionError("target wavefield was accessed")
    return {
        "schema": "input_only_multisource_panel_selection_v1",
        "source_h5": str(source_h5),
        "split": split_name,
        "medium": medium_name,
        "selection_rule": (
            "lower median velocity first-difference energy; tie break by group_id; "
            "no target wavefield, prediction, or error access"
        ),
        "candidate_group_count": len(ranked),
        "candidate_group_size_counts": dict(sorted(Counter(len(row[2]) for row in ranked).items())),
        "selected_rank_one_based": rank_zero_based + 1,
        "selected_group_id": selected_group,
        "velocity_gradient_energy": roughness,
        "samples": samples,
        "accessed_datasets": sorted(accessed),
        "forbidden_dataset_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--medium", default="marmousi")
    args = parser.parse_args()
    print(json.dumps(select_group(args.source_h5, args.split, args.medium), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
