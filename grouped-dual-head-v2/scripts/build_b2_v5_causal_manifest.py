#!/usr/bin/env python3
"""Build a stratified train manifest with source-only causal windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.b2_v5_components import causal_window_start

SOURCE_H5 = Path(
    "/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/"
    "dataset_v1.h5"
)
FAMILIES = ("uniform", "layered", "marmousi")
K_FRAMES = 64
N_VISIBLE = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _quantile_bins(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 1 or np.all(values == values[0]):
        return np.zeros(len(values), dtype=np.int64)
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, count + 1)[1:-1]))
    return np.digitize(values, edges, right=False).astype(np.int64)


def stratified_unique_group_sample(
    candidates: np.ndarray,
    group_ids: np.ndarray,
    features: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Round-robin source/medium strata while enforcing one record per group."""
    candidates = np.asarray(candidates, dtype=np.int64)
    values = np.asarray(features, dtype=np.float64)
    if values.shape != (len(candidates), 4):
        raise ValueError("features must be [N,4] for f0/x/z/vmin")
    bins = np.stack(
        (
            _quantile_bins(values[:, 0], 3),
            _quantile_bins(values[:, 1], 3),
            _quantile_bins(values[:, 2], 2),
            _quantile_bins(values[:, 3], 3),
        ),
        axis=1,
    )
    strata: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, key in zip(candidates, bins):
        strata[tuple(int(value) for value in key)].append(int(index))
    for key, records in strata.items():
        order = rng.permutation(len(records))
        strata[key] = [records[int(position)] for position in order]
    selected: list[int] = []
    selected_groups: set[str] = set()
    keys = sorted(strata)
    while len(selected) < count:
        progress = False
        for key in keys:
            queue = strata[key]
            while queue and str(group_ids[queue[0]]) in selected_groups:
                queue.pop(0)
            if not queue:
                continue
            index = queue.pop(0)
            selected.append(index)
            selected_groups.add(str(group_ids[index]))
            progress = True
            if len(selected) == count:
                break
        if not progress:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"only selected {len(selected)} unique groups for requested {count}"
        )
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--groups-per-family", type=int, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lead-cycles", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")
    exclusions = [json.loads(path.read_text()) for path in args.exclude_manifest]
    excluded_ids = {
        row["sample_id"] for manifest in exclusions for row in manifest["records"]
    }
    excluded_groups = {
        row["group_id"] for manifest in exclusions for row in manifest["records"]
    }
    with h5py.File(SOURCE_H5, "r", swmr=True) as source:
        split = source["split"][:].astype(str)
        family = source["medium_type"][:].astype(str)
        sample_ids = source["sample_id"][:].astype(str)
        group_ids = source["group_id"][:].astype(str)
        sample_hashes = source["sample_sha256"][:].astype(str)
        completed = source["completed_mask"][:].astype(bool)
        qc_status = source["qc_status"][:].astype(str)
        f0 = np.asarray(source["source_f0_hz"][:], dtype=np.float64)
        t0 = np.asarray(source["source_t0_s"][:], dtype=np.float64)
        sx = np.asarray(source["source_x_m"][:], dtype=np.float64)
        sz = np.asarray(source["source_z_m"][:], dtype=np.float64)
        vmin = np.asarray(source["vmin_mps"][:], dtype=np.float64)
        time_s = np.asarray(source["time_s"][:], dtype=np.float64)
        rng = np.random.default_rng(args.seed)
        records = []
        for name in FAMILIES:
            candidates = np.flatnonzero(
                (split == "train")
                & (family == name)
                & completed
                & (qc_status == "passed")
                & ~np.isin(sample_ids, list(excluded_ids))
                & ~np.isin(group_ids, list(excluded_groups))
            )
            chosen = stratified_unique_group_sample(
                candidates,
                group_ids,
                np.stack((f0[candidates], sx[candidates], sz[candidates], vmin[candidates]), axis=1),
                count=args.groups_per_family,
                rng=rng,
            )
            for index in chosen:
                records.append({
                    "source_index": int(index),
                    "sample_id": str(sample_ids[index]),
                    "group_id": str(group_ids[index]),
                    "sample_sha256": str(sample_hashes[index]),
                    "family": name,
                    "source_f0_hz": float(f0[index]),
                    "source_t0_s": float(t0[index]),
                    "window_start": causal_window_start(
                        time_s,
                        source_t0_s=float(t0[index]),
                        source_f0_hz=float(f0[index]),
                        k_frames=K_FRAMES,
                        lead_cycles=args.lead_cycles,
                    ),
                })
    ids = [row["sample_id"] for row in records]
    groups = [row["group_id"] for row in records]
    if len(ids) != len(set(ids)) or len(groups) != len(set(groups)):
        raise RuntimeError("manifest records are not sample/group disjoint")
    body = {
        "schema": "b2_v5_causal_stratified_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5": str(SOURCE_H5),
        "source_h5_byte_count": SOURCE_H5.stat().st_size,
        "split": "train",
        "families": FAMILIES,
        "groups_per_family": args.groups_per_family,
        "seed": args.seed,
        "selection_rule": (
            "round-robin quantile strata over source_f0/source_x/source_z/vmin, "
            "with one record per unique medium group"
        ),
        "n_visible": N_VISIBLE,
        "k_frames": K_FRAMES,
        "window_rule": "searchsorted(time_s, source_t0_s - lead_cycles/source_f0_hz)",
        "lead_cycles": args.lead_cycles,
        "future_truth_opened_for_window_selection": False,
        "excluded_manifests": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in args.exclude_manifest
        ],
        "records": records,
        "validation_opened": False,
        "test_id_opened": False,
    }
    body["selection_sha256"] = _canonical_sha256(body)
    _atomic_json(body, args.output)
    print(json.dumps({
        "output": str(args.output),
        "records": len(records),
        "unique_groups": len(set(groups)),
        "selection_sha256": body["selection_sha256"],
        "window_start_range": [
            min(row["window_start"] for row in records),
            max(row["window_start"] for row in records),
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
