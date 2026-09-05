#!/usr/bin/env python3
"""Freeze a group-disjoint train-only panel for B2 generalization diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_H5 = Path(
    "/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/"
    "dataset_v1.h5"
)
DEV_MANIFEST = ROOT / "results/b2_dev_snapshot_manifest_90rec_20260830.json"
FAMILIES = ("uniform", "layered", "marmousi")
N_VISIBLE = 8
K_FRAMES = 64
ONSET_RMS_FRACTION = 0.05


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _onset_start(wavefield: np.ndarray, time_count: int) -> int:
    rms = np.sqrt(np.square(wavefield.astype(np.float64)).mean(axis=(1, 2)))
    hits = np.flatnonzero(rms >= ONSET_RMS_FRACTION * float(rms.max()))
    start = int(hits[0]) if hits.size else 0
    return min(start, time_count - K_FRAMES)


def select_group_disjoint(
    candidates: np.ndarray,
    sample_ids: np.ndarray,
    group_ids: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Choose one sample per group using deterministic seeded draws."""
    ordered = candidates[np.argsort(sample_ids[candidates])]
    by_group: dict[str, list[int]] = {}
    for index in ordered:
        by_group.setdefault(str(group_ids[index]), []).append(int(index))
    groups = np.asarray(sorted(by_group), dtype=object)
    if groups.size < count:
        raise RuntimeError(f"only {groups.size} unique groups for requested {count}")
    selected_groups = sorted(str(value) for value in rng.choice(
        groups, size=count, replace=False
    ))
    chosen = []
    for group in selected_groups:
        members = by_group[group]
        chosen.append(int(members[int(rng.integers(0, len(members)))]))
    return sorted(chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=831104)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/b2_train_generalization_manifest_30rec_20260831.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")

    exclusion_paths = []
    for path in (DEV_MANIFEST, *args.exclude_manifest):
        resolved = path.resolve()
        if resolved not in [value.resolve() for value in exclusion_paths]:
            exclusion_paths.append(path)
    exclusions = [json.loads(path.read_text()) for path in exclusion_paths]
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
        time_count = int(source["time_s"].shape[0])
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
            chosen = select_group_disjoint(
                candidates,
                sample_ids,
                group_ids,
                count=args.per_family,
                rng=rng,
            )
            for index in chosen:
                records.append({
                    "source_index": index,
                    "sample_id": str(sample_ids[index]),
                    "group_id": str(group_ids[index]),
                    "sample_sha256": str(sample_hashes[index]),
                    "family": name,
                    "window_start": _onset_start(
                        np.asarray(source["wavefield"][index], dtype=np.float32),
                        time_count,
                    ),
                })

    ids = [row["sample_id"] for row in records]
    groups = [row["group_id"] for row in records]
    if len(set(ids)) != len(records) or len(set(groups)) != len(records):
        raise RuntimeError("train diagnostic selection is not record/group disjoint")
    if set(ids) & excluded_ids or set(groups) & excluded_groups:
        raise RuntimeError("train diagnostic overlaps an excluded manifest")

    body = {
        "schema": "b2_train_generalization_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5": str(SOURCE_H5),
        "source_h5_byte_count": SOURCE_H5.stat().st_size,
        "split": "train",
        "families": FAMILIES,
        "per_family": args.per_family,
        "seed": args.seed,
        "selection_rule": (
            "per family: completed qc-pass train records grouped by group_id; "
            "sample unique groups without replacement, then one record per group"
        ),
        "n_visible": N_VISIBLE,
        "k_frames": K_FRAMES,
        "onset_rms_fraction": ONSET_RMS_FRACTION,
        "window_rule": (
            "train-truth-only diagnostic selection: first frame with RMS >= "
            "onset_rms_fraction*max frame RMS, clamped to time_count-k_frames"
        ),
        "deployment_contract": (
            "first n_visible true frames plus static medium/source and solve-free "
            "warp anchor; metrics use frames >= n_visible"
        ),
        "excluded_manifests": [
            {"path": str(path), "sha256": _file_sha256(path)}
            for path in exclusion_paths
        ],
        "records": records,
        "train_truth_opened_for_window_selection": True,
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
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
