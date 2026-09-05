#!/usr/bin/env python3
"""Freeze the 90-record development subset for the anchored-B2 snapshot-IC line.

Selects 30 validation records per family (uniform/layered/marmousi) from the
marmousi1_4m_v2 dataset by deterministic seeded sampling over sample_id-sorted
candidates.  The manifest pins sample_id + sample_sha256 + the onset-aligned
window start per record, so every later run trains/evaluates on byte-identical
inputs.  test_id is never read; anomaly is excluded per ALLOWED_MEDIUM_TYPES.

The remaining validation records (uniform 60 / layered 210 / marmousi 120)
stay available for routine training-time monitoring, disjoint from this subset.
"""
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

SOURCE_H5 = (
    "/data/jiayh/data/acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/dataset_v1.h5"
)
FAMILIES = ("uniform", "layered", "marmousi")
PER_FAMILY = 30
N_VISIBLE = 8
K_FRAMES = 64
ONSET_RMS_FRACTION = 0.05   # same rule as the smoke driver


def _atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _onset_start(wavefield: np.ndarray, time_count: int) -> int:
    rms = np.sqrt((wavefield.astype(np.float64) ** 2).mean(axis=(1, 2)))
    threshold = ONSET_RMS_FRACTION * float(rms.max())
    hits = np.flatnonzero(rms >= threshold)
    start = int(hits[0]) if hits.size else 0
    return min(start, time_count - K_FRAMES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=830921)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/b2_dev_snapshot_manifest_90rec_20260830.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {args.output}")

    with h5py.File(SOURCE_H5, "r", swmr=True) as f:
        split = f["split"][:].astype(str)
        family = f["medium_type"][:].astype(str)
        sample_ids = f["sample_id"][:].astype(str)
        group_ids = f["group_id"][:].astype(str)
        sample_hashes = f["sample_sha256"][:].astype(str)
        time_count = int(f["time_s"].shape[0])

        rng = np.random.default_rng(args.seed)
        records = []
        for name in FAMILIES:
            candidates = np.flatnonzero((split == "validation") & (family == name))
            candidates = candidates[np.argsort(sample_ids[candidates])]
            if candidates.size < PER_FAMILY:
                raise RuntimeError(f"{name}: {candidates.size} < {PER_FAMILY}")
            chosen = sorted(
                int(v) for v in rng.choice(candidates, size=PER_FAMILY, replace=False)
            )
            for i in chosen:
                wavefield = np.asarray(f["wavefield"][i], dtype=np.float32)
                records.append({
                    "source_index": i,
                    "sample_id": str(sample_ids[i]),
                    "group_id": str(group_ids[i]),
                    "sample_sha256": str(sample_hashes[i]),
                    "family": name,
                    "window_start": _onset_start(wavefield, time_count),
                })

    if len({r["sample_id"] for r in records}) != len(records):
        raise RuntimeError("duplicate sample_ids selected")

    body = {
        "schema": "b2_dev_snapshot_manifest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_h5": SOURCE_H5,
        "source_h5_byte_count": Path(SOURCE_H5).stat().st_size,
        "selection_rule": (
            "per family: validation-split candidates sorted by sample_id, "
            f"rng.choice(size={PER_FAMILY}, replace=False) with the seed below"
        ),
        "seed": args.seed,
        "families": FAMILIES,
        "per_family": PER_FAMILY,
        "n_visible": N_VISIBLE,
        "k_frames": K_FRAMES,
        "onset_rms_fraction": ONSET_RMS_FRACTION,
        "window_rule": (
            "start = first frame with RMS >= onset_rms_fraction*max frame RMS, "
            "clamped to time_count-k_frames; frames [start, start+k_frames)"
        ),
        "deployment_contract": (
            "first n_visible TRUE wavefield frames of the window are deployment "
            "inputs (IC); loss/metric computed only on frames >= n_visible"
        ),
        "records": records,
        "validation_opened": True,
        "test_id_opened": False,
    }
    body["selection_sha256"] = _canonical_sha256(body)
    _atomic_json(body, args.output)
    print(json.dumps({
        "output": str(args.output),
        "records": len(records),
        "selection_sha256": body["selection_sha256"],
        "per_family_window_start_range": {
            name: [
                min(r["window_start"] for r in records if r["family"] == name),
                max(r["window_start"] for r in records if r["family"] == name),
            ] for name in FAMILIES
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
