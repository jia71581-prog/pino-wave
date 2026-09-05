#!/usr/bin/env python3
"""Derive four truth-independent PA-CORA P1 train-window manifests."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


FRACTIONS = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def derive_multiwindow_manifests(
    base_manifest: dict,
    time_s: np.ndarray,
    *,
    terminal_start_s: float = 0.65,
) -> list[dict]:
    """Return four slot manifests without reading any wavefield values."""
    if base_manifest.get("schema") != "b2_v5_causal_stratified_manifest_v1":
        raise ValueError("base manifest is not B2-v5 causal")
    if base_manifest.get("split") != "train":
        raise ValueError("multiwindow derivation accepts train records only")
    if base_manifest.get("validation_opened") or base_manifest.get("test_id_opened"):
        raise ValueError("sealed split flag is open")
    if base_manifest.get("future_truth_opened_for_window_selection"):
        raise ValueError("base window selection used future truth")
    axis = np.asarray(time_s, dtype=np.float64)
    if axis.ndim != 1 or not np.all(np.diff(axis) > 0.0):
        raise ValueError("time_s must be one-dimensional and strictly increasing")
    frame_count = int(base_manifest["k_frames"])
    maximum_start = len(axis) - frame_count
    if maximum_start < 0:
        raise ValueError("time axis is shorter than a training window")
    terminal_start = int(np.searchsorted(axis, float(terminal_start_s), side="left"))
    terminal_start = max(0, min(terminal_start, maximum_start))
    outputs = []
    base_records = list(base_manifest["records"])
    for slot, fraction in enumerate(FRACTIONS):
        payload = copy.deepcopy(base_manifest)
        payload.pop("selection_sha256", None)
        payload["created_utc"] = datetime.now(timezone.utc).isoformat()
        payload["selection_rule"] = (
            "PA-CORA P1 deterministic multiwindow derivation from the frozen V9 "
            "fit groups; no wavefield or energy statistic used"
        )
        payload["window_rule"] = (
            "round(base_causal_start + slot_fraction * "
            "(terminal_metadata_start - base_causal_start))"
        )
        payload["multiwindow"] = {
            "schema": "pa_cora_p1_multiwindow_v1",
            "slot": slot,
            "slot_fraction": fraction,
            "slot_count": len(FRACTIONS),
            "terminal_start_s": float(terminal_start_s),
            "terminal_start_index": terminal_start,
            "base_selection_sha256": base_manifest["selection_sha256"],
            "future_truth_used": False,
        }
        records = []
        for base in base_records:
            row = dict(base)
            base_start = int(base["window_start"])
            target_start = max(base_start, terminal_start)
            row["window_start"] = int(
                round(base_start + fraction * (target_start - base_start))
            )
            row["source_sample_id"] = str(base["sample_id"])
            row["sample_id"] = f"{base['sample_id']}__pacora_p1_w{slot}"
            row["multiwindow_slot"] = slot
            records.append(row)
        payload["records"] = records
        payload["groups_per_family"] = base_manifest["groups_per_family"]
        payload["future_truth_opened_for_window_selection"] = False
        payload["validation_opened"] = False
        payload["test_id_opened"] = False
        payload["selection_sha256"] = _canonical_sha256(payload)
        outputs.append(payload)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--terminal-start-s", type=float, default=0.65)
    args = parser.parse_args()
    base = json.loads(args.base_manifest.read_text())
    with h5py.File(base["source_h5"], "r", swmr=True) as source:
        time_s = np.asarray(source["time_s"][:], dtype=np.float64)
    manifests = derive_multiwindow_manifests(
        base, time_s, terminal_start_s=args.terminal_start_s
    )
    outputs = [
        args.output_prefix.with_name(f"{args.output_prefix.name}_slot{slot}.json")
        for slot in range(len(manifests))
    ]
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite PA-CORA P1 manifests")
    for payload, path in zip(manifests, outputs):
        _atomic_json(payload, path)
    print(
        json.dumps(
            {
                "schema": "pa_cora_p1_multiwindow_manifest_set_v1",
                "base_manifest": str(args.base_manifest),
                "base_manifest_sha256": _sha256(args.base_manifest),
                "outputs": [
                    {
                        "path": str(path),
                        "sha256": _sha256(path),
                        "selection_sha256": payload["selection_sha256"],
                        "start_range": [
                            min(row["window_start"] for row in payload["records"]),
                            max(row["window_start"] for row in payload["records"]),
                        ],
                    }
                    for path, payload in zip(outputs, manifests)
                ],
                "future_truth_used": False,
                "validation_opened": False,
                "test_id_opened": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
