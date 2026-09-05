#!/usr/bin/env python3
"""Train-only PA-CORA P1 window energy, error, and loss-scale audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


FAMILIES = ("uniform", "layered", "marmousi")
TEMPORAL_THIRDS = ((0, 19), (19, 38), (38, 56))
FFT_BANDS = ((0, 5), (5, 12), (12, 29))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial.{os.getpid()}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def _f0_bin(value: float) -> str:
    if value < 15.0:
        return "low_lt15"
    if value < 22.0:
        return "middle_15to22"
    return "high_gte22"


def _region_masks(height: int, width: int, margin: int = 20) -> dict[str, np.ndarray]:
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:margin] = True
    boundary[-margin:] = True
    boundary[:, :margin] = True
    boundary[:, -margin:] = True
    top = np.zeros_like(boundary)
    top[:margin] = True
    return {"boundary20": boundary, "interior": ~boundary, "top20": top}


def _quantiles(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    target_norm = np.asarray([row["target_norm"] for row in rows], np.float64)
    anchor_rel = np.asarray([row["anchor_relative_l2"] for row in rows], np.float64)
    result = {
        "count": len(rows),
        "target_norm": _quantiles(target_norm),
        "inverse_target_norm": _quantiles(1.0 / np.maximum(target_norm, 1.0e-16)),
        "anchor_relative_l2": _quantiles(anchor_rel),
        "near_zero_fraction_vs_own_slot0": float(
            np.mean([row["target_norm_ratio_to_slot0"] < 0.10 for row in rows])
        ),
        "temporal_target_energy_share": [
            float(np.mean([row["temporal_target_energy_share"][band] for row in rows]))
            for band in range(3)
        ],
        "frequency_target_energy_share_sampled": [
            float(np.mean([row["frequency_target_energy_share_sampled"][band] for row in rows]))
            for band in range(3)
        ],
        "frequency_anchor_relative_l2_sampled": [
            float(np.mean([row["frequency_anchor_relative_l2_sampled"][band] for row in rows]))
            for band in range(3)
        ],
        "region_anchor_relative_l2": {
            region: float(np.mean([row["region_anchor_relative_l2"][region] for row in rows]))
            for region in ("boundary20", "interior", "top20")
        },
        "spatial_gradient_anchor_relative_l2": float(
            np.mean([row["spatial_gradient_anchor_relative_l2"] for row in rows])
        ),
    }
    return result


def analyze_slot(cache_path: Path, manifest: dict, slot: int) -> list[dict]:
    rows = []
    with h5py.File(cache_path, "r", swmr=True) as cache:
        if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
            raise RuntimeError(f"unexpected cache schema: {cache_path}")
        if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
            raise RuntimeError(f"cache/manifest drift: {cache_path}")
        count, _, height, width = cache["target"].shape
        masks = _region_masks(height, width)
        for index, metadata in enumerate(manifest["records"]):
            target = cache["target"][index, 8:].astype(np.float32)
            anchor = cache["base_seq"][index, 8:].astype(np.float32)
            error = anchor - target
            target_energy = float(np.square(target, dtype=np.float64).sum())
            error_energy = float(np.square(error, dtype=np.float64).sum())
            temporal_energy = np.asarray(
                [
                    np.square(target[start:stop], dtype=np.float64).sum()
                    for start, stop in TEMPORAL_THIRDS
                ],
                dtype=np.float64,
            )
            sampled_target = target[:, ::4, ::4]
            sampled_error = error[:, ::4, ::4]
            target_fft = np.fft.rfft(sampled_target, axis=0)
            error_fft = np.fft.rfft(sampled_error, axis=0)
            frequency_target = np.asarray(
                [np.square(np.abs(target_fft[start:stop]), dtype=np.float64).sum() for start, stop in FFT_BANDS]
            )
            frequency_error = np.asarray(
                [np.square(np.abs(error_fft[start:stop]), dtype=np.float64).sum() for start, stop in FFT_BANDS]
            )
            region_relative = {}
            for name, mask in masks.items():
                region_target = float(np.square(target[:, mask], dtype=np.float64).sum())
                region_error = float(np.square(error[:, mask], dtype=np.float64).sum())
                region_relative[name] = float(
                    np.sqrt(region_error / max(region_target, 1.0e-30))
                )
            target_dx = np.diff(target, axis=2)
            target_dz = np.diff(target, axis=1)
            error_dx = np.diff(error, axis=2)
            error_dz = np.diff(error, axis=1)
            gradient_target_energy = float(
                np.square(target_dx, dtype=np.float64).sum()
                + np.square(target_dz, dtype=np.float64).sum()
            )
            gradient_error_energy = float(
                np.square(error_dx, dtype=np.float64).sum()
                + np.square(error_dz, dtype=np.float64).sum()
            )
            rows.append(
                {
                    "slot": slot,
                    "record_index": index,
                    "source_index": int(metadata["source_index"]),
                    "sample_id": str(metadata.get("source_sample_id", metadata["sample_id"])),
                    "family": str(metadata["family"]),
                    "source_f0_hz": float(metadata["source_f0_hz"]),
                    "f0_bin": _f0_bin(float(metadata["source_f0_hz"])),
                    "window_start": int(metadata["window_start"]),
                    "target_norm": float(np.sqrt(max(target_energy, 0.0))),
                    "anchor_relative_l2": float(
                        np.sqrt(error_energy / max(target_energy, 1.0e-30))
                    ),
                    "temporal_target_energy_share": (
                        temporal_energy / max(temporal_energy.sum(), 1.0e-30)
                    ).tolist(),
                    "frequency_target_energy_share_sampled": (
                        frequency_target / max(frequency_target.sum(), 1.0e-30)
                    ).tolist(),
                    "frequency_anchor_relative_l2_sampled": (
                        np.sqrt(
                            frequency_error
                            / np.maximum(
                                np.maximum(frequency_target, 0.005 * frequency_target.sum()),
                                1.0e-30,
                            )
                        )
                    ).tolist(),
                    "region_anchor_relative_l2": region_relative,
                    "spatial_gradient_anchor_relative_l2": float(
                        np.sqrt(
                            gradient_error_energy
                            / max(gradient_target_energy, 1.0e-30)
                        )
                    ),
                }
            )
            if index % 40 == 0:
                print(json.dumps({"event": "slot_progress", "slot": slot, "record": index, "of": count}), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.manifest) != 4 or len(args.cache) != 4:
        raise ValueError("exactly four aligned slots are required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite analysis: {args.output}")
    manifests = [json.loads(path.read_text()) for path in args.manifest]
    all_rows = []
    for slot, (cache, manifest) in enumerate(zip(args.cache, manifests)):
        all_rows.extend(analyze_slot(cache, manifest, slot))
    slot0_norm = {
        row["source_index"]: row["target_norm"] for row in all_rows if row["slot"] == 0
    }
    for row in all_rows:
        row["target_norm_ratio_to_slot0"] = row["target_norm"] / max(
            slot0_norm[row["source_index"]], 1.0e-16
        )
    payload = {
        "schema": "pa_cora_p1_window_statistics_v1",
        "scope": "train_only_diagnostic",
        "bindings": {
            "script_sha256": _sha256(Path(__file__)),
            "manifests": {str(path): _sha256(path) for path in args.manifest},
            "caches": {str(path): _sha256(path) for path in args.cache},
        },
        "spectral_sampling": "every fourth spatial node; all 56 future frames",
        "aggregate_by_slot": {
            str(slot): _summarize([row for row in all_rows if row["slot"] == slot])
            for slot in range(4)
        },
        "by_slot_family": {
            str(slot): {
                family: _summarize(
                    [
                        row
                        for row in all_rows
                        if row["slot"] == slot and row["family"] == family
                    ]
                )
                for family in FAMILIES
            }
            for slot in range(4)
        },
        "by_slot_f0": {
            str(slot): {
                name: _summarize(
                    [row for row in all_rows if row["slot"] == slot and row["f0_bin"] == name]
                )
                for name in ("low_lt15", "middle_15to22", "high_gte22")
            }
            for slot in range(4)
        },
        "records": all_rows,
        "validation_opened": False,
        "test_id_opened": False,
    }
    _atomic_json(payload, args.output)
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
