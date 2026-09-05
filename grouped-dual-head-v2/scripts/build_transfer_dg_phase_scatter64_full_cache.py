#!/usr/bin/env python3
"""Build one residual-only cache shard for full-train phase/scatter-64 fitting."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for value in (str(ROOT), str(ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from scripts.build_transfer_dg_phase_scatter64_pilot_cache import (  # noqa: E402
    PHYSICAL_SCALE,
    atomic_json,
    enriched_medium,
    parent_coefficients,
    sha256,
)
from scripts.train_transfer_dg_wfp_e1 import CacheCollection, FeatureBuilder  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection  # noqa: E402
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    if args.output.exists() or summary_path.exists():
        raise FileExistsError(args.output)
    if args.worker_count != 4 or not 0 <= args.worker_index < 4:
        raise RuntimeError("full cache requires four valid workers")
    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if manifest.get("split") != "train" or manifest.get("validation_opened") or manifest.get("test_id_opened"):
        raise RuntimeError("full correction cache is train-only")
    if sha256(args.manifest) != bindings["train_manifest_sha256"]:
        raise RuntimeError("train manifest binding drift")
    if sha256(args.parent_checkpoint) != bindings["parent_checkpoint_sha256"]:
        raise RuntimeError("parent checkpoint binding drift")
    if sha256(Path(__file__)) != bindings["cache_builder_sha256"]:
        raise RuntimeError("full cache builder binding drift")
    rows = manifest["records"][args.worker_index :: args.worker_count]
    if args.max_records:
        rows = rows[:args.max_records]

    collection = CacheCollection(args.base_cache, manifest, expected_count=2800)
    travel = TravelCollection(args.travel, expected_count=2800)
    positions = {row[2]: index for index, row in enumerate(collection.records)}
    builder = FeatureBuilder(collection)
    checkpoint = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    parent_model = BackgroundFrequencyOperator(
        medium_channels=12, source_channels=5, width=64, rank=32, depth=6,
        arm="wfp", radii=(1, 2, 3, 4, 5, 6),
    ).to(device)
    parent_model.load_state_dict(checkpoint["model_state"])
    parent_model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    source_path = Path(manifest["source_h5"])
    started = time.time()
    baseline_values = []
    maximum_normalized = 0.0
    try:
        with h5py.File(source_path, "r", swmr=True) as source, h5py.File(partial, "x") as output:
            output.attrs["schema"] = "transfer_dg_phase_scatter64_full_cache_v1"
            output.attrs["status"] = "building"
            output.attrs["worker_index"] = args.worker_index
            output.attrs["worker_count"] = args.worker_count
            output.attrs["frequency_count"] = 64
            output.attrs["physical_scale"] = PHYSICAL_SCALE
            output.attrs["parent_checkpoint_sha256"] = bindings["parent_checkpoint_sha256"]
            output.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
            output.attrs["validation_opened"] = False
            output.attrs["test_id_opened"] = False
            text_dtype = h5py.string_dtype("utf-8")
            for name in ("sample_id", "family"):
                output.create_dataset(name, shape=(len(rows),), dtype=text_dtype)
            output.create_dataset("source_index", shape=(len(rows),), dtype=np.int64)
            output.create_dataset("frequency_hz", data=np.fft.rfftfreq(401, d=0.0025)[:64])
            medium_ds = output.create_dataset(
                "medium", shape=(len(rows), 16, 221, 241), dtype=np.float16,
                chunks=(1, 16, 221, 241), compression="lzf",
            )
            source_map_ds = output.create_dataset(
                "source_map", shape=(len(rows), 201, 201), dtype=np.float32,
                chunks=(1, 201, 201), compression="lzf",
            )
            wavelet_ds = output.create_dataset("source_wavelet", shape=(len(rows), 401), dtype=np.float32)
            parameters_ds = output.create_dataset("source_parameters", shape=(len(rows), 4), dtype=np.float32)
            travel_ds = output.create_dataset(
                "travel_physical_s", shape=(len(rows), 201, 201), dtype=np.float32,
                chunks=(1, 201, 201), compression="lzf",
            )
            residual_ds = output.create_dataset(
                "residual_coeff_norm", shape=(len(rows), 64, 2, 201, 201), dtype=np.float16,
                chunks=(1, 1, 2, 201, 201), compression="lzf",
            )
            target_total_ds = output.create_dataset("target_modeled_total_square", shape=(len(rows),), dtype=np.float64)
            full_total_ds = output.create_dataset("full_target_time_square_norm", shape=(len(rows),), dtype=np.float64)
            unmodeled_ds = output.create_dataset("unmodeled_time_square_norm", shape=(len(rows),), dtype=np.float64)
            for local, row in enumerate(rows):
                sample_id = str(row["sample_id"])
                index = int(row["source_index"])
                position = positions[sample_id]
                if source["split"].asstr()[index] != "train" or source["sample_id"].asstr()[index] != sample_id:
                    raise RuntimeError(f"source binding mismatch: {sample_id}")
                item = builder.build(position, 0, device)
                raw = collection.raw(position, 0)
                medium, _ = enriched_medium(
                    item[0][0].cpu().numpy(), np.asarray(raw["velocity"], dtype=np.float32),
                    np.asarray(source["velocity_mps"][index], dtype=np.float32),
                )
                truth = np.asarray(source["wavefield"][index], dtype=np.float32)
                full_spectrum = np.fft.rfft(truth, axis=0, norm="ortho")
                target = np.stack((full_spectrum[:64].real, full_spectrum[:64].imag), axis=1).astype(np.float32) / PHYSICAL_SCALE
                parent = parent_coefficients(parent_model, builder, travel, position, device)
                residual = target - parent
                current_maximum = float(np.max(np.abs(residual)))
                if not np.isfinite(current_maximum) or current_maximum >= 60000.0:
                    raise FloatingPointError(f"residual exceeds fp16: {current_maximum}")
                weights = np.full((201, 1, 1), 2.0, dtype=np.float64)
                weights[0] = 1.0
                weights[-1] = 1.0
                full_total = float((np.square(np.abs(full_spectrum)).astype(np.float64) * weights).sum() / PHYSICAL_SCALE**2)
                unmodeled = float((np.square(np.abs(full_spectrum[64:])).astype(np.float64) * weights[64:]).sum() / PHYSICAL_SCALE**2)
                low_residual = np.square(residual.astype(np.float64))
                weighted_residual = float(low_residual[0].sum() + 2.0 * low_residual[1:].sum())
                baseline_values.append(math.sqrt((weighted_residual + unmodeled) / max(full_total, 1e-300)))
                maximum_normalized = max(maximum_normalized, current_maximum)
                output["sample_id"][local] = sample_id
                output["family"][local] = row["family"]
                output["source_index"][local] = index
                medium_ds[local] = medium.astype(np.float16)
                source_map_ds[local] = np.asarray(raw["source_map"], dtype=np.float32)
                wavelet_ds[local] = np.asarray(raw["source_wavelet"], dtype=np.float32)
                parameters_ds[local] = np.asarray(raw["source_parameters"], dtype=np.float32)
                travel_ds[local] = np.asarray(travel.read(sample_id)[0], dtype=np.float32)
                residual_ds[local] = residual.astype(np.float16)
                target_total_ds[local] = float(np.square(target.astype(np.float64)).sum())
                full_total_ds[local] = full_total
                unmodeled_ds[local] = unmodeled
                output.flush()
                if (local + 1) % 10 == 0 or local + 1 == len(rows):
                    print(json.dumps({
                        "event": "cache_progress", "worker": args.worker_index,
                        "record": local + 1, "of": len(rows),
                        "baseline_mean_so_far": float(np.mean(baseline_values)),
                        "elapsed_s": time.time() - started,
                    }, sort_keys=True), flush=True)
            output.attrs["maximum_normalized"] = maximum_normalized
            output.attrs["status"] = "complete"
            output.flush()
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)
        collection.close()
        travel.close()
    summary = {
        "schema": "transfer_dg_phase_scatter64_full_cache_summary_v1",
        "status": "complete", "worker_index": args.worker_index,
        "worker_count": args.worker_count, "record_count": len(rows),
        "output": str(args.output.resolve()), "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output), "baseline_mean": float(np.mean(baseline_values)),
        "maximum_normalized": maximum_normalized, "smoke": bool(args.max_records),
        "validation_opened": False, "test_id_opened": False,
        "elapsed_s": time.time() - started,
    }
    atomic_json(summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
