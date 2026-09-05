#!/usr/bin/env python3
"""Build one train-only phase/scatter-64 pilot cache shard."""
from __future__ import annotations

import argparse
import hashlib
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

from scripts.train_transfer_dg_wfp_e1 import CacheCollection, FeatureBuilder  # noqa: E402
from scripts.train_transfer_dg_wfp_e1d import TravelCollection, apply_phase  # noqa: E402
from saved_time_phase_operator_v4.exterior_cpml import (  # noqa: E402
    ExteriorCPMLContract,
    extend_velocity_to_exterior,
)
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator  # noqa: E402


FREQUENCY_COUNT = 64
PHYSICAL_SCALE = 1.0e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def enriched_medium(
    background_medium: np.ndarray,
    background_velocity: np.ndarray,
    true_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    contract = ExteriorCPMLContract()
    background_exterior = np.asarray(
        extend_velocity_to_exterior(background_velocity, contract), dtype=np.float32
    )
    true_exterior = np.asarray(
        extend_velocity_to_exterior(true_velocity, contract), dtype=np.float32
    )
    true_log = np.log(np.maximum(true_exterior, 1.0))
    true_grad_z, true_grad_x = np.gradient(true_log)
    interface_strength = np.clip(
        20.0 * np.sqrt(true_grad_x * true_grad_x + true_grad_z * true_grad_z),
        0.0,
        2.0,
    )
    contrast = (true_exterior - background_exterior) / 2500.0
    extra = np.stack(
        (contrast, 20.0 * true_grad_x, 20.0 * true_grad_z, interface_strength),
        axis=0,
    ).astype(np.float32)
    medium = np.concatenate((background_medium, extra), axis=0).astype(np.float32)
    interface_physical = interface_strength[:201, 20:221].astype(np.float32)
    if medium.shape != (16, 221, 241) or interface_physical.shape != (201, 201):
        raise RuntimeError("phase/scatter medium shape mismatch")
    return medium, interface_physical


@torch.inference_mode()
def parent_coefficients(
    model: BackgroundFrequencyOperator,
    builder: FeatureBuilder,
    travel: TravelCollection,
    position: int,
    device: torch.device,
) -> np.ndarray:
    sample_id = builder.collection.records[position][2]
    physical_np, exterior_np, _, _ = travel.read(sample_id)
    physical_travel = torch.from_numpy(physical_np).to(device)
    exterior_travel = torch.from_numpy(exterior_np).to(device)
    rows = []
    for start in range(0, FREQUENCY_COUNT, 8):
        items = [builder.build(position, frequency, device) for frequency in range(start, start + 8)]
        medium, source, scalars = (
            torch.cat([item[index] for item in items], dim=0) for index in range(3)
        )
        frequency_hz = torch.tensor(
            [float(item[5]["frequency_hz"]) for item in items], device=device
        )
        physical, auxiliary = model(medium, source, scalars)
        physical, _ = apply_phase(
            physical,
            auxiliary,
            physical_travel[None].expand(8, -1, -1),
            exterior_travel[None].expand(8, -1, -1),
            frequency_hz,
            True,
        )
        rows.append(physical.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


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
        raise RuntimeError("pilot cache requires four valid workers")
    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if manifest.get("split") != "train" or manifest.get("validation_opened") or manifest.get("test_id_opened"):
        raise RuntimeError("phase/scatter cache is restricted to train-only rows")
    if sha256(args.manifest) != bindings["pilot_manifest_sha256"]:
        raise RuntimeError("pilot manifest binding drift")
    if sha256(args.parent_checkpoint) != bindings["parent_checkpoint_sha256"]:
        raise RuntimeError("parent checkpoint binding drift")
    if sha256(Path(__file__)) != bindings["cache_builder_sha256"]:
        raise RuntimeError("cache builder binding drift")
    rows = manifest["records"][args.worker_index :: args.worker_count]
    if args.max_records:
        rows = rows[:args.max_records]
    if not rows:
        raise RuntimeError("empty pilot cache shard")

    collection = CacheCollection(args.base_cache, json.loads(
        (ROOT / "results/transfer_dg_wfp_full2800_manifest_20260902.json").read_text()
    ), expected_count=2800)
    travel = TravelCollection(args.travel, expected_count=2800)
    positions = {row[2]: index for index, row in enumerate(collection.records)}
    builder = FeatureBuilder(collection)
    checkpoint = torch.load(args.parent_checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda")
    model = BackgroundFrequencyOperator(
        medium_channels=12,
        source_channels=5,
        width=64,
        rank=32,
        depth=6,
        arm="wfp",
        radii=(1, 2, 3, 4, 5, 6),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    source_path = Path(manifest["source_h5"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f"{args.output.name}.partial.{os.getpid()}")
    started = time.time()
    maximum_normalized = 0.0
    baseline_values = []
    try:
        with h5py.File(source_path, "r", swmr=True) as source, h5py.File(partial, "x") as output:
            output.attrs["schema"] = "transfer_dg_phase_scatter64_pilot_cache_v1"
            output.attrs["status"] = "building"
            output.attrs["worker_index"] = args.worker_index
            output.attrs["worker_count"] = args.worker_count
            output.attrs["frequency_count"] = FREQUENCY_COUNT
            output.attrs["physical_scale"] = PHYSICAL_SCALE
            output.attrs["parent_checkpoint_sha256"] = bindings["parent_checkpoint_sha256"]
            output.attrs["manifest_selection_sha256"] = manifest["selection_sha256"]
            output.attrs["training_full_truth_allowed"] = True
            output.attrs["validation_opened"] = False
            output.attrs["test_id_opened"] = False
            text_dtype = h5py.string_dtype("utf-8")
            for name in ("sample_id", "family", "role"):
                output.create_dataset(name, shape=(len(rows),), dtype=text_dtype)
            output.create_dataset("source_index", shape=(len(rows),), dtype=np.int64)
            output.create_dataset("frequency_hz", data=np.fft.rfftfreq(401, d=0.0025)[:64])
            medium_ds = output.create_dataset(
                "medium", shape=(len(rows), 16, 221, 241), dtype=np.float16,
                chunks=(1, 16, 221, 241), compression="lzf",
            )
            interface_ds = output.create_dataset(
                "interface_strength", shape=(len(rows), 201, 201), dtype=np.float16,
                chunks=(1, 201, 201), compression="lzf",
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
            target_ds = output.create_dataset(
                "target_coeff_norm", shape=(len(rows), 64, 2, 201, 201), dtype=np.float16,
                chunks=(1, 1, 2, 201, 201), compression="lzf",
            )
            parent_ds = output.create_dataset(
                "parent_coeff_norm", shape=(len(rows), 64, 2, 201, 201), dtype=np.float16,
                chunks=(1, 1, 2, 201, 201), compression="lzf",
            )
            energy_ds = output.create_dataset("target_total_square", shape=(len(rows),), dtype=np.float64)
            full_time_energy_ds = output.create_dataset(
                "full_target_time_square_norm", shape=(len(rows),), dtype=np.float64
            )
            unmodeled_energy_ds = output.create_dataset(
                "unmodeled_time_square_norm", shape=(len(rows),), dtype=np.float64
            )
            for local, row in enumerate(rows):
                sample_id = str(row["sample_id"])
                index = int(row["source_index"])
                if sample_id not in positions:
                    raise RuntimeError(f"sample absent from parent cache: {sample_id}")
                position = positions[sample_id]
                if source["split"].asstr()[index] != "train" or source["sample_id"].asstr()[index] != sample_id:
                    raise RuntimeError(f"source binding mismatch: {sample_id}")
                item = builder.build(position, 0, device)
                background_medium = item[0][0].cpu().numpy()
                raw = collection.raw(position, 0)
                true_velocity = np.asarray(source["velocity_mps"][index], dtype=np.float32)
                medium, interface = enriched_medium(
                    background_medium,
                    np.asarray(raw["velocity"], dtype=np.float32),
                    true_velocity,
                )
                truth = np.asarray(source["wavefield"][index], dtype=np.float32)
                full_spectrum = np.fft.rfft(truth, axis=0, norm="ortho")
                spectrum = full_spectrum[:64]
                target = np.stack((spectrum.real, spectrum.imag), axis=1).astype(np.float32) / PHYSICAL_SCALE
                parent = parent_coefficients(model, builder, travel, position, device)
                current_maximum = max(float(np.max(np.abs(target))), float(np.max(np.abs(parent))))
                if not np.isfinite(current_maximum) or current_maximum >= 60000.0:
                    raise FloatingPointError(f"pilot coefficient exceeds fp16: {current_maximum}")
                target_total = float(np.square(target.astype(np.float64)).sum())
                frequency_weights = np.full((201, 1, 1), 2.0, dtype=np.float64)
                frequency_weights[0] = 1.0
                frequency_weights[-1] = 1.0
                full_time_energy_norm = float(
                    (
                        np.square(np.abs(full_spectrum)).astype(np.float64)
                        * frequency_weights
                    ).sum()
                    / (PHYSICAL_SCALE * PHYSICAL_SCALE)
                )
                unmodeled_energy_norm = float(
                    (
                        np.square(np.abs(full_spectrum[64:])).astype(np.float64)
                        * frequency_weights[64:]
                    ).sum()
                    / (PHYSICAL_SCALE * PHYSICAL_SCALE)
                )
                parent_error = float(np.square(parent.astype(np.float64) - target.astype(np.float64)).sum())
                baseline_values.append(math.sqrt(parent_error / max(target_total, 1.0e-300)))
                maximum_normalized = max(maximum_normalized, current_maximum)
                output["sample_id"][local] = sample_id
                output["family"][local] = row["family"]
                output["role"][local] = row["role"]
                output["source_index"][local] = index
                medium_ds[local] = medium.astype(np.float16)
                interface_ds[local] = interface.astype(np.float16)
                source_map_ds[local] = np.asarray(raw["source_map"], dtype=np.float32)
                wavelet_ds[local] = np.asarray(raw["source_wavelet"], dtype=np.float32)
                parameters_ds[local] = np.asarray(raw["source_parameters"], dtype=np.float32)
                travel_ds[local] = np.asarray(travel.read(sample_id)[0], dtype=np.float32)
                target_ds[local] = target.astype(np.float16)
                parent_ds[local] = parent.astype(np.float16)
                energy_ds[local] = target_total
                full_time_energy_ds[local] = full_time_energy_norm
                unmodeled_energy_ds[local] = unmodeled_energy_norm
                output.flush()
                print(json.dumps({
                    "event": "cache_record", "worker": args.worker_index,
                    "record": local + 1, "of": len(rows), "sample_id": sample_id,
                    "role": row["role"], "baseline_relative_l2": baseline_values[-1],
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
        "schema": "transfer_dg_phase_scatter64_pilot_cache_summary_v1",
        "status": "complete",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "record_count": len(rows),
        "role_counts": {role: sum(row["role"] == role for row in rows) for role in ("fit", "calibration", "confirmation")},
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
        "maximum_normalized": maximum_normalized,
        "baseline_mean": float(np.mean(baseline_values)),
        "training_full_truth_allowed": True,
        "validation_opened": False,
        "test_id_opened": False,
        "smoke": bool(args.max_records),
        "elapsed_s": time.time() - started,
    }
    atomic_json(summary, summary_path)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
