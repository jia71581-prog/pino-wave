#!/usr/bin/env python3
"""Train one 256-record E1d raw/phase Background-WFP generalization lane."""
from __future__ import annotations

import argparse
from collections import defaultdict
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
from saved_time_phase_operator_v4.phase_carrier import (  # noqa: E402
    rotate_complex_pairs,
    travel_phase_carrier,
)
from saved_time_phase_operator_v4.wfp import BackgroundFrequencyOperator, parameter_count  # noqa: E402


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_checkpoint(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class TravelCollection:
    def __init__(self, paths: list[Path], *, expected_count: int = 256) -> None:
        self.handles = []
        self.location = {}
        self.cache: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {}
        for file_index, path in enumerate(paths):
            summary = json.loads(path.with_suffix(path.suffix + ".summary.json").read_text())
            if summary.get("status") != "complete" or summary.get("validation_opened") or summary.get("test_id_opened"):
                raise RuntimeError(f"invalid E1d travel summary: {path}")
            handle = h5py.File(path, "r", swmr=True)
            self.handles.append(handle)
            if handle.attrs.get("schema") != "transfer_dg_wfp_e1d_travel_v1" or handle.attrs.get("status") != "complete":
                raise RuntimeError(f"invalid E1d travel cache: {path}")
            for local, sample in enumerate(handle["sample_id"].asstr()[:]):
                if sample in self.location:
                    raise RuntimeError(f"duplicate travel sample: {sample}")
                self.location[str(sample)] = (file_index, local)
        if len(self.location) != int(expected_count):
            raise RuntimeError(f"E1d travel census mismatch: {len(self.location)}")

    def read(self, sample_id: str):
        sample = str(sample_id)
        if sample not in self.cache:
            file_index, local = self.location[sample]
            handle = self.handles[file_index]
            self.cache[sample] = (
                np.asarray(handle["travel_physical_s"][local], dtype=np.float32),
                np.asarray(handle["travel_exterior_s"][local], dtype=np.float32),
                float(handle["physical_total_square"][local]),
                float(handle["auxiliary_total_square"][local]),
            )
        return self.cache[sample]

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles = []


def apply_phase(
    physical: torch.Tensor,
    auxiliary: torch.Tensor,
    physical_travel: torch.Tensor,
    exterior_travel: torch.Tensor,
    frequencies_hz: torch.Tensor,
    enabled: bool,
):
    if not enabled:
        return physical, auxiliary
    return (
        rotate_complex_pairs(
            physical, travel_phase_carrier(physical_travel, frequencies_hz)
        ),
        rotate_complex_pairs(
            auxiliary, travel_phase_carrier(exterior_travel, frequencies_hz)
        ),
    )


@torch.inference_mode()
def evaluate(model, builder, travel, positions, device, phase_enabled):
    model.eval()
    active = torch.from_numpy(builder.active).to(device)
    rows = []
    top = outer = 0.0
    started = time.perf_counter()
    for position in positions:
        sample_id = builder.collection.records[position][2]
        family = builder.collection.records[position][3]
        physical_np, exterior_np, _, _ = travel.read(sample_id)
        physical_travel = torch.from_numpy(physical_np).to(device)
        exterior_travel = torch.from_numpy(exterior_np).to(device)
        numerator = denominator = pred_square = dot = aux_num = aux_den = 0.0
        for start in range(0, 64, 4):
            items = [builder.build(position, frequency, device) for frequency in range(start, start + 4)]
            medium, source, scalars, target, auxiliary_target = (
                torch.cat([item[index] for item in items], dim=0) for index in range(5)
            )
            frequency_hz = torch.tensor(
                [float(item[5]["frequency_hz"]) for item in items], device=device
            )
            prediction, auxiliary = model(medium, source, scalars)
            prediction, auxiliary = apply_phase(
                prediction, auxiliary,
                physical_travel[None].expand(4, -1, -1),
                exterior_travel[None].expand(4, -1, -1),
                frequency_hz, phase_enabled,
            )
            numerator += float((prediction.double() - target.double()).square().sum())
            denominator += float(target.double().square().sum())
            pred_square += float(prediction.double().square().sum())
            dot += float((prediction.double() * target.double()).sum())
            mask = active[None, None].expand_as(auxiliary_target)
            aux_num += float((auxiliary.double()[mask] - auxiliary_target.double()[mask]).square().sum())
            aux_den += float(auxiliary_target.double()[mask].square().sum())
            top = max(top, float(prediction[..., 0, :].abs().max()), float(auxiliary[:, :2, 0].abs().max()))
            outer = max(outer, float(auxiliary[:, :2, -1].abs().max()), float(auxiliary[:, :2, :, 0].abs().max()), float(auxiliary[:, :2, :, -1].abs().max()))
        rows.append({
            "sample_id": sample_id,
            "family": family,
            "physical_relative_l2": math.sqrt(numerator / max(denominator, 1.0e-30)),
            "prediction_target_norm_ratio": math.sqrt(pred_square / max(denominator, 1.0e-30)),
            "cosine": dot / max(math.sqrt(pred_square * denominator), 1.0e-30),
            "cpml_normalized_relative_l2": math.sqrt(aux_num / max(aux_den, 1.0e-30)),
        })
    return {
        "record_count": len(rows),
        "physical_mean": float(np.mean([row["physical_relative_l2"] for row in rows])),
        "physical_max": float(np.max([row["physical_relative_l2"] for row in rows])),
        "prediction_target_norm_ratio_mean": float(np.mean([row["prediction_target_norm_ratio"] for row in rows])),
        "cosine_mean": float(np.mean([row["cosine"] for row in rows])),
        "cpml_normalized_mean": float(np.mean([row["cpml_normalized_relative_l2"] for row in rows])),
        "per_family": {
            family: float(np.mean([row["physical_relative_l2"] for row in rows if row["family"] == family]))
            for family in FAMILIES
        },
        "top_pressure_max_abs": top,
        "outer_pressure_max_abs": outer,
        "elapsed_s": time.perf_counter() - started,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--travel", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--carrier", choices=("raw", "phase"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    collection = CacheCollection(args.cache, manifest)
    travel = TravelCollection(args.travel)
    fit, calibration, confirmation = collection.split_positions()
    builder = FeatureBuilder(collection)
    by_family = defaultdict(list)
    for position in fit:
        by_family[collection.records[position][3]].append(position)
    if any(len(by_family[family]) != 48 for family in FAMILIES):
        raise RuntimeError("E1d requires 48 fit records per family")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = BackgroundFrequencyOperator(
        medium_channels=12, source_channels=5, width=32, rank=16, depth=4,
        arm="wfp", radii=(1, 2, 3, 4), fno_modes=32,
    ).to(device)
    total_updates = int(args.epochs) * 48 * 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=args.lr * 0.01
    )
    active = torch.from_numpy(builder.active).to(device)
    generator = np.random.default_rng(args.seed + 71)
    identity = {
        "schema": "transfer_dg_wfp_e1d_lane_identity_v1",
        "carrier": args.carrier, "seed": args.seed, "epochs": args.epochs,
        "updates_per_epoch": 48 * 64, "total_updates": total_updates,
        "parameter_count": parameter_count(model),
        "training_full_truth_allowed": True,
        "model_input_future_wavefield_frames": 0,
        "test_future_truth_access": False,
        "manifest_sha256": sha256(args.manifest),
        "preregistration_sha256": sha256(args.preregistration),
        "trainer_sha256": sha256(Path(__file__)),
        "phase_module_sha256": sha256(ROOT / "saved_time_phase_operator_v4/phase_carrier.py"),
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(identity, args.output_dir / "run_identity.json")
    best = None
    global_update = 0
    started = time.time()
    metrics_path = args.output_dir / "metrics.jsonl"
    for epoch in range(1, args.epochs + 1):
        schedules = {}
        for family in FAMILIES:
            schedule = np.asarray(
                [(position, frequency) for position in by_family[family] for frequency in range(64)],
                dtype=np.int64,
            )
            generator.shuffle(schedule)
            schedules[family] = schedule
        model.train()
        for step in range(48 * 64):
            rows = [schedules[family][step] for family in FAMILIES]
            items = [builder.build(int(position), int(frequency), device) for position, frequency in rows]
            medium, source, scalars, target, auxiliary_target = (
                torch.cat([item[index] for item in items], dim=0) for index in range(5)
            )
            sample_ids = [str(item[5]["sample_id"]) for item in items]
            frequency_hz = torch.tensor(
                [float(item[5]["frequency_hz"]) for item in items], device=device
            )
            travel_rows = [travel.read(sample_id) for sample_id in sample_ids]
            physical_travel = torch.from_numpy(np.stack([row[0] for row in travel_rows])).to(device)
            exterior_travel = torch.from_numpy(np.stack([row[1] for row in travel_rows])).to(device)
            physical_total = torch.tensor([row[2] for row in travel_rows], device=device, dtype=torch.float32)
            auxiliary_total = torch.tensor([row[3] for row in travel_rows], device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction, auxiliary = model(medium, source, scalars)
            prediction, auxiliary = apply_phase(
                prediction, auxiliary, physical_travel, exterior_travel,
                frequency_hz, args.carrier == "phase",
            )
            physical_error = (prediction.float() - target.float()).square().sum(dim=(1, 2, 3))
            physical_loss = (64.0 * physical_error / physical_total.clamp_min(1.0e-8)).mean()
            mask = active[None, None].expand_as(auxiliary_target)
            auxiliary_error = (
                (auxiliary.float() - auxiliary_target.float()).square() * mask
            ).sum(dim=(1, 2, 3))
            auxiliary_loss = (64.0 * auxiliary_error / auxiliary_total.clamp_min(1.0e-8)).mean()
            loss = physical_loss + 0.05 * auxiliary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_update += 1
            if global_update % 200 == 0:
                event = {
                    "event": "update", "epoch": epoch, "update": global_update,
                    "physical_loss": float(physical_loss.detach()),
                    "cpml_loss": float(auxiliary_loss.detach()),
                    "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
                }
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event), flush=True)
        calibration_metrics = evaluate(
            model, builder, travel, calibration, device, args.carrier == "phase"
        )
        event = {"event": "calibration", "epoch": epoch, "update": global_update, "metrics": calibration_metrics}
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        checkpoint = {
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "epoch": epoch, "update": global_update, "identity": identity,
            "calibration": calibration_metrics,
        }
        atomic_checkpoint(checkpoint, args.output_dir / "latest.pt")
        score = float(calibration_metrics["physical_mean"])
        if best is None or score < best["score"]:
            best = {"score": score, "epoch": epoch, "update": global_update, "metrics": calibration_metrics}
            atomic_checkpoint(checkpoint, args.output_dir / "best.pt")
            atomic_json(best, args.output_dir / "best.json")
        print(json.dumps({"event": "calibration", "epoch": epoch, "score": score}), flush=True)
    if best is None:
        raise RuntimeError("E1d produced no checkpoint")
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    confirmation_metrics = evaluate(
        model, builder, travel, confirmation, device, args.carrier == "phase"
    )
    terminal = {
        "schema": "transfer_dg_wfp_e1d_lane_terminal_v1",
        "status": "complete", "carrier": args.carrier, "seed": args.seed,
        "best_epoch": best["epoch"], "best_update": best["update"],
        "calibration": best["metrics"], "confirmation": confirmation_metrics,
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "best_checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_s": time.time() - started,
        "training_full_truth_allowed": True,
        "model_input_future_wavefield_frames": 0,
        "test_future_truth_access": False,
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(terminal, args.output_dir / "terminal.json")
    collection.close(); travel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
