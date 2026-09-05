#!/usr/bin/env python3
"""Train one true-interface phase/scatter correction seed on train-only data."""
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

from saved_time_phase_operator_v4.phase_carrier import travel_phase_carrier  # noqa: E402
from saved_time_phase_operator_v4.phase_scatter import (  # noqa: E402
    PhaseScatterCorrectionOperator,
    combine_phase_scatter,
    parameter_count,
)


FAMILIES = ("uniform", "layered", "anomaly", "marmousi")


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


def atomic_checkpoint(payload: dict, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


class PilotCollection:
    def __init__(self, paths: list[Path], manifest: dict) -> None:
        self.handles = []
        self.records = []
        expected_selection = str(manifest["selection_sha256"])
        try:
            for file_index, path in enumerate(paths):
                summary = json.loads(path.with_suffix(path.suffix + ".summary.json").read_text())
                if summary.get("status") != "complete" or summary.get("smoke"):
                    raise RuntimeError(f"invalid pilot cache summary: {path}")
                handle = h5py.File(path, "r", swmr=True)
                self.handles.append(handle)
                if handle.attrs.get("status") != "complete" or handle.attrs.get("manifest_selection_sha256") != expected_selection:
                    raise RuntimeError(f"invalid pilot cache shard: {path}")
                for local in range(len(handle["sample_id"])):
                    self.records.append((
                        file_index,
                        local,
                        str(handle["sample_id"].asstr()[local]),
                        str(handle["family"].asstr()[local]),
                        str(handle["role"].asstr()[local]),
                    ))
            if len(self.records) != int(manifest["record_count"]):
                raise RuntimeError("pilot cache census mismatch")
            self.frequency_hz = np.asarray(self.handles[0]["frequency_hz"], dtype=np.float32)
            if len(self.frequency_hz) != 64:
                raise RuntimeError("phase/scatter pilot requires 64 bins")
        except Exception:
            self.close()
            raise
        self.static_cache = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.close()
        self.handles = []

    def positions(self, role: str) -> list[int]:
        return [index for index, row in enumerate(self.records) if row[4] == role]

    def static(self, position: int) -> dict:
        if position not in self.static_cache:
            file_index, local, sample_id, family, role = self.records[position]
            handle = self.handles[file_index]
            wavelet_fft = np.fft.rfft(
                np.asarray(handle["source_wavelet"][local], dtype=np.float32), norm="ortho"
            )
            wavelet_fft /= max(float(np.max(np.abs(wavelet_fft))), 1.0e-12)
            self.static_cache[position] = {
                "sample_id": sample_id,
                "family": family,
                "role": role,
                "medium": np.asarray(handle["medium"][local], dtype=np.float32),
                "interface": np.asarray(handle["interface_strength"][local], dtype=np.float32),
                "source_map": np.asarray(handle["source_map"][local], dtype=np.float32),
                "wavelet_fft": wavelet_fft,
                "source_parameters": np.asarray(handle["source_parameters"][local], dtype=np.float32),
                "travel": np.asarray(handle["travel_physical_s"][local], dtype=np.float32),
                "target_total": float(handle["target_total_square"][local]),
                "full_time_total": float(handle["full_target_time_square_norm"][local]),
                "unmodeled_time": float(handle["unmodeled_time_square_norm"][local]),
            }
        return self.static_cache[position]

    def build(self, position: int, frequency: int, device: torch.device) -> dict:
        file_index, local, _, _, _ = self.records[position]
        handle = self.handles[file_index]
        row = self.static(position)
        source_map = row["source_map"]
        wave = row["wavelet_fft"][frequency]
        sx, sz, f0, t0 = (float(value) for value in row["source_parameters"])
        x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
        z = np.arange(221, dtype=np.float32) * 10.0
        xx, zz = np.meshgrid(x, z)
        extended_source = np.zeros((221, 241), dtype=np.float32)
        extended_source[:201, 20:221] = source_map
        source = np.stack((
            extended_source,
            extended_source * float(wave.real),
            extended_source * float(wave.imag),
            np.clip((xx - sx) / 2000.0, -1.2, 1.2),
            np.clip((zz - sz) / 2000.0, -0.2, 1.2),
        ), axis=0).astype(np.float32)
        scalars = np.asarray((
            float(self.frequency_hz[frequency]) / 200.0,
            f0 / 30.0,
            t0 / 0.2,
            float(wave.real),
            float(wave.imag),
        ), dtype=np.float32)
        return {
            "medium": torch.from_numpy(row["medium"])[None].to(device),
            "source": torch.from_numpy(source)[None].to(device),
            "scalars": torch.from_numpy(scalars)[None].to(device),
            "travel": torch.from_numpy(row["travel"])[None].to(device),
            "frequency_hz": torch.tensor([self.frequency_hz[frequency]], device=device),
            "target": torch.from_numpy(np.asarray(
                handle["target_coeff_norm"][local, frequency], dtype=np.float32
            ))[None].to(device),
            "parent": torch.from_numpy(np.asarray(
                handle["parent_coeff_norm"][local, frequency], dtype=np.float32
            ))[None].to(device),
            "target_total": row["target_total"],
        }


def predict_batch(model, items: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    medium = torch.cat([item["medium"] for item in items])
    source = torch.cat([item["source"] for item in items])
    scalars = torch.cat([item["scalars"] for item in items])
    travel = torch.cat([item["travel"] for item in items])
    frequency_hz = torch.cat([item["frequency_hz"] for item in items])
    parent = torch.cat([item["parent"] for item in items])
    phase_residual, scatter_residual = model(medium, source, scalars)
    carrier = travel_phase_carrier(travel, frequency_hz)
    prediction = combine_phase_scatter(parent, phase_residual, scatter_residual, carrier)
    return prediction, phase_residual, scatter_residual


@torch.inference_mode()
def evaluate(
    model: PhaseScatterCorrectionOperator,
    collection: PilotCollection,
    positions: list[int],
    device: torch.device,
) -> dict:
    model.eval()
    rows = []
    top_maximum = 0.0
    for position in positions:
        static = collection.static(position)
        candidate_low_error = parent_low_error = low_target = 0.0
        phase_square = scatter_square = 0.0
        candidate_square = target_square = dot = 0.0
        for start in range(0, 64, 8):
            items = [collection.build(position, frequency, device) for frequency in range(start, start + 8)]
            prediction, phase_residual, scatter_residual = predict_batch(model, items)
            target = torch.cat([item["target"] for item in items])
            parent = torch.cat([item["parent"] for item in items])
            weights = torch.full((8, 1, 1, 1), 2.0, device=device, dtype=torch.float64)
            if start == 0:
                weights[0] = 1.0
            candidate_low_error += float(
                ((prediction.double() - target.double()).square() * weights).sum()
            )
            parent_low_error += float(
                ((parent.double() - target.double()).square() * weights).sum()
            )
            low_target += float((target.double().square() * weights).sum())
            phase_square += float((phase_residual.double().square() * weights).sum())
            scatter_square += float((scatter_residual.double().square() * weights).sum())
            candidate_square += float((prediction.double().square() * weights).sum())
            target_square += float((target.double().square() * weights).sum())
            dot += float((prediction.double() * target.double() * weights).sum())
            top_maximum = max(top_maximum, float(prediction[..., 0, :].abs().max()))
        unmodeled = static["unmodeled_time"]
        full_total = static["full_time_total"]
        candidate_full = math.sqrt((candidate_low_error + unmodeled) / max(full_total, 1.0e-300))
        parent_full = math.sqrt((parent_low_error + unmodeled) / max(full_total, 1.0e-300))
        rows.append({
            "sample_id": static["sample_id"],
            "family": static["family"],
            "candidate_full401_relative_l2": candidate_full,
            "parent_full401_relative_l2": parent_full,
            "relative_improvement": 1.0 - candidate_full / max(parent_full, 1.0e-300),
            "candidate_modeled64_relative_l2": math.sqrt(candidate_low_error / max(low_target, 1.0e-300)),
            "parent_modeled64_relative_l2": math.sqrt(parent_low_error / max(low_target, 1.0e-300)),
            "prediction_target_norm_ratio": math.sqrt(candidate_square / max(target_square, 1.0e-300)),
            "cosine": dot / max(math.sqrt(candidate_square * target_square), 1.0e-300),
            "phase_residual_energy_ratio": math.sqrt(phase_square / max(target_square, 1.0e-300)),
            "scatter_residual_energy_ratio": math.sqrt(scatter_square / max(target_square, 1.0e-300)),
        })
    candidate_values = [row["candidate_full401_relative_l2"] for row in rows]
    parent_values = [row["parent_full401_relative_l2"] for row in rows]
    return {
        "record_count": len(rows),
        "candidate_mean": float(np.mean(candidate_values)),
        "parent_mean": float(np.mean(parent_values)),
        "relative_improvement": 1.0 - float(np.mean(candidate_values)) / max(float(np.mean(parent_values)), 1.0e-300),
        "nonworse_count": sum(row["candidate_full401_relative_l2"] <= row["parent_full401_relative_l2"] for row in rows),
        "per_family_candidate": {
            family: float(np.mean([
                row["candidate_full401_relative_l2"] for row in rows if row["family"] == family
            ])) for family in FAMILIES
        },
        "per_family_parent": {
            family: float(np.mean([
                row["parent_full401_relative_l2"] for row in rows if row["family"] == family
            ])) for family in FAMILIES
        },
        "top_pressure_max_abs": top_maximum,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    bindings = prereg["bindings"]
    if sha256(Path(__file__)) != bindings["trainer_sha256"]:
        raise RuntimeError("phase/scatter trainer binding drift")
    if sha256(args.manifest) != bindings["pilot_manifest_sha256"]:
        raise RuntimeError("pilot manifest binding drift")
    if args.seed not in prereg["training"]["seeds"]:
        raise RuntimeError("unregistered pilot seed")
    collection = PilotCollection(args.cache, manifest)
    fit = collection.positions("fit")
    calibration = collection.positions("calibration")
    confirmation = collection.positions("confirmation")
    if (len(fit), len(calibration), len(confirmation)) != (68, 34, 34):
        raise RuntimeError("pilot role census mismatch")
    by_family = defaultdict(list)
    for position in fit:
        by_family[collection.records[position][3]].append(position)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = PhaseScatterCorrectionOperator().to(device)
    plan = prereg["training"]
    steps_per_epoch = max(len(by_family[family]) * 64 for family in FAMILIES)
    total_updates = int(plan["epochs"]) * steps_per_epoch
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(plan["learning_rate"]), weight_decay=float(plan["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=float(plan["eta_min"])
    )
    identity = {
        "schema": "transfer_dg_phase_scatter64_pilot_lane_identity_v1",
        "seed": args.seed,
        "epochs": int(plan["epochs"]),
        "steps_per_epoch": steps_per_epoch,
        "total_updates": total_updates,
        "parameter_count": parameter_count(model),
        "fit_count": len(fit),
        "calibration_count": len(calibration),
        "confirmation_count": len(confirmation),
        "frequency_count": 64,
        "parent_frozen": True,
        "model_input_future_wavefield_frames": 0,
        "training_full_truth_allowed": True,
        "manifest_sha256": sha256(args.manifest),
        "preregistration_sha256": sha256(args.preregistration),
        "trainer_sha256": sha256(Path(__file__)),
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(identity, args.output_dir / "run_identity.json")
    initial = evaluate(model, collection, calibration, device)
    if abs(initial["candidate_mean"] - initial["parent_mean"]) > 1.0e-8:
        raise RuntimeError("zero-initialized correction does not reproduce parent")
    best = None
    global_update = 0
    rng = np.random.default_rng(args.seed + 191)
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.time()
    for epoch in range(1, int(plan["epochs"]) + 1):
        schedules = {}
        for family in FAMILIES:
            values = np.asarray([
                (position, frequency)
                for position in by_family[family]
                for frequency in range(64)
            ], dtype=np.int64)
            rng.shuffle(values)
            schedules[family] = values
        model.train()
        for step in range(steps_per_epoch):
            selected = [schedules[family][step % len(schedules[family])] for family in FAMILIES]
            items = [collection.build(int(position), int(frequency), device) for position, frequency in selected]
            target = torch.cat([item["target"] for item in items])
            target_total = torch.tensor(
                [item["target_total"] for item in items], device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            prediction, phase_residual, scatter_residual = predict_batch(model, items)
            error = (prediction.float() - target.float()).square().sum((1, 2, 3))
            physical_loss = (64.0 * error / target_total.clamp_min(1.0e-8)).mean()
            correction_penalty = 1.0e-6 * (
                phase_residual.float().square().mean() + scatter_residual.float().square().mean()
            )
            loss = physical_loss + correction_penalty
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at update {global_update + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_update += 1
            if global_update % 100 == 0:
                event = {
                    "event": "update", "epoch": epoch, "update": global_update,
                    "physical_loss": float(physical_loss.detach()),
                    "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                print(json.dumps(event), flush=True)
        calibration_metrics = evaluate(model, collection, calibration, device)
        event = {"event": "calibration", "epoch": epoch, "update": global_update, "metrics": calibration_metrics}
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        score = float(calibration_metrics["candidate_mean"])
        if best is None or score < best["score"]:
            best = {"score": score, "epoch": epoch, "update": global_update, "metrics": calibration_metrics}
            atomic_checkpoint({
                "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "identity": identity,
                "epoch": epoch,
                "update": global_update,
                "calibration": calibration_metrics,
            }, args.output_dir / "best.pt")
        print(json.dumps({"event": "calibration", "epoch": epoch, "candidate": score, "parent": calibration_metrics["parent_mean"]}), flush=True)
    checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    confirmation_metrics = evaluate(model, collection, confirmation, device)
    terminal = {
        "schema": "transfer_dg_phase_scatter64_pilot_lane_terminal_v1",
        "status": "complete",
        "seed": args.seed,
        "initial_calibration": initial,
        "best_epoch": best["epoch"],
        "best_update": best["update"],
        "calibration": best["metrics"],
        "confirmation": confirmation_metrics,
        "best_checkpoint": str((args.output_dir / "best.pt").resolve()),
        "best_checkpoint_sha256": sha256(args.output_dir / "best.pt"),
        "elapsed_s": time.time() - started,
        "validation_opened": False,
        "test_id_opened": False,
    }
    atomic_json(terminal, args.output_dir / "terminal.json")
    collection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
