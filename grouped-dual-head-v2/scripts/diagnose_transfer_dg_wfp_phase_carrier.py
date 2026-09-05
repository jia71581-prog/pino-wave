#!/usr/bin/env python3
"""E1c one-record paired capacity test for an analytic travel-phase carrier."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

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


def uniform_travel_fields(builder: FeatureBuilder, position: int) -> tuple[np.ndarray, np.ndarray]:
    row = builder.collection.raw(position, 0)
    velocity = np.asarray(row["velocity"], dtype=np.float32)
    if float(np.ptp(velocity)) != 0.0:
        raise RuntimeError("E1c registered capacity record must be uniform")
    speed = float(velocity[0, 0])
    sx, sz = (float(value) for value in row["source_parameters"][:2])
    physical_x = np.arange(201, dtype=np.float32) * 10.0
    physical_z = np.arange(201, dtype=np.float32) * 10.0
    pxx, pzz = np.meshgrid(physical_x, physical_z)
    extended_x = np.arange(241, dtype=np.float32) * 10.0 - 200.0
    extended_z = np.arange(221, dtype=np.float32) * 10.0
    exx, ezz = np.meshgrid(extended_x, extended_z)
    physical = np.sqrt((pxx - sx) ** 2 + (pzz - sz) ** 2) / speed
    extended = np.sqrt((exx - sx) ** 2 + (ezz - sz) ** 2) / speed
    return physical.astype(np.float32), extended.astype(np.float32)


def apply_carrier(
    physical: torch.Tensor,
    auxiliary: torch.Tensor,
    physical_travel: torch.Tensor,
    extended_travel: torch.Tensor,
    frequency_hz: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not enabled:
        return physical, auxiliary
    return (
        rotate_complex_pairs(
            physical, travel_phase_carrier(physical_travel, frequency_hz)
        ),
        rotate_complex_pairs(
            auxiliary, travel_phase_carrier(extended_travel, frequency_hz)
        ),
    )


@torch.inference_mode()
def measure(model, builder, position, device, physical_travel, extended_travel, carrier):
    numerator = denominator = pred_square = dot = 0.0
    top = outer = 0.0
    for start in range(0, 64, 4):
        items = [builder.build(position, frequency, device) for frequency in range(start, start + 4)]
        medium, source, scalars, target = (
            torch.cat([item[index] for item in items], dim=0) for index in range(4)
        )
        frequency_hz = torch.tensor(
            [float(item[5]["frequency_hz"]) for item in items], device=device
        )
        prediction, auxiliary = model(medium, source, scalars)
        prediction, auxiliary = apply_carrier(
            prediction, auxiliary,
            physical_travel.expand(4, -1, -1),
            extended_travel.expand(4, -1, -1),
            frequency_hz, carrier,
        )
        numerator += float((prediction.double() - target.double()).square().sum())
        denominator += float(target.double().square().sum())
        pred_square += float(prediction.double().square().sum())
        dot += float((prediction.double() * target.double()).sum())
        top = max(top, float(prediction[..., 0, :].abs().max()), float(auxiliary[:, :2, 0].abs().max()))
        outer = max(outer, float(auxiliary[:, :2, -1].abs().max()), float(auxiliary[:, :2, :, 0].abs().max()), float(auxiliary[:, :2, :, -1].abs().max()))
    return {
        "relative_l2": math.sqrt(numerator / max(denominator, 1.0e-30)),
        "prediction_target_norm_ratio": math.sqrt(pred_square / max(denominator, 1.0e-30)),
        "cosine": dot / max(math.sqrt(pred_square * denominator), 1.0e-30),
        "top_pressure_max_abs": top,
        "outer_pressure_max_abs": outer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--carrier", choices=("raw", "phase"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=800)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text())
    collection = CacheCollection(args.cache, manifest)
    fit, _, _ = collection.split_positions()
    position = min(
        (index for index in fit if collection.records[index][3] == "uniform"),
        key=lambda index: collection.records[index][2],
    )
    sample_id = collection.records[position][2]
    builder = FeatureBuilder(collection)
    physical_np, extended_np = uniform_travel_fields(builder, position)
    device = torch.device("cuda")
    physical_travel = torch.from_numpy(physical_np)[None].to(device)
    extended_travel = torch.from_numpy(extended_np)[None].to(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = BackgroundFrequencyOperator(
        medium_channels=12, source_channels=5, width=32, rank=16, depth=4,
        arm="wfp", radii=(1, 2, 3, 4), fno_modes=32,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates, eta_min=5.0e-6
    )
    active = torch.from_numpy(builder.active).to(device)
    # Full-record denominators make every 4-bin update an unbiased component of
    # one stable objective and remove E1b's low-energy per-batch explosions.
    physical_total = auxiliary_total = 0.0
    for frequency in range(64):
        _, _, _, target, auxiliary_target, _ = builder.build(position, frequency, device)
        physical_total += float(target.double().square().sum())
        mask = active[None, None].expand_as(auxiliary_target)
        auxiliary_total += float(auxiliary_target.double()[mask].square().sum())
    initial = measure(
        model, builder, position, device, physical_travel, extended_travel,
        args.carrier == "phase",
    )
    best = {"update": 0, "metrics": initial}
    generator = np.random.default_rng(args.seed)
    order = np.arange(64, dtype=np.int64)
    started = time.time()
    for update in range(1, args.updates + 1):
        if (update - 1) % 16 == 0:
            generator.shuffle(order)
        offset = ((update - 1) % 16) * 4
        frequencies = order[offset : offset + 4]
        items = [builder.build(position, int(frequency), device) for frequency in frequencies]
        medium, source, scalars, target, auxiliary_target = (
            torch.cat([item[index] for item in items], dim=0) for index in range(5)
        )
        frequency_hz = torch.tensor(
            [float(item[5]["frequency_hz"]) for item in items], device=device
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, auxiliary = model(medium, source, scalars)
        prediction, auxiliary = apply_carrier(
            prediction, auxiliary,
            physical_travel.expand(4, -1, -1),
            extended_travel.expand(4, -1, -1),
            frequency_hz, args.carrier == "phase",
        )
        physical_loss = 16.0 * (prediction.float() - target.float()).square().sum() / max(physical_total, 1.0e-8)
        mask = active[None, None].expand_as(auxiliary_target)
        auxiliary_loss = 16.0 * (auxiliary.float()[mask] - auxiliary_target.float()[mask]).square().sum() / max(auxiliary_total, 1.0e-8)
        loss = physical_loss + 0.05 * auxiliary_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update % 100 == 0:
            metrics = measure(
                model, builder, position, device, physical_travel, extended_travel,
                args.carrier == "phase",
            )
            print(json.dumps({"update": update, "loss": float(loss.detach()), **metrics}), flush=True)
            if metrics["relative_l2"] < best["metrics"]["relative_l2"]:
                best = {"update": update, "metrics": metrics}
                torch.save({
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "carrier": args.carrier, "seed": args.seed, "sample_id": sample_id,
                    "update": update, "metrics": metrics,
                }, args.output_dir / "best.pt")
    passed = (
        best["metrics"]["relative_l2"] < 0.95
        and best["metrics"]["prediction_target_norm_ratio"] > 0.05
        and best["metrics"]["top_pressure_max_abs"] == 0.0
        and best["metrics"]["outer_pressure_max_abs"] == 0.0
    )
    terminal = {
        "schema": "transfer_dg_wfp_e1c_phase_carrier_terminal_v1",
        "status": "passed" if passed else "rejected",
        "carrier": args.carrier, "seed": args.seed, "sample_id": sample_id,
        "parameter_count": parameter_count(model), "initial": initial, "best": best,
        "elapsed_s": time.time() - started,
        "preregistration_sha256": sha256(args.preregistration),
        "trainer_sha256": sha256(Path(__file__)),
        "phase_module_sha256": sha256(ROOT / "saved_time_phase_operator_v4/phase_carrier.py"),
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(terminal, args.output_dir / "terminal.json")
    collection.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
