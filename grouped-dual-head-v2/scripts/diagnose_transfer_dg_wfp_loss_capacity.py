#!/usr/bin/env python3
"""One-record energy-relative loss capacity test for Background-WFP E1b."""
from __future__ import annotations

import argparse
import hashlib
import json
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

from scripts.train_transfer_dg_wfp_e1 import (  # noqa: E402
    CacheCollection,
    FeatureBuilder,
)
from saved_time_phase_operator_v4.wfp import (  # noqa: E402
    BackgroundFrequencyOperator,
    parameter_count,
)


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


@torch.inference_mode()
def measure(model, builder, position, device):
    numerator = denominator = pred_square = dot = 0.0
    top = outer = 0.0
    for start in range(0, 64, 4):
        items = [builder.build(position, frequency, device) for frequency in range(start, start + 4)]
        medium, source, scalars, target = (
            torch.cat([item[index] for item in items], dim=0) for index in range(4)
        )
        prediction, auxiliary = model(medium, source, scalars)
        numerator += float((prediction.double() - target.double()).square().sum())
        denominator += float(target.double().square().sum())
        pred_square += float(prediction.double().square().sum())
        dot += float((prediction.double() * target.double()).sum())
        top = max(top, float(prediction[..., 0, :].abs().max()), float(auxiliary[:, :2, 0].abs().max()))
        outer = max(outer, float(auxiliary[:, :2, -1].abs().max()), float(auxiliary[:, :2, :, 0].abs().max()), float(auxiliary[:, :2, :, -1].abs().max()))
    return {
        "relative_l2": (numerator / max(denominator, 1.0e-30)) ** 0.5,
        "prediction_target_norm_ratio": (pred_square / max(denominator, 1.0e-30)) ** 0.5,
        "cosine": dot / max((pred_square * denominator) ** 0.5, 1.0e-30),
        "top_pressure_max_abs": top,
        "outer_pressure_max_abs": outer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("fno", "wfp"), required=True)
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
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = BackgroundFrequencyOperator(
        medium_channels=12, source_channels=5, width=32, rank=16, depth=4,
        arm=args.arm, radii=(1, 2, 3, 4), fno_modes=32,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.updates, eta_min=5.0e-6
    )
    active = torch.from_numpy(builder.active).to(device)
    rng = np.random.default_rng(args.seed)
    initial = measure(model, builder, position, device)
    best = {"update": 0, "metrics": initial}
    started = time.time()
    for update in range(1, args.updates + 1):
        frequencies = rng.choice(64, size=4, replace=False)
        items = [builder.build(position, int(frequency), device) for frequency in frequencies]
        medium, source, scalars, target, auxiliary_target = (
            torch.cat([item[index] for item in items], dim=0) for index in range(5)
        )
        optimizer.zero_grad(set_to_none=True)
        prediction, auxiliary = model(medium, source, scalars)
        physical = (prediction.float() - target.float()).square().sum() / target.float().square().sum().clamp_min(1.0e-8)
        mask = active[None, None].expand_as(auxiliary_target)
        cpml = (auxiliary.float()[mask] - auxiliary_target.float()[mask]).square().sum() / auxiliary_target.float()[mask].square().sum().clamp_min(1.0e-8)
        loss = physical + 0.05 * cpml
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if update % 100 == 0 or update == args.updates:
            metrics = measure(model, builder, position, device)
            print(json.dumps({"update": update, "loss": float(loss.detach()), **metrics}), flush=True)
            if metrics["relative_l2"] < best["metrics"]["relative_l2"]:
                best = {"update": update, "metrics": metrics}
                torch.save({
                    "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    "arm": args.arm, "seed": args.seed, "sample_id": sample_id,
                    "update": update, "metrics": metrics,
                }, args.output_dir / "best.pt")
    passed = (
        best["metrics"]["relative_l2"] < 0.95
        and best["metrics"]["prediction_target_norm_ratio"] > 0.05
        and best["metrics"]["top_pressure_max_abs"] == 0.0
        and best["metrics"]["outer_pressure_max_abs"] == 0.0
    )
    terminal = {
        "schema": "transfer_dg_wfp_e1b_loss_capacity_terminal_v1",
        "status": "passed" if passed else "rejected",
        "arm": args.arm, "seed": args.seed, "sample_id": sample_id,
        "parameter_count": parameter_count(model),
        "initial": initial, "best": best,
        "elapsed_s": time.time() - started,
        "preregistration_sha256": sha256(args.preregistration),
        "trainer_sha256": sha256(Path(__file__)),
        "wfp_module_sha256": sha256(ROOT / "saved_time_phase_operator_v4/wfp.py"),
        "validation_opened": False, "test_id_opened": False,
    }
    atomic_json(terminal, args.output_dir / "terminal.json")
    collection.close()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
