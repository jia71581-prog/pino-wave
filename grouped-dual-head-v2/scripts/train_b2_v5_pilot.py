#!/usr/bin/env python3
"""Train one arm of the causal/source/loss B2-v5 screening pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.b2_v5_components import (
    b2_v5_loss,
    per_record_relative_l2,
    temporal_band_relative_l2,
)
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator

ARMS = ("control", "source_cond", "physics_cond", "nonworse_hinge", "spectral")
FAMILIES = ("uniform", "layered", "marmousi")
TIME_SLICES = {"early": (0, 19), "middle": (19, 38), "late": (38, 56)}


def _sha256(path: Path) -> str:
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


def conditioning_key_for_arm(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    if arm == "source_cond":
        return "cond_source"
    if arm == "physics_cond":
        return "cond_physics"
    return "cond"


def _load_cache(path: Path, manifest: dict, arm: str) -> dict:
    key = conditioning_key_for_arm(arm)
    with h5py.File(path, "r", swmr=True) as cache:
        if cache.attrs.get("schema", "") != "b2_v5_causal_cache_v1":
            raise RuntimeError(f"cache is not B2-v5 causal: {path}")
        if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
            raise RuntimeError(f"cache manifest drift: {path}")
        return {
            "base": torch.from_numpy(cache["base_seq"][:].astype(np.float32))[:, :, None],
            "target": torch.from_numpy(cache["target"][:].astype(np.float32))[:, :, None],
            "cond": torch.from_numpy(cache[key][:].astype(np.float32)),
            "families": cache["family"][:].astype(str),
        }


def _baseline(data: dict) -> tuple[dict, torch.Tensor]:
    anchor = data["base"][:, 8:]
    target = data["target"][:, 8:]
    rel = per_record_relative_l2(anchor.double(), target.double()).cpu()
    return {
        "aggregate": float(rel.mean()),
        "maximum": float(rel.max()),
        "per_family": {
            family: float(rel[torch.from_numpy(data["families"] == family)].mean())
            for family in FAMILIES
        },
    }, rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=372)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--micro-records", type=int, default=3)
    parser.add_argument("--nonworse-weight", type=float, default=0.3)
    parser.add_argument("--spectral-weight", type=float, default=0.1)
    parser.add_argument("--spectral-energy-floor-fraction", type=float, default=0.005)
    parser.add_argument("--maximum-epoch1-s", type=float, default=240.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if _sha256(Path(__file__)) != bindings["trainer_sha256"]:
            raise RuntimeError("trainer binding drift")
        fit_manifest = json.loads(args.fit_manifest.read_text())
        cal_manifest = json.loads(args.calibration_manifest.read_text())
        for path, manifest in (
            (args.fit_manifest, fit_manifest),
            (args.calibration_manifest, cal_manifest),
        ):
            if manifest.get("future_truth_opened_for_window_selection"):
                raise RuntimeError("future-derived window selection is forbidden")
            if _sha256(path) != bindings["manifest_sha256"][str(path)]:
                raise RuntimeError(f"manifest binding drift: {path}")
        if {row["group_id"] for row in fit_manifest["records"]} & {
            row["group_id"] for row in cal_manifest["records"]
        }:
            raise RuntimeError("fit/calibration group overlap")
        fit = _load_cache(args.fit_cache, fit_manifest, args.arm)
        calibration = _load_cache(args.calibration_cache, cal_manifest, args.arm)
        fit_initial = fit["target"][:, :8, 0]
        cal_initial = calibration["target"][:, :8, 0]
        baseline, baseline_rel = _baseline(calibration)
        device = torch.device("cuda")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = FrameConditionedPropagator(
            state_channels=8,
            cond_channels=fit["cond"].shape[1],
            width=64,
            spectral_rank=32,
            modes=24,
            depth=4,
            gate_init=1.0,
            activation_checkpointing=True,
        ).to(device)
        torch.nn.init.zeros_(model.decoder[-1].weight)
        torch.nn.init.zeros_(model.decoder[-1].bias)
        model.gate.requires_grad_(False)
        steps_per_epoch = (len(fit["families"]) + args.micro_records - 1) // args.micro_records
        total_steps = args.epochs * steps_per_epoch
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=1.0e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.01
        )

        def evaluate() -> dict:
            rel_values = []
            correction_values = []
            temporal_values = {name: [] for name in TIME_SLICES}
            frequency_values = []
            with torch.no_grad():
                for lo in range(0, len(calibration["families"]), args.micro_records):
                    hi = min(lo + args.micro_records, len(calibration["families"]))
                    anchor = calibration["base"][lo:hi].to(device)
                    target = calibration["target"][lo:hi].to(device)
                    prediction = model.forward_anchored(
                        anchor,
                        calibration["cond"][lo:hi].to(device),
                        initial_state=cal_initial[lo:hi].to(device),
                    )
                    pred_future = prediction[:, 8:]
                    target_future = target[:, 8:]
                    anchor_future = anchor[:, 8:]
                    rel_values.append(
                        per_record_relative_l2(pred_future.double(), target_future.double()).cpu()
                    )
                    correction_values.append(
                        per_record_relative_l2(pred_future.double(), anchor_future.double()).cpu()
                    )
                    for name, (start, stop) in TIME_SLICES.items():
                        temporal_values[name].append(
                            per_record_relative_l2(
                                pred_future[:, start:stop].double(),
                                target_future[:, start:stop].double(),
                            ).cpu()
                        )
                    frequency_values.append(
                        temporal_band_relative_l2(
                            pred_future,
                            target_future,
                            energy_floor_fraction=args.spectral_energy_floor_fraction,
                        ).cpu()
                    )
            rel = torch.cat(rel_values)
            frequency = torch.cat(frequency_values)
            return {
                "aggregate": float(rel.mean()),
                "maximum": float(rel.max()),
                "nonworse": int((rel.double() <= baseline_rel).sum()),
                "correction_energy": float(torch.cat(correction_values).mean()),
                "per_family": {
                    family: float(rel[torch.from_numpy(calibration["families"] == family)].mean())
                    for family in FAMILIES
                },
                "temporal": {
                    name: float(torch.cat(values).mean())
                    for name, values in temporal_values.items()
                },
                "frequency": {
                    name: float(frequency[:, index].mean())
                    for index, name in enumerate(("low", "middle", "high"))
                },
            }

        identity = {
            "schema": "b2_v5_pilot_lane_identity_v1",
            "arm": args.arm,
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "trainer_sha256": _sha256(Path(__file__)),
            "fit_manifest_sha256": _sha256(args.fit_manifest),
            "calibration_manifest_sha256": _sha256(args.calibration_manifest),
            "fit_cache_sha256": _sha256(args.fit_cache),
            "calibration_cache_sha256": _sha256(args.calibration_cache),
            "conditioning_key": conditioning_key_for_arm(args.arm),
            "cond_channels": int(fit["cond"].shape[1]),
            "seed": args.seed,
            "epochs": args.epochs,
            "nonworse_weight": args.nonworse_weight,
            "spectral_weight": args.spectral_weight,
            "spectral_energy_floor_fraction": args.spectral_energy_floor_fraction,
            "baseline": baseline,
            "future_truth_opened_for_window_selection": False,
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        order_rng = np.random.default_rng(args.seed + 101)
        best = None
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = order_rng.permutation(len(fit["families"]))
            component_rows = []
            losses = []
            for lo in range(0, len(order), args.micro_records):
                indices = torch.from_numpy(order[lo:lo + args.micro_records].copy())
                optimizer.zero_grad(set_to_none=True)
                anchor = fit["base"][indices].to(device)
                target = fit["target"][indices].to(device)
                prediction = model.forward_anchored(
                    anchor,
                    fit["cond"][indices].to(device),
                    initial_state=fit_initial[indices].to(device),
                )
                loss, components = b2_v5_loss(
                    prediction[:, 8:],
                    target[:, 8:],
                    anchor[:, 8:],
                    arm=args.arm,
                    nonworse_weight=args.nonworse_weight,
                    spectral_weight=args.spectral_weight,
                    spectral_energy_floor_fraction=args.spectral_energy_floor_fraction,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                component_rows.append({key: float(value) for key, value in components.items()})
            model.eval()
            metrics = evaluate()
            row = {
                "event": "epoch",
                "arm": args.arm,
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_components": {
                    key: float(np.mean([value[key] for value in component_rows]))
                    for key in component_rows[0]
                },
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                **metrics,
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if epoch == 1 and row["elapsed_s"] > args.maximum_epoch1_s:
                raise RuntimeError("epoch-1 wall time budget exceeded")
            if best is None or metrics["aggregate"] < best["aggregate"]:
                best = {"epoch": epoch, **metrics}
                torch.save({
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "identity": identity,
                    "metrics": metrics,
                }, args.output_dir / "best.pt")
                _atomic_json(best, args.output_dir / "best.json")
        _atomic_json({
            "status": "complete",
            "arm": args.arm,
            "best": best,
            "baseline": baseline,
            "elapsed_s": round(time.time() - started, 1),
        }, terminal)
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "arm": args.arm,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
