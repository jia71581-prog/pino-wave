#!/usr/bin/env python3
"""Train B2 on fit groups and select checkpoints on disjoint calibration groups."""
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

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator

FAMILIES = ("uniform", "layered", "marmousi")


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


def _load_cache(cache_path: Path, manifest: dict) -> dict:
    with h5py.File(cache_path, "r", swmr=True) as cache:
        if cache.attrs["manifest_selection_sha256"] != manifest["selection_sha256"]:
            raise RuntimeError(f"cache manifest binding drift: {cache_path}")
        if cache.attrs.get("data_split", "") != "train":
            raise RuntimeError(f"cache is not train-only: {cache_path}")
        return {
            "base": torch.from_numpy(cache["base_seq"][:].astype(np.float32))[:, :, None],
            "target": torch.from_numpy(cache["target"][:].astype(np.float32))[:, :, None],
            "cond": torch.from_numpy(cache["cond"][:].astype(np.float32)),
            "families": cache["family"][:].astype(str),
        }


def calibration_gate(lanes: list[dict], gate: dict) -> dict:
    required_nonworse = int(gate["minimum_nonworse_records"])
    minimum_gain = float(gate["minimum_absolute_gain_vs_anchor"])
    results = []
    for lane in lanes:
        aggregate_pass = lane["aggregate"] <= lane["baseline"] - minimum_gain
        family_pass = all(
            lane["per_family"][family] < lane["baseline_per_family"][family]
            for family in FAMILIES
        )
        nonworse_pass = lane["nonworse"] >= required_nonworse
        results.append({
            "lane": lane["lane"],
            "aggregate_pass": aggregate_pass,
            "family_pass": family_pass,
            "nonworse_pass": nonworse_pass,
            "passed": aggregate_pass and family_pass and nonworse_pass,
        })
    return {"passed": len(results) == 2 and all(row["passed"] for row in results), "lanes": results}


def _baseline(data: dict) -> tuple[dict, torch.Tensor]:
    base = data["base"][:, 8:].reshape(len(data["families"]), -1).double()
    target = data["target"][:, 8:].reshape(len(data["families"]), -1).double()
    rel = (
        (base - target).square().sum(1).sqrt()
        / target.square().sum(1).clamp_min(1.0e-16).sqrt()
    )
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
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--fit-cache", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--micro-records", type=int, default=3)
    parser.add_argument("--maximum-epoch1-s", type=float, default=360.0)
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
            if manifest.get("split") != "train":
                raise RuntimeError(f"manifest is not train-only: {path}")
            if manifest.get("validation_opened") or manifest.get("test_id_opened"):
                raise RuntimeError(f"sealed split flag is open: {path}")
            if _sha256(path) != bindings["manifest_sha256"][str(path)]:
                raise RuntimeError(f"manifest binding drift: {path}")
        fit_groups = {row["group_id"] for row in fit_manifest["records"]}
        cal_groups = {row["group_id"] for row in cal_manifest["records"]}
        if fit_groups & cal_groups:
            raise RuntimeError("fit/calibration group overlap")
        fit = _load_cache(args.fit_cache, fit_manifest)
        calibration = _load_cache(args.calibration_cache, cal_manifest)
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
            values = []
            with torch.no_grad():
                for lo in range(0, len(calibration["families"]), args.micro_records):
                    hi = min(lo + args.micro_records, len(calibration["families"]))
                    prediction = model.forward_anchored(
                        calibration["base"][lo:hi].to(device),
                        calibration["cond"][lo:hi].to(device),
                        initial_state=cal_initial[lo:hi].to(device),
                    )[:, 8:]
                    target = calibration["target"][lo:hi, 8:].to(device)
                    pred = prediction.reshape(hi - lo, -1).double()
                    truth = target.reshape(hi - lo, -1).double()
                    values.append(
                        ((pred - truth).square().sum(1).sqrt()
                         / truth.square().sum(1).clamp_min(1.0e-16).sqrt()).cpu()
                    )
            rel = torch.cat(values)
            return {
                "aggregate": float(rel.mean()),
                "maximum": float(rel.max()),
                "nonworse": int((rel <= baseline_rel).sum()),
                "per_family": {
                    family: float(rel[torch.from_numpy(calibration["families"] == family)].mean())
                    for family in FAMILIES
                },
            }

        identity = {
            "schema": "b2_group_disjoint_lane_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "trainer_sha256": _sha256(Path(__file__)),
            "fit_manifest": str(args.fit_manifest),
            "fit_manifest_sha256": _sha256(args.fit_manifest),
            "calibration_manifest": str(args.calibration_manifest),
            "calibration_manifest_sha256": _sha256(args.calibration_manifest),
            "fit_cache": str(args.fit_cache),
            "fit_cache_sha256": _sha256(args.fit_cache),
            "calibration_cache": str(args.calibration_cache),
            "calibration_cache_sha256": _sha256(args.calibration_cache),
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "micro_records": args.micro_records,
            "width": 64,
            "spectral_rank": 32,
            "ic_frames": 8,
            "baseline": baseline,
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
            losses = []
            for lo in range(0, len(order), args.micro_records):
                indices = torch.from_numpy(order[lo:lo + args.micro_records].copy())
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_anchored(
                    fit["base"][indices].to(device),
                    fit["cond"][indices].to(device),
                    initial_state=fit_initial[indices].to(device),
                )[:, 8:]
                target = fit["target"][indices, 8:].to(device)
                pred = prediction.reshape(len(indices), -1)
                truth = target.reshape(len(indices), -1)
                loss = (
                    (pred - truth).square().sum(1).clamp_min(0.0).sqrt()
                    / truth.square().sum(1).clamp_min(1.0e-16).sqrt()
                ).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            model.eval()
            metrics = evaluate()
            row = {
                "event": "epoch",
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                **metrics,
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
            if epoch == 1 and row["elapsed_s"] > args.maximum_epoch1_s:
                raise RuntimeError(
                    f"epoch-1 wall time {row['elapsed_s']} exceeds "
                    f"{args.maximum_epoch1_s} s budget"
                )
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
            "seed": args.seed,
            "best": best,
            "baseline": baseline,
            "elapsed_s": round(time.time() - started, 1),
        }, terminal)
        return 0
    except Exception as error:
        import traceback
        _atomic_json({
            "status": "failed",
            "seed": args.seed,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, terminal)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
