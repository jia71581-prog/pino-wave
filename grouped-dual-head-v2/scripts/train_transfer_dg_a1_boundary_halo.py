#!/usr/bin/env python3
"""Transfer DG A1: paired matched-halo continuation from the P1b parents."""
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

from scripts.train_b2_snapshot_ic import FrameConditionedPropagator
from scripts.train_pa_cora_p1b_curriculum import (
    ThreeWindowCache,
    _load_holdout,
    evaluate_phase,
    robust_multiphase_loss,
)


SLOT_WEIGHTS = (0.60, 0.25, 0.15)


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


def weighted_multiphase_score(phases: list[dict]) -> float:
    if len(phases) != len(SLOT_WEIGHTS):
        raise ValueError("Transfer DG A1 requires exactly three phase metrics")
    return float(
        sum(weight * float(row["aggregate"]) for weight, row in zip(SLOT_WEIGHTS, phases))
    )


def halo_radius_for_arm(arm: str) -> int:
    if arm == "halo":
        return 20
    if arm == "control":
        return 0
    raise ValueError(f"unknown A1 arm: {arm}")


def _model(*, halo_radius: int, device: torch.device) -> FrameConditionedPropagator:
    return FrameConditionedPropagator(
        state_channels=8,
        cond_channels=7,
        width=64,
        spectral_rank=32,
        modes=24,
        depth=4,
        gate_init=1.0,
        activation_checkpointing=True,
        boundary_halo_radius=int(halo_radius),
        hard_free_surface=True,
    ).to(device)


def _verify_binding(paths: list[Path], expected: dict[str, str], label: str) -> None:
    for path in paths:
        if _sha256(path) != expected[str(path)]:
            raise RuntimeError(f"{label} binding drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("halo", "control"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-manifest", type=Path, action="append", required=True)
    parser.add_argument("--fit-cache", type=Path, action="append", required=True)
    parser.add_argument("--holdout-manifest", type=Path, action="append", required=True)
    parser.add_argument("--holdout-cache", type=Path, action="append", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--micro-records", type=int, default=4)
    args = parser.parse_args()
    if len(args.fit_manifest) != 3 or len(args.fit_cache) != 3:
        raise ValueError("Transfer DG A1 requires three fit windows")
    if len(args.holdout_manifest) != 3 or len(args.holdout_cache) != 3:
        raise ValueError("Transfer DG A1 requires three holdout windows")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    terminal = args.output_dir / "terminal.json"
    fit = None
    try:
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if prereg["method"] != "Transfer DG" or prereg["stage"] != "A1_boundary_halo":
            raise RuntimeError("wrong Transfer DG preregistration")
        if _sha256(Path(__file__)) != bindings["trainer_sha256"]:
            raise RuntimeError("trainer binding drift")
        _verify_binding(
            [Path(path) for path in bindings["code_sha256"]],
            bindings["code_sha256"],
            "dependency code",
        )
        if _sha256(args.checkpoint) != bindings["checkpoint_sha256"][str(args.checkpoint)]:
            raise RuntimeError("checkpoint binding drift")
        _verify_binding(args.fit_manifest, bindings["fit_manifest_sha256"], "fit manifest")
        _verify_binding(args.fit_cache, bindings["fit_cache_sha256"], "fit cache")
        _verify_binding(
            args.holdout_manifest,
            bindings["holdout_manifest_sha256"],
            "holdout manifest",
        )
        _verify_binding(args.holdout_cache, bindings["holdout_cache_sha256"], "holdout cache")
        fit_manifests = [json.loads(path.read_text()) for path in args.fit_manifest]
        holdout_manifests = [json.loads(path.read_text()) for path in args.holdout_manifest]
        for manifest in fit_manifests + holdout_manifests:
            if (
                manifest.get("split") != "train"
                or manifest.get("validation_opened")
                or manifest.get("test_id_opened")
            ):
                raise RuntimeError("Transfer DG A1 accepts sealed train manifests only")

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        parent_identity = checkpoint.get("identity", {})
        if (
            parent_identity.get("method") != "PA-CORA"
            or parent_identity.get("stage") != "P1b_stable_multiphase_continuation"
            or parent_identity.get("validation_opened")
            or parent_identity.get("test_id_opened")
        ):
            raise RuntimeError("checkpoint is not a sealed P1b parent")
        device = torch.device("cuda")
        holdouts = _load_holdout(args.holdout_cache, holdout_manifests)

        parent = _model(halo_radius=0, device=device)
        parent.load_state_dict(checkpoint["model_state"], strict=True)
        parent.gate.requires_grad_(False)
        parent_phases = [
            evaluate_phase(parent, data, device, args.micro_records) for data in holdouts
        ]
        parent_score = weighted_multiphase_score(parent_phases)
        del parent
        torch.cuda.empty_cache()

        radius = halo_radius_for_arm(args.arm)
        model = _model(halo_radius=radius, device=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.gate.requires_grad_(False)
        initial_phases = [
            evaluate_phase(model, data, device, args.micro_records) for data in holdouts
        ]
        initial_score = weighted_multiphase_score(initial_phases)

        fit = ThreeWindowCache(args.fit_cache, fit_manifests)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed + 401)
        total_steps = int(args.epochs) * int(args.steps_per_epoch)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=1.0e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.learning_rate * 0.01,
        )
        identity = {
            "schema": "transfer_dg_a1_lane_identity_v1",
            "method": "Transfer DG",
            "stage": "A1_boundary_halo",
            "arm": args.arm,
            "seed": args.seed,
            "parent_checkpoint": str(args.checkpoint),
            "parent_checkpoint_sha256": _sha256(args.checkpoint),
            "parent_epoch": int(checkpoint.get("epoch", -1)),
            "hard_free_surface": True,
            "boundary_halo_radius": radius,
            "slot_weights": list(SLOT_WEIGHTS),
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "total_steps": total_steps,
            "micro_records": args.micro_records,
            "learning_rate": args.learning_rate,
            "denominator_floor_fraction": 0.25,
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "validation_opened": False,
            "test_id_opened": False,
        }
        _atomic_json(identity, args.output_dir / "run_identity.json")
        _atomic_json(
            {
                "parent_phases": parent_phases,
                "parent_score": parent_score,
                "candidate_phases": initial_phases,
                "candidate_score": initial_score,
            },
            args.output_dir / "initial_holdout.json",
        )
        best = {"epoch": 0, "score": initial_score, "phases": initial_phases}
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": 0,
                "identity": identity,
                "metrics": best,
            },
            args.output_dir / "best.pt",
        )
        _atomic_json(best, args.output_dir / "best.json")

        started = time.time()
        stratum_offset = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, components, slot_counts = [], [], [0, 0, 0]
            for _ in range(args.steps_per_epoch):
                slot = int(rng.choice(3, p=np.asarray(SLOT_WEIGHTS)))
                indices = fit.balanced_indices(rng, args.micro_records, stratum_offset)
                stratum_offset += args.micro_records
                anchor, target, conditioning, onset_norm = fit.batch(slot, indices)
                anchor, target = anchor.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model.forward_anchored(
                    anchor,
                    conditioning.to(device),
                    initial_state=target[:, :8, 0],
                )
                loss, row = robust_multiphase_loss(
                    prediction[:, 8:], target[:, 8:], onset_norm.to(device)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite Transfer DG A1 loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
                components.append({key: float(value) for key, value in row.items()})
                slot_counts[slot] += len(indices)

            phases = [
                evaluate_phase(model, data, device, args.micro_records) for data in holdouts
            ]
            score = weighted_multiphase_score(phases)
            event = {
                "event": "epoch",
                "arm": args.arm,
                "seed": args.seed,
                "epoch": epoch,
                "score": score,
                "parent_score": parent_score,
                "slot_weights": SLOT_WEIGHTS,
                "slot_sample_counts": slot_counts,
                "train_loss": float(np.mean(losses)),
                "train_components": {
                    key: float(np.mean([row[key] for row in components]))
                    for key in components[0]
                },
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": round(time.time() - started, 1),
                "phases": phases,
            }
            with (args.output_dir / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            if score < best["score"]:
                best = {"epoch": epoch, "score": score, "phases": phases}
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "epoch": epoch,
                        "identity": identity,
                        "metrics": best,
                    },
                    args.output_dir / "best.pt",
                )
                _atomic_json(best, args.output_dir / "best.json")

        passed = best["score"] < parent_score
        _atomic_json(
            {
                "status": "passed" if passed else "rejected",
                "arm": args.arm,
                "seed": args.seed,
                "best_epoch": best["epoch"],
                "parent_score": parent_score,
                "initial_candidate_score": initial_score,
                "best_score": best["score"],
                "relative_gain_vs_parent": (parent_score - best["score"])
                / max(parent_score, 1.0e-16),
                "elapsed_s": round(time.time() - started, 1),
                "validation_opened": False,
                "test_id_opened": False,
            },
            terminal,
        )
        fit.close()
        return 0
    except Exception as error:
        import traceback

        if fit is not None:
            fit.close()
        _atomic_json(
            {
                "status": "failed",
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
            terminal,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
