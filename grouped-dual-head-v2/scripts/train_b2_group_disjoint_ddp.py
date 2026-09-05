#!/usr/bin/env python3
"""Two-rank exact-global-batch DDP trainer for one B2 seed."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_b2_group_disjoint import (
    FAMILIES,
    _atomic_json,
    _baseline,
    _load_cache,
    _sha256,
)
from scripts.train_b2_snapshot_ic import FrameConditionedPropagator


class AnchoredDDPModule(nn.Module):
    """Expose the anchored path as the DDP-visible forward method."""

    def __init__(self, propagator: FrameConditionedPropagator) -> None:
        super().__init__()
        self.propagator = propagator

    def forward(self, base_seq, cond, initial_state):
        return self.propagator.forward_anchored(
            base_seq, cond, initial_state=initial_state
        )


def split_global_batch(indices: np.ndarray, world_size: int) -> list[np.ndarray]:
    """Split one global batch without duplication or omission."""
    if len(indices) < world_size:
        raise ValueError("global batch must contain at least one sample per rank")
    return [part.astype(np.int64, copy=False) for part in np.array_split(indices, world_size)]


def ddp_loss_scale(local_count: int, global_count: int, world_size: int) -> float:
    """Scale local means so DDP's rank average equals the global sample mean."""
    if local_count <= 0 or global_count <= 0 or world_size <= 0:
        raise ValueError("counts and world_size must be positive")
    return float(local_count * world_size / global_count)


def _slice_records(data: dict, limit: int | None) -> dict:
    if limit is None:
        return data
    if limit <= 0 or limit > len(data["families"]):
        raise ValueError("record limit is outside the cache census")
    return {
        "base": data["base"][:limit],
        "target": data["target"][:limit],
        "cond": data["cond"][:limit],
        "families": data["families"][:limit],
    }


def _gather_calibration(
    model: FrameConditionedPropagator,
    data: dict,
    initial: torch.Tensor,
    baseline_rel: torch.Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
    micro_records: int,
) -> dict | None:
    indices = np.arange(rank, len(data["families"]), world_size, dtype=np.int64)
    local_values = []
    with torch.no_grad():
        for lo in range(0, len(indices), micro_records):
            local_indices = torch.from_numpy(indices[lo:lo + micro_records].copy())
            prediction = model.forward_anchored(
                data["base"][local_indices].to(device),
                data["cond"][local_indices].to(device),
                initial_state=initial[local_indices].to(device),
            )[:, 8:]
            target = data["target"][local_indices, 8:].to(device)
            pred = prediction.reshape(len(local_indices), -1).double()
            truth = target.reshape(len(local_indices), -1).double()
            local_values.append(
                ((pred - truth).square().sum(1).sqrt()
                 / truth.square().sum(1).clamp_min(1.0e-16).sqrt()).float()
            )
    local_rel = torch.cat(local_values)
    local_indices_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
    counts = [torch.zeros(1, device=device, dtype=torch.long) for _ in range(world_size)]
    own_count = torch.tensor([len(indices)], device=device, dtype=torch.long)
    dist.all_gather(counts, own_count)
    count_values = [int(value.item()) for value in counts]
    if len(set(count_values)) != 1:
        raise RuntimeError("calibration shards must have equal sizes")
    gathered_rel = [torch.empty_like(local_rel) for _ in range(world_size)]
    gathered_indices = [torch.empty_like(local_indices_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_rel, local_rel)
    dist.all_gather(gathered_indices, local_indices_tensor)
    if rank != 0:
        return None
    rel = torch.empty(len(data["families"]), dtype=torch.float32)
    for shard_indices, shard_values in zip(gathered_indices, gathered_rel):
        rel[shard_indices.cpu()] = shard_values.cpu()
    return {
        "aggregate": float(rel.double().mean()),
        "maximum": float(rel.max()),
        "nonworse": int((rel.double() <= baseline_rel).sum()),
        "per_family": {
            family: float(rel[torch.from_numpy(data["families"] == family)].double().mean())
            for family in FAMILIES
        },
    }


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
    parser.add_argument("--global-micro-records", type=int, default=3)
    parser.add_argument("--calibration-micro-records", type=int, default=3)
    parser.add_argument("--maximum-epoch1-s", type=float, default=240.0)
    parser.add_argument("--fit-record-limit", type=int)
    parser.add_argument("--calibration-record-limit", type=int)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"expected exactly two ranks, observed {world_size}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    terminal = args.output_dir / "terminal.json"
    try:
        if rank == 0:
            if args.output_dir.exists():
                raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
            args.output_dir.mkdir(parents=True)
        dist.barrier()
        prereg = json.loads(args.preregistration.read_text())
        bindings = prereg["bindings"]
        if _sha256(Path(__file__)) != bindings["ddp_trainer_sha256"]:
            raise RuntimeError("DDP trainer binding drift")
        if _sha256(ROOT / "scripts/train_b2_group_disjoint.py") != bindings["base_trainer_sha256"]:
            raise RuntimeError("base trainer binding drift")
        fit_manifest = json.loads(args.fit_manifest.read_text())
        calibration_manifest = json.loads(args.calibration_manifest.read_text())
        for path, manifest in (
            (args.fit_manifest, fit_manifest),
            (args.calibration_manifest, calibration_manifest),
        ):
            if manifest.get("split") != "train":
                raise RuntimeError(f"manifest is not train-only: {path}")
            if manifest.get("validation_opened") or manifest.get("test_id_opened"):
                raise RuntimeError(f"sealed split flag is open: {path}")
            if _sha256(path) != bindings["manifest_sha256"][str(path)]:
                raise RuntimeError(f"manifest binding drift: {path}")
        fit_groups = {row["group_id"] for row in fit_manifest["records"]}
        cal_groups = {row["group_id"] for row in calibration_manifest["records"]}
        if fit_groups & cal_groups:
            raise RuntimeError("fit/calibration group overlap")

        fit = _slice_records(_load_cache(args.fit_cache, fit_manifest), args.fit_record_limit)
        calibration = _slice_records(
            _load_cache(args.calibration_cache, calibration_manifest),
            args.calibration_record_limit,
        )
        if len(fit["families"]) % args.global_micro_records:
            raise RuntimeError("fit census must be divisible by global microbatch")
        if len(calibration["families"]) % world_size:
            raise RuntimeError("calibration census must be divisible by world size")
        fit_initial = fit["target"][:, :8, 0]
        calibration_initial = calibration["target"][:, :8, 0]
        baseline, baseline_rel = _baseline(calibration)

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
        distributed = DistributedDataParallel(
            AnchoredDDPModule(model), device_ids=[local_rank]
        )
        steps_per_epoch = len(fit["families"]) // args.global_micro_records
        total_steps = args.epochs * steps_per_epoch
        optimizer = torch.optim.AdamW(
            [parameter for parameter in distributed.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=1.0e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * 0.01
        )
        identity = {
            "schema": "b2_group_disjoint_ddp_lane_identity_v1",
            "preregistration": str(args.preregistration),
            "preregistration_sha256": _sha256(args.preregistration),
            "ddp_trainer_sha256": _sha256(Path(__file__)),
            "base_trainer_sha256": _sha256(ROOT / "scripts/train_b2_group_disjoint.py"),
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
            "global_micro_records": args.global_micro_records,
            "rank_local_counts": [2, 1],
            "optimizer_steps_per_epoch": steps_per_epoch,
            "world_size": world_size,
            "gradient_semantics": "sample-weighted rank losses; DDP average equals global batch mean",
            "width": 64,
            "spectral_rank": 32,
            "ic_frames": 8,
            "baseline": baseline,
            "fit_record_limit": args.fit_record_limit,
            "calibration_record_limit": args.calibration_record_limit,
            "validation_opened": False,
            "test_id_opened": False,
        }
        if rank == 0:
            _atomic_json(identity, args.output_dir / "run_identity.json")

        order_rng = np.random.default_rng(args.seed + 101)
        best = None
        started = time.time()
        for epoch in range(1, args.epochs + 1):
            distributed.train()
            order = order_rng.permutation(len(fit["families"]))
            epoch_loss_sum = torch.zeros(1, device=device, dtype=torch.float64)
            epoch_sample_count = torch.zeros(1, device=device, dtype=torch.float64)
            for lo in range(0, len(order), args.global_micro_records):
                global_indices = order[lo:lo + args.global_micro_records]
                rank_indices = split_global_batch(global_indices, world_size)[rank]
                indices = torch.from_numpy(rank_indices.copy())
                optimizer.zero_grad(set_to_none=True)
                prediction = distributed(
                    fit["base"][indices].to(device),
                    fit["cond"][indices].to(device),
                    initial_state=fit_initial[indices].to(device),
                )[:, 8:]
                target = fit["target"][indices, 8:].to(device)
                pred = prediction.reshape(len(indices), -1)
                truth = target.reshape(len(indices), -1)
                sample_losses = (
                    (pred - truth).square().sum(1).clamp_min(0.0).sqrt()
                    / truth.square().sum(1).clamp_min(1.0e-16).sqrt()
                )
                local_mean = sample_losses.mean()
                if not torch.isfinite(local_mean):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}")
                scaled_loss = local_mean * ddp_loss_scale(
                    len(indices), len(global_indices), world_size
                )
                scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(distributed.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss_sum += sample_losses.detach().double().sum()
                epoch_sample_count += len(indices)
            dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(epoch_sample_count, op=dist.ReduceOp.SUM)
            distributed.eval()
            metrics = _gather_calibration(
                distributed.module.propagator,
                calibration,
                calibration_initial,
                baseline_rel,
                device,
                rank,
                world_size,
                args.calibration_micro_records,
            )
            elapsed_s = round(time.time() - started, 1)
            if rank == 0:
                row = {
                    "event": "epoch",
                    "seed": args.seed,
                    "epoch": epoch,
                    "train_loss": float((epoch_loss_sum / epoch_sample_count).item()),
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_s": elapsed_s,
                    **metrics,
                }
                with (args.output_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
                if best is None or metrics["aggregate"] < best["aggregate"]:
                    best = {"epoch": epoch, **metrics}
                    torch.save({
                        "model_state": distributed.module.propagator.state_dict(),
                        "epoch": epoch,
                        "identity": identity,
                        "metrics": metrics,
                    }, args.output_dir / "best.pt")
                    _atomic_json(best, args.output_dir / "best.json")
            stop_tensor = torch.tensor(
                [1 if rank == 0 and epoch == 1 and elapsed_s > args.maximum_epoch1_s else 0],
                device=device,
                dtype=torch.int32,
            )
            dist.broadcast(stop_tensor, src=0)
            if int(stop_tensor.item()):
                raise RuntimeError(
                    f"epoch-1 wall time {elapsed_s} exceeds "
                    f"{args.maximum_epoch1_s} s budget"
                )
            dist.barrier()
        if rank == 0:
            _atomic_json({
                "status": "complete",
                "seed": args.seed,
                "best": best,
                "baseline": baseline,
                "elapsed_s": round(time.time() - started, 1),
            }, terminal)
        dist.barrier()
        return 0
    except Exception as error:
        if rank == 0:
            import traceback
            _atomic_json({
                "status": "failed",
                "seed": args.seed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }, terminal)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
