#!/usr/bin/env python3
"""v17 long stage on 4 GPUs: v16 long_ddp4 transcription + longtrain plan,
optional no-harm hinge, and the v17 nonworse tolerance gates.

Everything numeric (plan, gates, hinge) is read from the FROZEN v17
preregistration JSON; this file contains no tunable constants, so freezing
the prereg freezes the run.  Deviations from the frozen v16 single-GPU
semantics are exactly the recorded long_ddp4 ones (rank%3 shard partition,
all_reduce-mean before shared clip/step, distributed dedup eval, rank-0
artifacts).  Per the lead directive of 2026-08-27 the shard partition scheme
is kept as-is; no memory-layout work.
Launch with: torchrun --standalone --nproc_per_node=4 <this file>."""
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_r16_dscp_v16", ROOT / "scripts/train_r16_dscp_v16.py"
)
v16 = importlib.util.module_from_spec(spec)
sys.modules["train_r16_dscp_v16"] = v16
spec.loader.exec_module(v16)

SHARD_DIR = Path("/dev/shm/v16_long_bundles")
PREREG_V17 = ROOT / "results/r16_dscp_v17_preregistration_20260827.json"
OUT_V17 = ROOT / "results/r16_dscp_v17"
WORLD = 4
WORLD_SHARDS = 3


def load_my_shard(rank: int, log):
    shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    if len(shards) != WORLD_SHARDS:
        raise v16.V16Refusal(f"expected {WORLD_SHARDS} shards, found {len(shards)}")
    mine = shards[rank % WORLD_SHARDS]
    time.sleep(12.0 * rank)
    payload = torch.load(mine, map_location="cpu", weights_only=False)
    log(f"[v17] rank {rank} holds {mine.name} ({len(payload['bundles'])} bundles)")
    return (payload["bundles"], payload["bases"], payload["scales"],
            float(payload["build_s"]), mine.name)


def gather_scores(head, scales, bases_device, my_confirm, device, expected_ids):
    local = [v16.score_bundle(head, scales, bases_device, b, device) for b in my_confirm]
    buckets = [None] * WORLD
    dist.all_gather_object(buckets, local)
    merged = {}
    for chunk in buckets:
        for row in chunk:
            merged[row["sample_id"]] = row
    missing = [s for s in expected_ids if s not in merged]
    if missing:
        raise v16.V16Refusal(f"confirm coverage incomplete, missing {missing[:5]}")
    return [merged[s] for s in expected_ids]


def v17_eval_gates(gates_cfg, scores, peak_vram, checkpoint_bytes):
    """v16 joint/family/finite/vram/checkpoint logic + the v17 nonworse
    tolerance redefinition (gain_iii >= -tolerance for ALL records, plus a
    worst-harm floor), all thresholds from the frozen prereg."""
    joint = sum(r["gain_iii"] for r in scores) / max(len(scores), 1)
    by_family = {}
    for row in scores:
        by_family.setdefault(row["family"], []).append(row["gain_iii"])
    family_means = {k: sum(v) / len(v) for k, v in by_family.items()}
    tol = float(gates_cfg["nonworse_tolerance"])
    within = sum(1 for r in scores if r["gain_iii"] >= -tol)
    worst = min(r["gain_iii"] for r in scores)
    finite = all(
        math.isfinite(r[k]) for r in scores
        for k in ("loss", "aggregate_rel_l2", "parent_rel_l2")
    )
    return {
        "joint_improvement": {
            "value": joint, "threshold": gates_cfg["joint_improvement"],
            "passed": joint >= gates_cfg["joint_improvement"],
        },
        "per_family_improvement": {
            "value": family_means, "threshold": gates_cfg["per_family_improvement"],
            "passed": len(family_means) == 3
            and all(v >= gates_cfg["per_family_improvement"] for v in family_means.values()),
        },
        "nonworse_within_tolerance": {
            "value": within, "tolerance": tol,
            "threshold": gates_cfg["nonworse_min_count"],
            "passed": within >= gates_cfg["nonworse_min_count"],
        },
        "worst_harm": {
            "value": worst, "floor": gates_cfg["worst_harm_floor"],
            "passed": worst >= gates_cfg["worst_harm_floor"],
        },
        "finite": {"value": finite, "passed": finite},
        "vram": {
            "value_bytes": peak_vram, "limit_bytes": v16.TRAIN_VRAM_LIMIT_BYTES,
            "passed": 0 < peak_vram <= v16.TRAIN_VRAM_LIMIT_BYTES,
        },
        "checkpoint": {
            "value_bytes": checkpoint_bytes, "limit_bytes": v16.CHECKPOINT_LIMIT_BYTES,
            "passed": 0 < checkpoint_bytes <= v16.CHECKPOINT_LIMIT_BYTES,
        },
    }


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is0 = rank == 0
    log = (lambda m: print(f"{v16.utc_now()} {m}", flush=True)) if is0 else (lambda m: None)

    prereg = json.loads(PREREG_V17.read_text())
    prereg_sha = v16.sha256_file(PREREG_V17)
    plan = prereg["long_ddp4"]["plan"]
    gates_cfg = prereg["long_ddp4"]["gates"]
    hinge = prereg["long_ddp4"]["hinge"]  # {"enabled": bool, "lam": float, "margin": float}

    run = OUT_V17 / "long_ddp4"
    if (run / "terminal.json").exists():
        raise v16.V16Refusal("v17 long_ddp4 terminal already exists")
    started = time.monotonic()
    deadline = started + plan["wall_s"]
    bindings = v16.verify_bindings() if is0 else None

    fit_ids = v16.role_sample_ids(plan["fit_role"])
    eval_ids = v16.role_sample_ids(plan["eval_role"])
    fit_set, eval_set = set(fit_ids), set(eval_ids)
    log_all = lambda m: print(f"{v16.utc_now()} {m}", flush=True)
    my_bundles, bases, scales, cache_build_s, shard_name = load_my_shard(rank, log_all)
    fit = [b for b in my_bundles if b.sample_id in fit_set]
    my_confirm = [b for b in my_bundles if b.sample_id in eval_set]
    if not fit:
        raise v16.V16Refusal(f"rank {rank} shard {shard_name} carries no fit records")
    bases_device = bases.to(device).float()
    parent_rel = {
        b.sample_id: float(v16.relative_l2(
            b.parent_full[0, b.k1 + 1:].float(), b.truth_future))
        for b in fit
    } if hinge["enabled"] else {}
    counts = [None] * WORLD
    dist.all_gather_object(counts, {"rank": rank, "shard": shard_name,
                                    "fit": len(fit), "confirm": len(my_confirm)})
    if is0:
        log(f"[v17] partitions {counts}; global fit {len(fit_ids)} confirm {len(eval_ids)}; "
            f"prereg {prereg_sha[:12]} hinge {hinge}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    v16.configure_determinism(v16.SEED)
    head = v16.Wide128Head().to(device).float()
    if sum(p.numel() for p in head.parameters()) != v16.EXPECTED_PARAMETERS:
        raise v16.V16Refusal("head parameter count is not the preregistered 25266")
    for p in head.parameters():
        dist.broadcast(p.data, src=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=v16.LR, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(v16.SEED + rank)

    def order_for_epoch():
        return torch.randperm(len(fit), generator=generator).tolist()

    def one_update(update, order):
        bundle = fit[order[update % len(order)]]
        losses, adapted, _parent_full, truth_future = v16.bundle_loss(
            head, scales, bases_device, bundle, device
        )
        total = losses["total"]
        if hinge["enabled"]:
            cand_rel = torch.linalg.vector_norm(
                adapted[0, bundle.k1 + 1:].float() - truth_future.float()
            ) / torch.linalg.vector_norm(truth_future.float()).clamp_min(1e-30)
            ratio = cand_rel / parent_rel[bundle.sample_id]
            total = total + float(hinge["lam"]) * torch.relu(
                ratio - (1.0 - float(hinge["margin"]))
            )
        if not torch.isfinite(total):
            raise v16.V16Refusal(f"non-finite loss at update {update} ({bundle.sample_id})")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise v16.V16Refusal("nonfinite or absent gradient")
        for p in head.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(float(WORLD))
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        return float(total)

    best, bad, best_epoch = float("inf"), 0, -1
    trajectory, status = [], "completed"
    checkpoint_bytes = 0
    control = torch.zeros(2, device=device)
    for epoch in range(plan["max_epochs"]):
        if is0:
            control[1] = 1.0 if time.monotonic() > deadline else 0.0
        dist.broadcast(control, src=0)
        if control[1].item() >= 1.0:
            status = "budget_limit"
            break
        order = order_for_epoch()
        for update in range(plan["updates_per_epoch"]):
            if update % len(fit) == 0 and update:
                order = order_for_epoch()
            one_update(update, order)
        snap = gather_scores(head, scales, bases_device, my_confirm, device, eval_ids)
        if is0:
            monitor = sum(r["loss"] for r in snap) / len(snap)
            joint = sum(r["gain_iii"] for r in snap) / len(snap)
            trajectory.append({"epoch": epoch + 1,
                               "monitor_mean_calibration_loss": monitor,
                               "joint_mean_gain": joint,
                               "records": snap})
            checkpoint_bytes = v16.save_checkpoint(head, optimizer, run / "last.pt", {
                "stage": "v17_long_ddp4", "epoch": epoch + 1, "seed": v16.SEED,
            })
            if monitor < best:
                best, best_epoch, bad = monitor, epoch + 1, 0
                v16.save_checkpoint(head, optimizer, run / "best.pt", {
                    "stage": "v17_long_ddp4", "epoch": epoch + 1, "seed": v16.SEED,
                    "monitor": monitor,
                })
            else:
                bad += 1
            control[0] = monitor
            control[1] = 1.0 if bad >= plan["patience"] else 0.0
            log(f"[v17] epoch {epoch + 1} monitor {monitor:.6f} joint {joint:+.4f} "
                f"best {best:.6f}@{best_epoch} bad {bad}")
        dist.broadcast(control, src=0)
        if control[1].item() >= 1.0:
            status = "early_stop_patience"
            break

    reload_best = torch.zeros(1, device=device)
    if is0:
        reload_best[0] = 1.0 if (run / "best.pt").exists() else 0.0
    dist.broadcast(reload_best, src=0)
    if reload_best[0].item() >= 1.0:
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        head.load_state_dict(payload["model_state"])
        best_path = run / "best.pt"
    else:
        best_path = run / "last.pt"
    final_scores = gather_scores(head, scales, bases_device, my_confirm, device, eval_ids)
    if is0:
        peak_vram = int(torch.cuda.max_memory_reserved(device))
        gates = v17_eval_gates(gates_cfg, final_scores, peak_vram, checkpoint_bytes)
        all_passed = all(bool(v.get("passed")) for v in gates.values())
        final_status = status if status not in {"completed", "early_stop_patience"} \
            else ("passed" if all_passed else "fail_gate")
        parent_after = v16.sha256_file(v16.PARENT_PATH)
        terminal = {
            "schema": "r16_dscp_v17_long_ddp4_terminal_v1",
            "candidate": prereg["candidate"],
            "stage": "long_ddp4",
            "status": final_status,
            "run_status": status,
            "preregistration_sha256": prereg_sha,
            "v16_prereg_sha256": v16.PREREG_SHA256,
            "authorization": prereg["authorization"],
            "hinge": hinge,
            "ddp_deviations": "identical to v16 long_ddp4 terminal (rank%3 shard partition, "
                              "all_reduce mean, distributed dedup eval, rank-0 artifacts); "
                              "shard scheme kept per lead directive 2026-08-27",
            "gates": gates,
            "metrics": {
                "final_records": final_scores,
                "trajectory": trajectory,
                "fit_records": len(fit_ids), "eval_records": len(eval_ids),
                "cache_build_s": cache_build_s,
            },
            "resources": {
                "wall_s": time.monotonic() - started,
                "peak_cuda_reserved_bytes": peak_vram,
                "checkpoint_bytes": checkpoint_bytes,
                "best_checkpoint": str(best_path),
            },
            "parent": {"sha256_before": v16.PARENT_SHA256, "sha256_after": parent_after,
                        "match": parent_after == v16.PARENT_SHA256, "writes": 0},
            "sealed_splits_opened": False,
            "thresholds_moved_toward_measured_values": False,
            "threshold_provenance": prereg["threshold_provenance"],
            "disclaimers": {"stage": v16.DISCLAIMERS["long"],
                            "correction_nature": v16.DISCLAIMERS["correction_nature"]},
            "bindings": bindings,
            "completed_utc": v16.utc_now(),
        }
        v16.atomic_json(terminal, run / "terminal.json")
        log(f"[v17] {final_status} wall {terminal['resources']['wall_s']:.0f}s "
            f"peak {peak_vram / 2**30:.2f}GiB epochs {len(trajectory)}")
        if parent_after != v16.PARENT_SHA256:
            raise v16.V16Refusal("parent checkpoint changed during the stage")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
