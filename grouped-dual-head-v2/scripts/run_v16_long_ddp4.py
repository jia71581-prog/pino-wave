#!/usr/bin/env python3
"""v16 long stage on 4 GPUs: transcribed from the frozen runner's long branch.

Deviations from the frozen single-GPU semantics, all deliberate and recorded:
  - data parallel world=4 over the 3 prebuilt /dev/shm shards.  Rank r holds
    ONLY shard r % 3 (~9.7 GB); loading all shards on every rank cost 4x28 GB
    of anon memory plus page cache and got rank 1 SIGKILLed against the 240 GiB
    cgroup ceiling at 23:31 (memory.events max counter, first attempt logged in
    results/r16_dscp_v16_long_ddp4_run_20260826.log.oom_attempt).  Fit sampling
    is therefore PARTITIONED, not per-rank shuffling of the full 192: ranks 0
    and 3 share shard 0's partition and differ only by generator seed.
  - gradients all-reduced and averaged before the shared clip/step (v14 order),
    so the effective batch is 4 at unchanged lr.
  - evaluation is distributed: each rank scores the confirm records inside its
    own shard, results are all-gathered and deduplicated by sample_id on rank 0,
    which refuses unless the union covers every preregistered confirm record.
  - checkpoints, trajectory and terminal.json are rank-0 only; the monitor and
    stop flag are broadcast so all ranks take identical epoch decisions.
  - artifacts land in OUT/long_ddp4/, never in the frozen chain's OUT/long/.
The frozen runner is imported, never modified.
Launch with: torchrun --standalone --nproc_per_node=4 <this file>."""
import importlib.util
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
WORLD = 4
WORLD_SHARDS = 3


def load_my_shard(rank: int, log):
    """Load only shard rank % WORLD_SHARDS, staggered to flatten the page-cache spike."""
    shards = sorted(SHARD_DIR.glob("shard_*.pt"))
    if len(shards) != WORLD_SHARDS:
        raise v16.V16Refusal(f"expected {WORLD_SHARDS} shards, found {len(shards)}")
    mine = shards[rank % WORLD_SHARDS]
    time.sleep(12.0 * rank)
    payload = torch.load(mine, map_location="cpu", weights_only=False)
    log(f"[long_ddp4] rank {rank} holds {mine.name} "
        f"({len(payload['bundles'])} bundles, build {payload['build_s']:.0f}s)")
    return (payload["bundles"], payload["bases"], payload["scales"],
            float(payload["build_s"]), mine.name)


def gather_scores(head, scales, bases_device, my_confirm, device, expected_ids):
    """Score this rank's confirm bundles, gather, dedupe; rank 0 gets the full set."""
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


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is0 = rank == 0
    log = (lambda m: print(f"{v16.utc_now()} {m}", flush=True)) if is0 else (lambda m: None)

    plan = v16.STAGE_PLAN["long"]
    run = v16.OUT / "long_ddp4"
    if (run / "terminal.json").exists():
        raise v16.V16Refusal("long_ddp4 terminal already exists")
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
    counts = [None] * WORLD
    dist.all_gather_object(counts, {"rank": rank, "shard": shard_name,
                                    "fit": len(fit), "confirm": len(my_confirm)})
    if is0:
        log(f"[long_ddp4] partitions {counts}; global fit {len(fit_ids)} "
            f"confirm {len(eval_ids)}, world {WORLD} over {WORLD_SHARDS} shards")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    v16.configure_determinism(v16.SEED)
    head = v16.Wide128Head().to(device).float()
    if sum(p.numel() for p in head.parameters()) != v16.EXPECTED_PARAMETERS:
        raise v16.V16Refusal("head parameter count is not the preregistered 25266")
    for p in head.parameters():  # identical init by seed; broadcast for belt and braces
        dist.broadcast(p.data, src=0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=v16.LR, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(v16.SEED + rank)

    def order_for_epoch():
        return torch.randperm(len(fit), generator=generator).tolist()

    def one_update(update, order):
        bundle = fit[order[update % len(order)]]
        losses, *_ = v16.bundle_loss(head, scales, bases_device, bundle, device)
        total = losses["total"]
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
    control = torch.zeros(2, device=device)  # [monitor, stop_flag]
    # Budget is decided by rank 0 alone and broadcast at epoch boundaries: a per-rank
    # deadline check lets ranks leave the loop at different epochs and deadlocks the
    # survivors inside all_reduce.  Cost of this choice is at most one epoch of overshoot.
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
            trajectory.append({"epoch": epoch + 1,
                               "monitor_mean_calibration_loss": monitor,
                               "records": snap})
            checkpoint_bytes = v16.save_checkpoint(head, optimizer, run / "last.pt", {
                "stage": "long_ddp4", "epoch": epoch + 1, "seed": v16.SEED,
            })
            if monitor < best:
                best, best_epoch, bad = monitor, epoch + 1, 0
                v16.save_checkpoint(head, optimizer, run / "best.pt", {
                    "stage": "long_ddp4", "epoch": epoch + 1, "seed": v16.SEED,
                    "monitor": monitor,
                })
            else:
                bad += 1
            control[0] = monitor
            control[1] = 1.0 if bad >= plan["patience"] else 0.0
            log(f"[long_ddp4] epoch {epoch + 1} monitor {monitor:.6f} "
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
        gates = v16.eval_gates(plan, final_scores, peak_vram, checkpoint_bytes)
        all_passed = all(bool(v.get("passed")) for v in gates.values())
        final_status = status if status not in {"completed", "early_stop_patience"} \
            else ("passed" if all_passed else "fail_gate")
        parent_after = v16.sha256_file(v16.PARENT_PATH)
        terminal = {
            "schema": "r16_dscp_v16_long_ddp4_terminal_v1",
            "candidate": v16.CANDIDATE,
            "stage": "long_ddp4",
            "status": final_status,
            "run_status": status,
            "preregistration_sha256": v16.PREREG_SHA256,
            "authorization": "lead directives 2026-08-26: direct long launch + 4-card training; see results/r16_dscp_v16_long_launch_authorization_20260826.json(.note)",
            "ddp_deviations": {
                "world_size": WORLD,
                "shards_held_per_rank": 1,
                "shard_assignment": "rank % 3",
                "partitions": counts,
                "sampling": "PARTITIONED: each rank shuffles only its own shard's fit records, generator seed SEED+rank; ranks 0 and 3 share shard 0's partition",
                "evaluation": "distributed per-shard scoring, all_gather_object + dedupe, coverage of every preregistered confirm record enforced",
                "first_attempt": "all-shards-per-rank variant SIGKILLed by the 240GiB cgroup ceiling at 2026-08-26T15:31Z, no artifacts written",
                "budget_check": "rank 0 decides at epoch boundaries and broadcasts; overshoot bounded by one epoch",
                "effective_batch": 4,
                "lr": "unchanged 3e-3",
                "gradient_sync": "all_reduce mean before shared clip 1.0 and step (v14 order)",
                "cold_start": True,
            },
            "gates": gates,
            "metrics": {
                "final_records": final_scores, "trajectory": trajectory,
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
            "disclaimers": {"stage": v16.DISCLAIMERS["long"],
                            "correction_nature": v16.DISCLAIMERS["correction_nature"]},
            "bindings": bindings,
            "completed_utc": v16.utc_now(),
        }
        v16.atomic_json(terminal, run / "terminal.json")
        log(f"[long_ddp4] {final_status} wall {terminal['resources']['wall_s']:.0f}s "
            f"peak {peak_vram / 2**30:.2f}GiB")
        if parent_after != v16.PARENT_SHA256:
            raise v16.V16Refusal("parent checkpoint changed during the stage")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
