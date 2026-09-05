#!/usr/bin/env python3
"""v18: all routed-family train data (fit2=2165), two arms, 4-GPU DDP.

Prereg: results/r16_dscp_v18_preregistration_20260827.json (all numerics read
from it at runtime).  Bundles are built IN-PROCESS per rank (no shard files:
145GB fits neither /dev/shm nor disk) and stored fp16 in anonymous RAM; all
compute casts to fp32 at use.  The build path is a transcription of the frozen
v16 build_bundles with exactly two declared deviations: fp16 storage and
abstain-skip (fit2 was never routing-screened; skips are gathered into the
terminal).  Training/eval/gates are the v17 runner's semantics; both arms run
sequentially in the same processes over the same held bundles, cold start each.
Launch: torchrun --standalone --nproc_per_node=4 <this file>."""
import datetime
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

PREREG_V18 = ROOT / "results/r16_dscp_v18_preregistration_20260827.json"
OUT_V18 = ROOT / "results/r16_dscp_v18"
WORLD = 4
#: fields are ~1e-8 physical pressure, below the fp16 subnormal floor; storage shifts
#: the exponent by an exact power of two (zero mantissa cost) and unshifts at load.
FP16_FIELD_SCALE = 2.0 ** 30
#: per-rank byte cap for parking bundle tensors in VRAM instead of CPU RAM.
#: The platform watchdog SIGKILLs the container at memory.current ~237GiB and
#: ~124GiB of file-cache charge is pinned dead weight (unevictable from inside,
#: attempts 1-3 all died); parking 13GiB/GPU keeps terminal CPU anon ~93GiB.
#: Bit-identical tensors, only the storage device changes (v18.1 amendment).
VRAM_PARK_BYTES = 13 * 1024**3
#: engineering guard amended with parking (v18.1): reserved now includes ~13GiB
#: of parked storage on top of the ~2.5GiB training peak.  Not a science gate.
VRAM_GATE_BYTES = 20 * 1024**3


def fadvise_dontneed(path: Path) -> None:
    """Drop this file's clean page cache: the 264G dataset read floods the 240GiB
    cgroup (anon+shmem+file) and reclaim lost the race on the first attempt
    (SIGKILL at build 400/547, 2026-08-27T04:59Z).  drop_caches/memory.reclaim
    are read-only in this container, so eviction must be per-file from inside."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def rss_gib() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 2**20
    return -1.0


def build_partition_fp16(device, my_ids, *, log):
    """Transcribed v16.build_bundles: fp16 storage + abstain-skip."""
    from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
    from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
        CONDITION_MAXIMUM, R16DSCP, deployment_features,
    )
    from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import (
        V4GuardedOnsetLoader, canonical_public, canonical_travel, deployment_args,
    )
    from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
    from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
    import scripts.probe_r4e8_masked_oracle as r4e8

    v16.configure_determinism(v16.SEED)
    artifact = torch.load(v16.BASIS, map_location="cpu", weights_only=False)
    parent_model, normalizer, manifest_payload, _ = parent_runtime.load_model_context(device)
    manifest = parent_runtime.manifest_object(manifest_payload)
    allowed = frozenset(my_ids)
    loader = V4GuardedOnsetLoader(
        GuardedOnsetDataset(
            parent_runtime.SOURCE_H5_PATH, manifest, split="train",
            sample_ids=list(my_ids),
        )
    )

    @torch.inference_mode()
    def parent_predictor(public):
        v = public.velocity_mps[None, None].to(device)
        s = public.source_parameters[None].to(device)
        sm = public.source_map[None, None].to(device)
        medium = parent_model.encode_medium(v, normalizer)
        prepared = parent_model.prepare_sources(
            medium, s, sm, normalizer,
            record_to_medium=torch.zeros(1, dtype=torch.long, device=device),
        )
        normalized = parent_model.dense_normalized(
            prepared, public.time_s.to(device),
            x_m=public.x_m.to(device), z_m=public.z_m.to(device), time_block=16,
        )
        return normalizer.decode_pressure(normalized.float(), s[:, 4])[0].cpu()

    travel_builder = lambda v, s: grid_eikonal_travel_time(
        v, source_indices=[(round(float(s[1]) / 10), round(float(s[0]) / 10))],
        dx_m=10, dz_m=10,
    )[0]

    reference = R16DSCP(artifact["basis"], artifact["coefficient_scales"])
    bases = reference.bases.clone()
    scales = reference.coefficient_scales.clone()
    bases_device = bases.to(device).float()

    loaded = {}
    for index in range(len(loader)):
        public = canonical_public(loader[index])
        loaded[public.sample_id] = public
    missing = allowed - set(loaded)
    if missing:
        raise v16.V16Refusal(f"records missing from the guarded loader: {sorted(missing)[:5]}")

    bundles, skipped = [], []
    parked_bytes = [0]
    started = time.monotonic()
    for position, sample_id in enumerate(my_ids):
        public = loaded.pop(sample_id)
        parent = torch.as_tensor(parent_predictor(public)).detach().cpu().float()
        travel = canonical_travel(public, travel_builder).float()
        args = deployment_args(public, parent, travel, device)
        features, decisions, conditions = deployment_features(
            *args[:-2], bases_device, args[-2], args[-1]
        )
        decision = decisions[0]
        if decision.abstain or float(conditions[0]) > CONDITION_MAXIMUM:
            skipped.append({"sample_id": sample_id, "abstain": bool(decision.abstain),
                            "condition": float(conditions[0])})
            del args, features, parent
            continue
        record = r4e8.record_from_manifest(manifest_payload, sample_id)
        truth = v16.load_truth_allowlisted(record, allowed)
        k1 = int(public.observed_indices[1])
        parent_field = args[7][0].detach().float()
        keep_full, _floor = v16.full_time_keep_mask(parent_field, k1=k1, tau=v16.TAU)
        feat_t = features.detach().half()
        parent_t = (args[7].detach() * FP16_FIELD_SCALE).half()
        truth_t = (truth[k1 + 1:] * FP16_FIELD_SCALE).half()
        need = (feat_t.numel() + parent_t.numel() + truth_t.numel()) * 2
        if parked_bytes[0] + need <= VRAM_PARK_BYTES:
            feat_t, parent_t, truth_t = feat_t.to(device), parent_t.to(device), truth_t.to(device)
            parked_bytes[0] += need
        else:
            feat_t, parent_t, truth_t = feat_t.cpu(), parent_t.cpu(), truth_t.cpu()
        bundles.append(v16.CpuBundle(
            family=str(record.family), sample_id=sample_id,
            route_index=int(decision.index), k1=k1,
            features=feat_t,
            parent_full=parent_t,
            truth_future=truth_t,
            keep_full=keep_full.detach().cpu(),
        ))
        del args, features, parent, truth
        if (position + 1) % 50 == 0:
            fadvise_dontneed(Path(parent_runtime.SOURCE_H5_PATH))
            log(f"build {position + 1}/{len(my_ids)} "
                f"({time.monotonic() - started:.0f}s, rss {rss_gib():.1f}GiB, "
                f"parked {parked_bytes[0] / 2**30:.1f}GiB)")
    del parent_model, loaded
    torch.cuda.empty_cache()
    return bundles, skipped, bases, scales, time.monotonic() - started


def v18_bundle_loss(head, scales, bases_device, bundle, device):
    """v16.bundle_loss with fp32 casts for fp16-stored bundles."""
    features = bundle.features.to(device, non_blocking=True).float()
    parent_full = bundle.parent_full.to(device, non_blocking=True).float() / FP16_FIELD_SCALE
    truth_future = bundle.truth_future.to(device, non_blocking=True).float() / FP16_FIELD_SCALE
    coefficient = v16.head_coefficient(head, scales, bundle, features)
    adapted = v16.materialize_confined(parent_full, bases_device, bundle, coefficient)
    keep_future = bundle.keep_full[bundle.k1 + 1:].to(device)
    losses = v16.masked_confined_loss(
        adapted[:, bundle.k1 + 1:].float(), truth_future[None], coefficient,
        parent_full[:, bundle.k1 + 1:].float(), keep_future, tau=v16.TAU,
    )
    return losses, adapted, parent_full, truth_future


@torch.no_grad()
def v18_score_bundle(head, scales, bases_device, bundle, device):
    losses, adapted, parent_full, truth_future = v18_bundle_loss(
        head, scales, bases_device, bundle, device
    )
    cand = float(v16.relative_l2(adapted[0, bundle.k1 + 1:].float(), truth_future))
    par = float(v16.relative_l2(parent_full[0, bundle.k1 + 1:].float(), truth_future))
    return {
        "sample_id": bundle.sample_id, "family": bundle.family,
        "loss": float(losses["total"]),
        "aggregate_rel_l2": cand, "parent_rel_l2": par,
        "gain_iii": float((par - cand) / max(abs(par), 1e-30)),
        "nonworse": bool(cand <= par),
    }


def gather_scores(head, scales, bases_device, my_confirm, device, expected_ids):
    local = [v18_score_bundle(head, scales, bases_device, b, device) for b in my_confirm]
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


def v18_eval_gates(gates_cfg, scores, peak_vram, checkpoint_bytes):
    joint = sum(r["gain_iii"] for r in scores) / max(len(scores), 1)
    by_family = {}
    for row in scores:
        by_family.setdefault(row["family"], []).append(row["gain_iii"])
    family_means = {k: sum(v) / len(v) for k, v in by_family.items()}
    tol = float(gates_cfg["nonworse_tolerance"])
    within = sum(1 for r in scores if r["gain_iii"] >= -tol)
    worst = min(r["gain_iii"] for r in scores)
    finite = all(math.isfinite(r[k]) for r in scores
                 for k in ("loss", "aggregate_rel_l2", "parent_rel_l2"))
    return {
        "joint_improvement": {"value": joint, "threshold": gates_cfg["joint_improvement"],
                              "passed": joint >= gates_cfg["joint_improvement"]},
        "per_family_improvement": {"value": family_means,
                                   "threshold": gates_cfg["per_family_improvement"],
                                   "passed": len(family_means) == 3 and all(
                                       v >= gates_cfg["per_family_improvement"]
                                       for v in family_means.values())},
        "nonworse_within_tolerance": {"value": within, "tolerance": tol,
                                      "threshold": gates_cfg["nonworse_min_count"],
                                      "passed": within >= gates_cfg["nonworse_min_count"]},
        "worst_harm": {"value": worst, "floor": gates_cfg["worst_harm_floor"],
                       "passed": worst >= gates_cfg["worst_harm_floor"]},
        "finite": {"value": finite, "passed": finite},
        "vram": {"value_bytes": peak_vram, "limit_bytes": VRAM_GATE_BYTES,
                 "passed": 0 < peak_vram <= VRAM_GATE_BYTES},
        "checkpoint": {"value_bytes": checkpoint_bytes,
                       "limit_bytes": v16.CHECKPOINT_LIMIT_BYTES,
                       "passed": 0 < checkpoint_bytes <= v16.CHECKPOINT_LIMIT_BYTES},
    }


def run_arm(arm_name, arm_cfg, plan, gates_cfg, fit, my_confirm, eval_ids,
            bases_device, scales, parent_rel, device, rank, log):
    is0 = rank == 0
    run = OUT_V18 / arm_name
    if is0 and (run / "terminal.json").exists():
        raise v16.V16Refusal(f"arm terminal already exists: {run}")
    hinge = arm_cfg["hinge"]
    started = time.monotonic()
    deadline = started + plan["arm_train_wall_s"]

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
        losses, adapted, _pf, truth_future = v18_bundle_loss(
            head, scales, bases_device, bundle, device
        )
        total = losses["total"]
        if hinge["enabled"]:
            cand_rel = torch.linalg.vector_norm(
                adapted[0, bundle.k1 + 1:].float() - truth_future.float()
            ) / torch.linalg.vector_norm(truth_future.float()).clamp_min(1e-30)
            ratio = cand_rel / parent_rel[bundle.sample_id]
            total = total + float(hinge["lam"]) * torch.relu(
                ratio - (1.0 - float(hinge["margin"])))
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
                               "joint_mean_gain": joint, "records": snap})
            checkpoint_bytes = v16.save_checkpoint(head, optimizer, run / "last.pt", {
                "stage": arm_name, "epoch": epoch + 1, "seed": v16.SEED})
            if monitor < best:
                best, best_epoch, bad = monitor, epoch + 1, 0
                v16.save_checkpoint(head, optimizer, run / "best.pt", {
                    "stage": arm_name, "epoch": epoch + 1, "seed": v16.SEED,
                    "monitor": monitor})
            else:
                bad += 1
            control[0] = monitor
            control[1] = 1.0 if bad >= plan["patience"] else 0.0
            log(f"[{arm_name}] epoch {epoch + 1} monitor {monitor:.6f} joint {joint:+.4f} "
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
    result = None
    if is0:
        peak_vram = int(torch.cuda.max_memory_reserved(device))
        gates = v18_eval_gates(gates_cfg, final_scores, peak_vram, checkpoint_bytes)
        all_passed = all(bool(v.get("passed")) for v in gates.values())
        final_status = status if status not in {"completed", "early_stop_patience"} \
            else ("passed" if all_passed else "fail_gate")
        result = {
            "arm": arm_name, "hinge": hinge, "status": final_status,
            "run_status": status, "gates": gates,
            "metrics": {"final_records": final_scores, "trajectory": trajectory},
            "resources": {"wall_s": time.monotonic() - started,
                          "peak_cuda_reserved_bytes": peak_vram,
                          "checkpoint_bytes": checkpoint_bytes,
                          "best_checkpoint": str(best_path)},
        }
        v16.atomic_json(result, run / "terminal.json")
        log(f"[{arm_name}] {final_status} wall {result['resources']['wall_s']:.0f}s")
    dist.barrier()
    return result


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=6))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is0 = rank == 0
    log = (lambda m: print(f"{v16.utc_now()} {m}", flush=True)) if is0 else (lambda m: None)
    log_all = lambda m: print(f"{v16.utc_now()} [rank{rank}] {m}", flush=True)

    prereg = json.loads(PREREG_V18.read_text())
    prereg_sha = v16.sha256_file(PREREG_V18)
    plan, gates_cfg = prereg["plan"], prereg["gates_per_arm"]
    panel_path = ROOT / prereg["panel_v2"]["path"]
    panel_sha = v16.sha256_file(panel_path)
    if panel_sha != prereg["panel_v2"]["sha256"]:
        raise v16.V16Refusal("panels_v2 sha mismatch against the frozen prereg")
    panel = json.loads(panel_path.read_text())["records"]
    fit_ids = sorted(r["sample_id"] for r in panel if r["role"] == plan["fit_role"])
    eval_ids = sorted(r["sample_id"] for r in panel if r["role"] == plan["eval_role"])

    if is0:
        OUT_V18.mkdir(parents=True, exist_ok=True)
        if (OUT_V18 / "chain_terminal.json").exists():
            raise v16.V16Refusal("v18 chain terminal already exists")
    bindings = v16.verify_bindings() if is0 else None

    my_fit_ids = [s for i, s in enumerate(fit_ids) if i % WORLD == rank]
    my_eval_ids = [s for i, s in enumerate(eval_ids) if i % WORLD == rank]
    log_all(f"partition fit {len(my_fit_ids)} confirm {len(my_eval_ids)}")
    time.sleep(3.0 * rank)
    bundles, skipped, bases, scales, build_s = build_partition_fp16(
        device, my_fit_ids + my_eval_ids, log=log_all)
    eval_set = set(my_eval_ids)
    fit = [b for b in bundles if b.sample_id not in eval_set]
    my_confirm = [b for b in bundles if b.sample_id in eval_set]
    if [s for s in my_eval_ids if s not in {b.sample_id for b in my_confirm}]:
        raise v16.V16Refusal("a confirm record abstained or failed to build")
    log_all(f"built fit {len(fit)} confirm {len(my_confirm)} skipped {len(skipped)} "
            f"in {build_s:.0f}s, rss {rss_gib():.1f}GiB")
    bases_device = bases.to(device).float()
    parent_rel = {}
    for b in fit:
        parent_rel[b.sample_id] = float(v16.relative_l2(
            b.parent_full[0, b.k1 + 1:].float(), b.truth_future.float()))

    counts = [None] * WORLD
    dist.all_gather_object(counts, {"rank": rank, "fit": len(fit),
                                    "confirm": len(my_confirm),
                                    "skipped": skipped, "build_s": build_s,
                                    "rss_gib": rss_gib()})
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if is0:
        total_skip = sum(len(c["skipped"]) for c in counts)
        log(f"[v18] global fit {sum(c['fit'] for c in counts)} "
            f"confirm {sum(c['confirm'] for c in counts)} skipped {total_skip}; "
            f"prereg {prereg_sha[:12]}")

    chain_started = time.monotonic()
    arm_results = {}
    for arm_name in prereg["arms_order_fixed"]:
        arm_results[arm_name] = run_arm(
            arm_name, prereg["arms"][arm_name], plan, gates_cfg, fit, my_confirm,
            eval_ids, bases_device, scales, parent_rel, device, rank, log)

    if is0:
        parent_after = v16.sha256_file(v16.PARENT_PATH)
        chain = {
            "schema": "r16_dscp_v18_chain_terminal_v1",
            "candidate": prereg["candidate"],
            "status": {k: (v or {}).get("status") for k, v in arm_results.items()},
            "preregistration_sha256": prereg_sha,
            "panels_v2_sha256": panel_sha,
            "authorization": prereg["authorization"],
            "partitions": counts,
            "arms": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "metrics"}
                     for k, v in arm_results.items()},
            "resources": {"total_wall_s": time.monotonic() - chain_started,
                          "build_s_per_rank": [c["build_s"] for c in counts],
                          "rss_gib_per_rank": [c["rss_gib"] for c in counts]},
            "parent": {"sha256_before": v16.PARENT_SHA256, "sha256_after": parent_after,
                        "match": parent_after == v16.PARENT_SHA256, "writes": 0},
            "sealed_splits_opened": False,
            "thresholds_moved_toward_measured_values": False,
            "disclaimers": {"stage": v16.DISCLAIMERS["long"],
                            "correction_nature": v16.DISCLAIMERS["correction_nature"]},
            "bindings": bindings,
            "completed_utc": v16.utc_now(),
        }
        v16.atomic_json(chain, OUT_V18 / "chain_terminal.json")
        log(f"[v18] chain done: {chain['status']}")
        if parent_after != v16.PARENT_SHA256:
            raise v16.V16Refusal("parent checkpoint changed during the chain")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
