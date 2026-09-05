#!/usr/bin/env python3
"""r4e13: cold-start attribution and data-scaling ladder for cross-record transfer.

Frozen spec: results/r4e13_crossrecord_scaling_spec_20260826.md

v16 pilot failed its gates (joint -0.032, nonworse 10/24).  Two entangled
explanations: the warm start from the smoke checkpoint (an implementation
choice not named in the preregistration), and fit-set size.  Three cold-start
arms (24 / 96 / 192 fit records) all evaluated on the same held-out
pilot_confirm 24 separate H-W (warm-start harm), H-S (data scaling), and
H-F (fundamental non-transfer).  Train-only, not a stage, not promotable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    configure_determinism,
    sha256_file,
)
import scripts.train_r16_dscp_v16 as v16
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts.probe_r4e12_convergence_horizon import Wide128Head

CANDIDATE = "r4e13_crossrecord_scaling"
SPEC_PATH = PROJECT_ROOT / "results/r4e13_crossrecord_scaling_spec_20260826.md"
SPEC_SHA256 = "76c5f3df4c272cd2a86e6a79fc5db342e04492323cd7f78a2036df4288a9faeb"
RESULT_DIR = PROJECT_ROOT / "results/r4e13_crossrecord_scaling_20260826"

RUNTIME_PINNED = (
    "scripts/train_r16_dscp_v16.py",
    "scripts/probe_r4e13_crossrecord_scaling.py",
    "results/r16_dscp_v16/pilot/terminal.json",
    "results/r4e13_crossrecord_scaling_spec_20260826.md",
)

SEED = 372
UPDATES = 3072
EVAL_EVERY = 384
LR = 3e-3
WALL_S_MAXIMUM = 6000.0
PEAK_RESERVED_MAXIMUM = int(20 * 1024**3)
PILOT_MEASURED_MEAN_GAIN = -0.03228693078903511  # v16 pilot terminal, frozen reference
PILOT_GATES = {"joint": 0.01, "per_family": 0.005, "nonworse": 23}

RULE = {
    "E1": "cold-24 mean gain >= pilot measured + 0.02 -> warm-start harm; if it also passes the pilot gates, v17 = cold pilot",
    "E2": "mean gain monotonic A->B->C and C passes the pilot gates -> data scaling works, v17 = long-data pilot",
    "E3": "C mean gain < A mean gain + 0.01 and C fails the gates -> fundamental non-transfer at this architecture",
    "E4": "otherwise per-arm per-family report",
}


class ProbeRefusal(RuntimeError):
    pass


def mean_gain(scores) -> float:
    return statistics.mean(r["gain_iii"] for r in scores)


def family_means(scores) -> dict[str, float]:
    fams: dict[str, list[float]] = {}
    for row in scores:
        fams.setdefault(row["family"], []).append(row["gain_iii"])
    return {k: statistics.mean(v) for k, v in fams.items()}


def passes_pilot_gates(scores) -> dict[str, Any]:
    joint = mean_gain(scores)
    fams = family_means(scores)
    nonworse = sum(bool(r["nonworse"]) for r in scores)
    return {
        "joint": joint, "per_family": fams, "nonworse": nonworse,
        "passed": (
            joint >= PILOT_GATES["joint"]
            and len(fams) == 3
            and all(v >= PILOT_GATES["per_family"] for v in fams.values())
            and nonworse >= PILOT_GATES["nonworse"]
        ),
    }


def train_arm(name, fit, confirm, scales, bases_device, device, deadline):
    configure_determinism(SEED)
    head = Wide128Head().to(device).float()
    if sum(p.numel() for p in head.parameters()) != v16.EXPECTED_PARAMETERS:
        raise ProbeRefusal("head parameter count drifted")
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(fit), generator=generator).tolist()
    trajectory = []
    status = "completed"
    for update in range(UPDATES):
        if time.monotonic() > deadline:
            status = "partial_budget"
            break
        if update % len(fit) == 0 and update:
            order = torch.randperm(len(fit), generator=generator).tolist()
        bundle = fit[order[update % len(fit)]]
        losses, *_ = v16.bundle_loss(head, scales, bases_device, bundle, device)
        total = losses["total"]
        if not torch.isfinite(total):
            raise ProbeRefusal(f"non-finite loss in arm {name} at update {update}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ProbeRefusal("nonfinite or absent gradient")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if (update + 1) % EVAL_EVERY == 0:
            snap = [v16.score_bundle(head, scales, bases_device, b, device) for b in confirm]
            trajectory.append({
                "update": update + 1,
                "confirm_mean_gain": mean_gain(snap),
                "confirm_family_means": family_means(snap),
                "confirm_nonworse": sum(bool(r["nonworse"]) for r in snap),
            })
    confirm_final = [v16.score_bundle(head, scales, bases_device, b, device) for b in confirm]
    fit_final = [v16.score_bundle(head, scales, bases_device, b, device) for b in fit]
    result = {
        "status": status, "fit_records": len(fit), "updates": UPDATES,
        "learning_rate": LR, "cold_start": True,
        "trajectory": trajectory,
        "confirm_final": confirm_final,
        "confirm_summary": passes_pilot_gates(confirm_final),
        "fit_summary": {
            "mean_gain": mean_gain(fit_final),
            "family_means": family_means(fit_final),
            "nonworse": sum(bool(r["nonworse"]) for r in fit_final),
            "n": len(fit_final),
        },
        "fit_final": fit_final,
    }
    del head, optimizer
    torch.cuda.empty_cache()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    args = parser.parse_args(argv)

    started = time.monotonic()
    deadline = started + WALL_S_MAXIMUM
    terminal_path = RESULT_DIR / "terminal.json"
    if terminal_path.exists():
        raise ProbeRefusal(f"terminal already exists: {terminal_path}")
    if sha256_file(SPEC_PATH) != SPEC_SHA256:
        raise ProbeRefusal("r4e13 spec drifted after freeze")
    bindings = v16.verify_bindings()
    for rel in RUNTIME_PINNED:
        bindings.setdefault(rel, {"runtime_pinned": sha256_file(PROJECT_ROOT / rel)})

    device = torch.device("cuda:0")
    gpu = parent_runtime.gpu_identity(args.physical_gpu_index, device)
    log = lambda msg: print(f"{v16.utc_now()} {msg}", flush=True)

    pilot_fit = v16.role_sample_ids("pilot_fit")
    long_fit = v16.role_sample_ids("long_fit")
    confirm_ids = v16.role_sample_ids("pilot_confirm")
    arm_fit_ids = {
        "A24": list(pilot_fit),
        "B96": list(pilot_fit) + long_fit[:72],
        "C192": list(long_fit),
    }
    all_ids = list(dict.fromkeys([*pilot_fit, *long_fit, *confirm_ids]))
    log(f"building {len(all_ids)} bundles")
    bundles, context = v16.build_bundles(device, all_ids, log=log)
    by_id = {b.sample_id: b for b in bundles}
    confirm = [by_id[s] for s in confirm_ids]
    scales = context["scales"]
    bases_device = context["bases"].to(device).float()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    arms: dict[str, Any] = {}
    for name, ids in arm_fit_ids.items():
        log(f"arm {name} ({len(ids)} fit records)")
        arms[name] = train_arm(name, [by_id[s] for s in ids], confirm,
                               scales, bases_device, device, deadline)
        log(f"arm {name} confirm mean {arms[name]['confirm_summary']['joint']:+.4f} "
            f"fit mean {arms[name]['fit_summary']['mean_gain']:+.4f}")

    g = {n: arms[n]["confirm_summary"]["joint"] for n in arms}
    verdict: dict[str, Any] = {"rule": RULE, "confirm_mean_gain": g,
                               "pilot_measured_reference": PILOT_MEASURED_MEAN_GAIN}
    e1 = g["A24"] >= PILOT_MEASURED_MEAN_GAIN + 0.02
    e1_pass_gates = bool(arms["A24"]["confirm_summary"]["passed"])
    monotonic = g["A24"] <= g["B96"] <= g["C192"]
    c_pass = bool(arms["C192"]["confirm_summary"]["passed"])
    e2 = monotonic and c_pass
    e3 = (g["C192"] < g["A24"] + 0.01) and not c_pass
    verdict["E1_warm_start_harm"] = bool(e1)
    verdict["E1_cold24_passes_pilot_gates"] = e1_pass_gates
    verdict["E2_data_scaling_works"] = bool(e2)
    verdict["E2_monotonic"] = bool(monotonic)
    verdict["E3_fundamental_non_transfer"] = bool(e3)
    branch = [n for n, hit in (("E1", e1), ("E2", e2), ("E3", e3)) if hit]
    verdict["branch"] = "+".join(branch) if branch else "E4_mixed_report"

    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_reserved(device))
    if peak > PEAK_RESERVED_MAXIMUM:
        raise ProbeRefusal("peak reserved memory exceeded the probe gate")
    parent_after = sha256_file(v16.PARENT_PATH)
    if parent_after != v16.PARENT_SHA256:
        raise ProbeRefusal("parent checkpoint changed during the probe")

    payload = {
        "schema": "r4e13_crossrecord_scaling_result_v1",
        "candidate": CANDIDATE,
        "kind": "diagnostic_probe_not_a_promotion_candidate",
        "spec_sha256": SPEC_SHA256,
        "gpu": gpu,
        "status": "success",
        "arms": arms,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "cache_build_s": context["cache_build_s"],
        "peak_cuda_reserved_bytes": peak,
        "parent_checkpoint_before_after_match": True,
        "sealed_splits_opened": False,
        "thresholds_moved_toward_measured_values": False,
        "claim_scope": (
            "train_only_cold_start_data_ladder_on_held_out_train_confirm_"
            "not_validation_not_a_stage_not_promotable"
        ),
        "bindings": bindings,
        "completed_utc": v16.utc_now(),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    temp = RESULT_DIR / "terminal.tmp"
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    os.replace(temp, RESULT_DIR / "terminal.json")
    print(json.dumps({"status": "success", "elapsed_s": round(elapsed, 1),
                      "branch": verdict["branch"], "confirm_mean_gain": g}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
