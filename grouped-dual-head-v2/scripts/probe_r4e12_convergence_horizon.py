#!/usr/bin/env python3
"""r4e12: convergence horizon of the M2 wide head (field loss).

Frozen spec: results/r4e12_convergence_horizon_spec_20260826.md

r4e11 landed in C4: the 8690-parameter M2/M3 heads lift every record over the
v15 head but none reaches f>=0.5 on 2/3 records, while all deployable arms were
still climbing +0.12..+0.28 over their last 576 updates.  This probe measures
the plateau at 4x budget, a cosine schedule, a 2x-width variant, and a
regression-pretrain/field-finetune hybrid.  Train-only, not a stage, not
promotable, no threshold moved, validation/test_id untouched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    INPUT_CHANNELS,
    RANK,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    configure_determinism,
    sha256_file,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4e10_optimization_gap as r4e10
from scripts import probe_r4e11_head_capacity_ladder as r4e11

CANDIDATE = "r4e12_convergence_horizon"
SPEC_PATH = PROJECT_ROOT / "results/r4e12_convergence_horizon_spec_20260826.md"
SPEC_SHA256 = "cfb464927de0e660c4277e0f81504303c131e450a35140e2594c1627d48a0817"
RESULT_DIR = PROJECT_ROOT / "results/r4e12_convergence_horizon_20260826"

HARD_FROZEN = {
    "scripts/probe_r4e10_optimization_gap.py": "8ea80e3327b206534430cee7fc209261f22bc215a0944a75d544f2c999e1d0b9",
    "results/r4e10_optimization_gap_20260826/leg_a/terminal.json": "7b22fa2418fadd3266fa9e668b1f0bdc1bc06241f23c84267a34c55e94c23c2d",
}
RUNTIME_PINNED = (
    "scripts/probe_r4e11_head_capacity_ladder.py",
    "results/r4e11_head_capacity_ladder_20260826/terminal.json",
)

UPDATES = 3072
EVAL_EVERY = 384
LR = 3e-3
COSINE_FLOOR = 3e-5
HYBRID_PRETRAIN = 768
CLIMB_EPSILON = 0.02
EXTENSION_UPDATES = 3072
GPU_SECONDS_MAXIMUM = 1500.0
PEAK_RESERVED_MAXIMUM = int(20 * 1024**3)

RULE = {
    "D1": "an arm reaches f>=0.5 on >=2/3 records at its final -> head family + budget suffice",
    "D2": "no arm still climbing (f(final)-f(final-768) >= 0.02 on any record) and no D1 -> plateau below the line",
    "D3": "best arm still climbing without D1 -> one pre-authorized 3072-update extension, then reclassify D1/D2",
    "D4": "otherwise per-record report; W+ (width margin >= +0.05 on >=2 records) may hold alongside",
}


class ProbeRefusal(RuntimeError):
    pass


class Wide128Head(nn.Module):
    """W arm: the M2 topology at width 128 (~30k parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, INPUT_CHANNELS, 3, padding=1, groups=INPUT_CHANNELS, bias=True),
            nn.Conv2d(INPUT_CHANNELS, 128, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=True),
            nn.Conv2d(128, 128, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128, bias=True),
            nn.Conv2d(128, RANK, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features).tanh()


def make_optimizer(head: nn.Module):
    return torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.99),
                             eps=1e-8, weight_decay=1e-4)


def train_span(pipe, head, optimizer, scheduler, *, start: int, updates: int,
               supervision_for, targets, oracle_bounds, trajectory, deadline: float) -> str:
    bundles = pipe["bundles"]
    for offset in range(updates):
        if time.monotonic() > deadline:
            return "partial_budget"
        update = start + offset
        bundle = bundles[update % len(bundles)]
        coefficient = r4e11.head_coefficient(pipe, head, bundle)
        supervision = supervision_for(update)
        if supervision == "field":
            losses, _ = r4e10.record_loss(pipe, bundle, coefficient)
            total = losses["total"]
        else:
            total = r4e11.target_loss(pipe, bundle, coefficient, targets[bundle.sample_id])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ProbeRefusal("nonfinite or absent gradient")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None and update + 1 <= UPDATES:
            scheduler.step()
        if (update + 1) % EVAL_EVERY == 0:
            snap = r4e11.evaluate_head(pipe, head, oracle_bounds, targets)
            trajectory.append({
                "update": update + 1,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "records": snap,
            })
    return "completed"


def fractions_at(trajectory, update):
    for snap in trajectory:
        if snap["update"] == update:
            return {sid: row["oracle_fraction"] for sid, row in snap["records"].items()}
    return None


def climbing(trajectory, final_update):
    now = fractions_at(trajectory, final_update)
    before = fractions_at(trajectory, final_update - 2 * EVAL_EVERY)
    if now is None or before is None:
        return False, {}
    deltas = {sid: now[sid] - before[sid] for sid in now}
    return any(v >= CLIMB_EPSILON for v in deltas.values()), deltas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    args = parser.parse_args(argv)

    started = time.monotonic()
    deadline = started + GPU_SECONDS_MAXIMUM
    terminal_path = RESULT_DIR / "terminal.json"
    if terminal_path.exists():
        raise ProbeRefusal(f"terminal already exists: {terminal_path}")
    if sha256_file(SPEC_PATH) != SPEC_SHA256:
        raise ProbeRefusal("r4e12 spec drifted after freeze")

    bindings = r4e10.verify_bindings()
    for rel, expected in HARD_FROZEN.items():
        observed = sha256_file(PROJECT_ROOT / rel)
        bindings[rel] = {"expected": expected, "observed": observed}
        if observed != expected:
            raise ProbeRefusal(f"binding drift: {rel}")
    for rel in RUNTIME_PINNED:
        bindings[rel] = {"runtime_pinned": sha256_file(PROJECT_ROOT / rel)}
    parent_runtime.require_free_disk(minimum=r4e10.MIN_FREE_BYTES)

    device = torch.device("cuda:0")
    gpu = parent_runtime.gpu_identity(args.physical_gpu_index, device)
    torch.cuda.reset_peak_memory_stats(device)

    smoke = json.loads(r4e10.SMOKE_TERMINAL.read_text())
    pipe = r4e10.build_pipeline(device)
    fidelity = r4e10.fidelity_gate(pipe, smoke)
    loss_fidelity = r4e10.initial_loss_fidelity(pipe, smoke)
    base = {
        "schema": "r4e12_convergence_horizon_result_v1",
        "candidate": CANDIDATE,
        "kind": "diagnostic_probe_not_a_promotion_candidate",
        "spec_sha256": SPEC_SHA256,
        "started_utc": r4e10.utc_now(),
        "gpu": gpu,
        "pipeline_fidelity": fidelity,
        "initial_loss_fidelity": loss_fidelity,
        "claim_scope": (
            "offline_train_only_three_record_convergence_horizon_"
            "not_generalization_not_a_stage_not_promotable"
        ),
        "thresholds_moved_toward_measured_values": False,
        "sealed_splits_opened": False,
    }
    if not (fidelity["passed"] and loss_fidelity["passed"]):
        payload = {**base, "status": "invalid_pipeline", "completed_utc": r4e10.utc_now()}
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        terminal_path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(json.dumps({"status": "invalid_pipeline"}))
        return 1

    oracle_bounds = {b.sample_id: r4e10.oracle_for(pipe, b) for b in pipe["bundles"]}
    targets = {b.sample_id: r4e11.oracle_coefficient_target(pipe, b) for b in pipe["bundles"]}

    arm_specs = {
        "L": {"factory": r4e11.WideHead, "schedule": "constant",
              "supervision": lambda u: "field"},
        "S": {"factory": r4e11.WideHead, "schedule": "cosine",
              "supervision": lambda u: "field"},
        "W": {"factory": Wide128Head, "schedule": "constant",
              "supervision": lambda u: "field"},
        "H": {"factory": r4e11.WideHead, "schedule": "constant",
              "supervision": lambda u: "target" if u < HYBRID_PRETRAIN else "field"},
    }

    arms: dict[str, Any] = {}
    heads: dict[str, nn.Module] = {}
    optimizers: dict[str, Any] = {}
    for name, spec in arm_specs.items():
        configure_determinism(372)
        head = spec["factory"]().to(device).float()
        optimizer = make_optimizer(head)
        scheduler = None
        if spec["schedule"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=UPDATES, eta_min=COSINE_FLOOR)
        trajectory: list[dict] = []
        if name == "H":
            status = train_span(pipe, head, optimizer, scheduler, start=0,
                                updates=HYBRID_PRETRAIN, supervision_for=spec["supervision"],
                                targets=targets, oracle_bounds=oracle_bounds,
                                trajectory=trajectory, deadline=deadline)
            if status == "completed":
                optimizer = make_optimizer(head)  # spec: reset at the switch
                status = train_span(pipe, head, optimizer, scheduler,
                                    start=HYBRID_PRETRAIN, updates=UPDATES - HYBRID_PRETRAIN,
                                    supervision_for=spec["supervision"], targets=targets,
                                    oracle_bounds=oracle_bounds, trajectory=trajectory,
                                    deadline=deadline)
        else:
            status = train_span(pipe, head, optimizer, scheduler, start=0, updates=UPDATES,
                                supervision_for=spec["supervision"], targets=targets,
                                oracle_bounds=oracle_bounds, trajectory=trajectory,
                                deadline=deadline)
        final = r4e11.evaluate_head(pipe, head, oracle_bounds, targets)
        arms[name] = {
            "status": status,
            "schedule": spec["schedule"],
            "supervision": "hybrid_target768_field" if name == "H" else "field",
            "updates": UPDATES,
            "parameter_count": r4e11.parameter_count(head),
            "trajectory": trajectory,
            "final": final,
        }
        heads[name] = head
        optimizers[name] = optimizer

    # pre-registered classification
    records = [sid for _f, sid in r4e10.RECORDS]

    def final_fractions(arm):
        return {sid: arm["final"][sid]["oracle_fraction"] for sid in records}

    def d1_hit(arm):
        return sum(v >= 0.5 for v in final_fractions(arm).values()) >= 2

    verdict: dict[str, Any] = {"rule": RULE}
    d1_arms = {n: final_fractions(a) for n, a in arms.items() if d1_hit(a)}
    climb_state = {}
    for name, arm in arms.items():
        is_climbing, deltas = climbing(arm["trajectory"], UPDATES)
        climb_state[name] = {"climbing": is_climbing, "deltas_last_768": deltas}
    verdict["climb_state"] = climb_state

    best_name = max(arms, key=lambda n: sum(final_fractions(arms[n]).values()))
    verdict["best_arm"] = best_name
    extension = None
    if d1_arms:
        verdict["branch"] = "D1_head_and_budget_sufficient"
        verdict["D1_arms"] = d1_arms
    elif climb_state[best_name]["climbing"]:
        # D3: one pre-authorized extension of the best arm
        head = heads[best_name]
        trajectory = arms[best_name]["trajectory"]
        status = train_span(pipe, head, optimizers[best_name], None, start=UPDATES,
                            updates=EXTENSION_UPDATES,
                            supervision_for=lambda u: "field", targets=targets,
                            oracle_bounds=oracle_bounds, trajectory=trajectory,
                            deadline=deadline)
        final = r4e11.evaluate_head(pipe, head, oracle_bounds, targets)
        arms[best_name]["extension"] = {
            "status": status, "updates": UPDATES + EXTENSION_UPDATES, "final": final,
        }
        ext_fractions = {sid: final[sid]["oracle_fraction"] for sid in records}
        still, deltas = climbing(trajectory, UPDATES + EXTENSION_UPDATES)
        extension = {"arm": best_name, "fractions": ext_fractions,
                     "climbing": still, "deltas_last_768": deltas}
        if sum(v >= 0.5 for v in ext_fractions.values()) >= 2:
            verdict["branch"] = "D3_then_D1_after_extension"
        elif still:
            verdict["branch"] = "D3_horizon_beyond_6144"
        else:
            verdict["branch"] = "D3_then_D2_plateau_below_line"
        verdict["extension"] = extension
    elif not any(v["climbing"] for v in climb_state.values()):
        verdict["branch"] = "D2_plateau_below_line"
    else:
        verdict["branch"] = "D4_mixed_report_per_record"

    width_margin = {sid: final_fractions(arms["W"])[sid] - final_fractions(arms["L"])[sid]
                    for sid in records}
    verdict["width_margin_W_minus_L"] = width_margin
    verdict["W_plus"] = sum(v >= 0.05 for v in width_margin.values()) >= 2

    del heads, optimizers
    torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_reserved(device))
    if peak > PEAK_RESERVED_MAXIMUM:
        raise ProbeRefusal("peak reserved memory exceeded the probe gate")
    parent_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if parent_after != parent_runtime.CHECKPOINT_SHA256:
        raise ProbeRefusal("parent checkpoint changed during the probe")

    payload = {
        **base,
        "status": "success",
        "oracle_bounds": {k: {"acceptance_convention_gain": float(v["acceptance_convention_gain"])}
                          for k, v in oracle_bounds.items()},
        "arms": arms,
        "verdict": verdict,
        "elapsed_s": elapsed,
        "peak_cuda_reserved_bytes": peak,
        "parent_checkpoint_before_after_match": True,
        "bindings": bindings,
        "completed_utc": r4e10.utc_now(),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    temp = RESULT_DIR / "terminal.tmp"
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(temp, terminal_path)
    print(json.dumps({"status": "success", "elapsed_s": round(elapsed, 1),
                      "branch": verdict["branch"],
                      "best_arm": verdict["best_arm"],
                      "W_plus": verdict["W_plus"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
