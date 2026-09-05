#!/usr/bin/env python3
"""r4e11: feature-head capacity / information ladder behind the r4e10 finding.

Frozen spec: results/r4e11_head_capacity_ladder_spec_20260826.md

r4e10 showed the v15 ansatz reaches 80-99% of its per-record oracle bound when
the 1202-parameter feature head is bypassed, and at most 30% through it.  This
probe separates three explanations before any v16 design is frozen:

- H-C capacity: the 29 deployment channels carry the information, the head
  cannot express the mapping (width / receptive field);
- H-D information: the channels do not determine the oracle coefficients, so no
  head helps;
- H-E loss surface: the field loss is the obstacle; supervising directly on the
  oracle coefficient maps lets the same head learn.

Train-only diagnostic on the three smoke records.  Not a stage, not promotable,
no threshold is moved, validation/test_id untouched, no model checkpoint is
written.  Reuses the r4e10 pipeline (fidelity gates included) verbatim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    INPUT_CHANNELS,
    RANK,
    R16DSCP,
    RidgePointwiseBaseline,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    configure_determinism,
    sha256_file,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v3 import (
    TAU,
    energy_keep_mask,
    parent_energy_keep_mask,
    weighted_coefficient_map,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4e10_optimization_gap as r4e10


CANDIDATE = "r4e11_head_capacity_ladder"
SPEC_PATH = PROJECT_ROOT / "results/r4e11_head_capacity_ladder_spec_20260826.md"
SPEC_SHA256 = "2e01ef64bba79544eeaae6880bba4b1016e5697ecafdcf9f85b3412bede0fe9d"
RESULT_DIR = PROJECT_ROOT / "results/r4e11_head_capacity_ladder_20260826"

EXTRA_FROZEN = {
    "scripts/probe_r4e10_optimization_gap.py": "8ea80e3327b206534430cee7fc209261f22bc215a0944a75d544f2c999e1d0b9",
    "results/r4e10_optimization_gap_20260826/leg_a/terminal.json": "7b22fa2418fadd3266fa9e668b1f0bdc1bc06241f23c84267a34c55e94c23c2d",
    "results/r4e10_optimization_gap_20260826/leg_b/terminal.json": "38e83546d269c285349bdb00b28bc417cbffd072cf74ac2e0de4141a544aa5f4",
}

UPDATES = 768
SNAPSHOT_UPDATE = 192
LR = 3e-3
CLIP_EDGE = 0.999
GPU_SECONDS_MAXIMUM = 1200.0
PEAK_RESERVED_MAXIMUM = int(20 * 1024**3)

RULE = {
    "C1": "a deployable arm (M2*/M3*) reaches f>=0.5 on >=2/3 records -> H-C, v16 = that head family",
    "C2": "all shared heads f<0.5 on >=2 records and best target-arm fit_rel_err>0.9 -> H-D, v16 = feature enrichment",
    "C3": "M1t improves >=0.15 over the r4e10 lr3e-3 arm on >=2 records without an equal field-vs-target gap at M2/M3 -> H-E participates",
    "C4": "otherwise report per record; C1 and C3 may hold simultaneously",
}


class ProbeRefusal(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# heads (all end in tanh * per-family scale, exactly the v15 output contract)
# ---------------------------------------------------------------------------


class WideHead(nn.Module):
    """M2: depthwise-separable 29->64->64->16, dilation 1 throughout."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, INPUT_CHANNELS, 3, padding=1, groups=INPUT_CHANNELS, bias=True),
            nn.Conv2d(INPUT_CHANNELS, 64, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True),
            nn.Conv2d(64, 64, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=True),
            nn.Conv2d(64, RANK, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features).tanh()


class DilatedHead(nn.Module):
    """M3: same width, depthwise dilations 1/4/16 (receptive field ~43 px)."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, INPUT_CHANNELS, 3, padding=1, dilation=1, groups=INPUT_CHANNELS, bias=True),
            nn.Conv2d(INPUT_CHANNELS, 64, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4, groups=64, bias=True),
            nn.Conv2d(64, 64, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=16, dilation=16, groups=64, bias=True),
            nn.Conv2d(64, RANK, 1, bias=True),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features).tanh()


class V15HeadAdapter(nn.Module):
    """M1: the frozen 1202-parameter head, exposed under the same interface."""

    def __init__(self, pipe) -> None:
        super().__init__()
        self.model = r4e10.fresh_model(pipe)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model.coefficient_head(features)


def head_coefficient(pipe, head: nn.Module, bundle) -> torch.Tensor:
    unit = head(bundle.features)
    coefficient = torch.zeros_like(unit, dtype=torch.float32)
    scale = pipe["scales"].to(unit.device)[bundle.route_index, :, None, None].float()
    coefficient[0] = unit[0].float() * scale
    return coefficient


def parameter_count(head: nn.Module) -> int:
    return sum(p.numel() for p in head.parameters())


# ---------------------------------------------------------------------------
# oracle coefficient targets (the leg-A saturation arithmetic, kept as maps)
# ---------------------------------------------------------------------------


def oracle_coefficient_target(pipe, bundle) -> torch.Tensor:
    """Per-pixel weighted-LS oracle coefficients, clipped into the tanh range."""
    parent_future = bundle.args[7][0, bundle.k1 + 1:].detach().cpu().double()
    truth_future = bundle.truth_future.detach().cpu().double()
    frames = parent_future.shape[0]
    height, width = parent_future.shape[1], parent_future.shape[2]
    points = height * width
    flat_parent = parent_future.reshape(frames, points)
    flat_truth = truth_future.reshape(frames, points)
    residual = flat_truth - flat_parent
    keep, _floor = parent_energy_keep_mask(parent_future, tau=TAU)
    keep_index = keep.nonzero(as_tuple=True)[0]
    truth_energy = flat_truth.square().sum(dim=1)
    _mk, metric_floor = energy_keep_mask(truth_energy, tau=TAU)
    weights = truth_energy.clamp_min(metric_floor).reciprocal()
    basis = pipe["bases"][bundle.route_index].detach().cpu().double()[bundle.k1 + 1:]
    mapping = weighted_coefficient_map(basis[keep_index], weights[keep_index])
    coefficients = mapping @ residual[keep_index]                    # [rank, points]
    scale = pipe["scales"][bundle.route_index].double()[:, None]
    clipped = coefficients.clamp(-CLIP_EDGE * scale, CLIP_EDGE * scale)
    return clipped.reshape(RANK, height, width).float().to(pipe["device"])


def target_loss(pipe, bundle, coefficient: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = pipe["scales"].to(coefficient.device)[bundle.route_index, :, None, None].float()
    return ((coefficient[0] - target) / scale).square().mean()


def fit_rel_err(pipe, bundle, coefficient: torch.Tensor, target: torch.Tensor) -> float:
    num = float((coefficient[0].double() - target.double()).square().sum().sqrt())
    den = float(target.double().square().sum().sqrt())
    return num / max(den, 1e-300)


# ---------------------------------------------------------------------------
# arm runners
# ---------------------------------------------------------------------------


def evaluate_head(pipe, head, oracle_bounds, targets) -> dict[str, Any]:
    out = {}
    with torch.no_grad():
        for b in pipe["bundles"]:
            coefficient = head_coefficient(pipe, head, b)
            losses, adapted = r4e10.record_loss(pipe, b, coefficient)
            gain = r4e10.achieved_gain(pipe, b, adapted)
            bound = float(oracle_bounds[b.sample_id]["acceptance_convention_gain"])
            out[b.sample_id] = {
                "field_loss": float(losses["total"]), **gain,
                "oracle_bound_iii": bound,
                "oracle_fraction": gain["gain_iii"] / max(abs(bound), 1e-30),
                "fit_rel_err": fit_rel_err(pipe, b, coefficient, targets[b.sample_id]),
            }
    return out


def train_arm(pipe, head_factory, *, supervision: str, oracle_bounds, targets,
              deadline: float) -> dict[str, Any]:
    configure_determinism(372)
    head = head_factory().to(pipe["device"]).float()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)
    bundles = pipe["bundles"]
    snapshot = None
    status = "completed"
    forward_s = []
    for update in range(UPDATES):
        if time.monotonic() > deadline:
            status = "partial_budget"
            break
        bundle = bundles[update % len(bundles)]
        t0 = time.perf_counter()
        coefficient = head_coefficient(pipe, head, bundle)
        forward_s.append(time.perf_counter() - t0)
        if supervision == "field":
            losses, _ = r4e10.record_loss(pipe, bundle, coefficient)
            total = losses["total"]
        else:
            total = target_loss(pipe, bundle, coefficient, targets[bundle.sample_id])
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grads = [p.grad for p in head.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads):
            raise ProbeRefusal("nonfinite or absent gradient in an arm")
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if update + 1 == SNAPSHOT_UPDATE:
            snapshot = evaluate_head(pipe, head, oracle_bounds, targets)
    final = evaluate_head(pipe, head, oracle_bounds, targets)
    result = {
        "status": status,
        "supervision": supervision,
        "updates": UPDATES,
        "learning_rate": LR,
        "parameter_count": parameter_count(head),
        "mean_forward_s": sum(forward_s) / max(len(forward_s), 1),
        "snapshot_at_192": snapshot,
        "final": final,
    }
    del head, optimizer
    torch.cuda.empty_cache()
    return result


def ridge_arm(pipe, oracle_bounds, targets) -> dict[str, Any]:
    configure_determinism(372)
    ridge = RidgePointwiseBaseline()
    features = torch.cat([b.features.detach().cpu() for b in pipe["bundles"]], dim=0)
    stacked = torch.stack([targets[b.sample_id].detach().cpu()
                           for b in pipe["bundles"]], dim=0)
    ridge.fit_closed_form(features, stacked)
    ridge = ridge.to(pipe["device"]).float()

    # The linear ridge predicts coefficients directly; its output is clamped
    # into the representable tanh range before the confined evaluation.
    out = {}
    with torch.no_grad():
        for b in pipe["bundles"]:
            raw = ridge(b.features)[0]
            scale = pipe["scales"].to(raw.device)[b.route_index, :, None, None].float()
            clamped = raw.clamp(-CLIP_EDGE * scale, CLIP_EDGE * scale)
            coefficient = torch.zeros(1, *clamped.shape, device=raw.device)
            coefficient[0] = clamped
            losses, adapted = r4e10.record_loss(pipe, b, coefficient)
            gain = r4e10.achieved_gain(pipe, b, adapted)
            bound = float(oracle_bounds[b.sample_id]["acceptance_convention_gain"])
            out[b.sample_id] = {
                "field_loss": float(losses["total"]), **gain,
                "oracle_bound_iii": bound,
                "oracle_fraction": gain["gain_iii"] / max(abs(bound), 1e-30),
                "fit_rel_err": fit_rel_err(pipe, b, coefficient, targets[b.sample_id]),
            }
    return {"status": "completed", "supervision": "closed_form_weighted_ridge",
            "parameter_count": parameter_count(ridge),
            "output_composition": "linear_clamped_to_0.999_scale",
            "final": out}


# ---------------------------------------------------------------------------
# classification (transcribed from the frozen spec)
# ---------------------------------------------------------------------------


def classify(arms: Mapping[str, Any], r4e10_leg_a: Mapping[str, Any]) -> dict[str, Any]:
    verdict: dict[str, Any] = {"rule": RULE}
    records = [sid for _f, sid in r4e10.RECORDS]

    def fractions(arm):
        return {sid: arm["final"][sid]["oracle_fraction"] for sid in records}

    deployable = {k: v for k, v in arms.items()
                  if k.startswith(("M2", "M3")) and v.get("final")}
    c1_hits = {}
    for name, arm in deployable.items():
        f = fractions(arm)
        if sum(v >= 0.5 for v in f.values()) >= 2:
            c1_hits[name] = f
    verdict["C1_capacity_confirmed"] = bool(c1_hits)
    verdict["C1_arms"] = c1_hits

    shared = {k: v for k, v in arms.items() if v.get("final")}
    all_below = all(
        sum(v < 0.5 for v in fractions(arm).values()) >= 2 for arm in shared.values()
    )
    target_arms = {k: v for k, v in arms.items()
                   if v.get("supervision") == "target" and v.get("final")}
    best_fit = None
    if target_arms:
        best_fit = min(
            min(arm["final"][sid]["fit_rel_err"] for sid in records)
            for arm in target_arms.values()
        )
    verdict["C2_information_bottleneck"] = bool(
        all_below and best_fit is not None and best_fit > 0.9
    )
    verdict["best_target_fit_rel_err"] = best_fit

    lega = r4e10_leg_a["result"]["arms"]["lr_0.003"]["final"]
    m1t = arms.get("M1t")
    if m1t and m1t.get("final"):
        gains = {sid: m1t["final"][sid]["oracle_fraction"] - lega[sid]["oracle_fraction"]
                 for sid in records}
        c3_primary = sum(v >= 0.15 for v in gains.values()) >= 2
        gap_ok = True
        for base in ("M2", "M3"):
            ft, ff = arms.get(f"{base}t"), arms.get(f"{base}f")
            if ft and ff and ft.get("final") and ff.get("final"):
                deltas = [ft["final"][sid]["oracle_fraction"] - ff["final"][sid]["oracle_fraction"]
                          for sid in records]
                if sum(d >= 0.15 for d in deltas) >= 2:
                    gap_ok = False  # an equal field-vs-target gap exists at M2/M3
        verdict["C3_loss_surface_participates"] = bool(c3_primary and gap_ok)
        verdict["M1t_fraction_gain_over_r4e10"] = gains
    else:
        verdict["C3_loss_surface_participates"] = False
    if not (verdict["C1_capacity_confirmed"] or verdict["C2_information_bottleneck"]
            or verdict["C3_loss_surface_participates"]):
        verdict["branch"] = "C4_mixed_report_per_record"
    else:
        verdict["branch"] = "+".join(
            name for name, hit in (
                ("C1", verdict["C1_capacity_confirmed"]),
                ("C2", verdict["C2_information_bottleneck"]),
                ("C3", verdict["C3_loss_surface_participates"]),
            ) if hit
        )
    return verdict


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


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
        raise ProbeRefusal("r4e11 spec drifted after freeze")
    bindings = r4e10.verify_bindings()
    for rel, expected in EXTRA_FROZEN.items():
        observed = sha256_file(PROJECT_ROOT / rel)
        bindings[rel] = {"expected": expected, "observed": observed}
        if observed != expected:
            raise ProbeRefusal(f"binding drift: {rel}")
    parent_runtime.require_free_disk(minimum=r4e10.MIN_FREE_BYTES)

    device = torch.device("cuda:0")
    gpu = parent_runtime.gpu_identity(args.physical_gpu_index, device)
    torch.cuda.reset_peak_memory_stats(device)

    smoke = json.loads(r4e10.SMOKE_TERMINAL.read_text())
    r4e10_leg_a = json.loads(
        (PROJECT_ROOT / "results/r4e10_optimization_gap_20260826/leg_a/terminal.json").read_text()
    )
    pipe = r4e10.build_pipeline(device)
    fidelity = r4e10.fidelity_gate(pipe, smoke)
    loss_fidelity = r4e10.initial_loss_fidelity(pipe, smoke)
    base = {
        "schema": "r4e11_head_capacity_ladder_result_v1",
        "candidate": CANDIDATE,
        "kind": "diagnostic_probe_not_a_promotion_candidate",
        "spec_sha256": SPEC_SHA256,
        "started_utc": r4e10.utc_now(),
        "gpu": gpu,
        "pipeline_fidelity": fidelity,
        "initial_loss_fidelity": loss_fidelity,
        "claim_scope": (
            "offline_train_only_three_record_fit_capacity_ladder_"
            "not_generalization_not_a_stage_not_promotable"
        ),
        "thresholds_moved_toward_measured_values": False,
        "sealed_splits_opened": False,
    }
    if not (fidelity["passed"] and loss_fidelity["passed"]):
        payload = {**base, "status": "invalid_pipeline", "completed_utc": r4e10.utc_now()}
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        (RESULT_DIR / "terminal.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(json.dumps({"status": "invalid_pipeline"}))
        return 1

    oracle_bounds = {b.sample_id: r4e10.oracle_for(pipe, b) for b in pipe["bundles"]}
    targets = {b.sample_id: oracle_coefficient_target(pipe, b) for b in pipe["bundles"]}

    arms: dict[str, Any] = {}
    arms["R"] = ridge_arm(pipe, oracle_bounds, targets)
    arms["M1t"] = train_arm(pipe, lambda: V15HeadAdapter(pipe), supervision="target",
                            oracle_bounds=oracle_bounds, targets=targets, deadline=deadline)
    arms["M2f"] = train_arm(pipe, WideHead, supervision="field",
                            oracle_bounds=oracle_bounds, targets=targets, deadline=deadline)
    arms["M2t"] = train_arm(pipe, WideHead, supervision="target",
                            oracle_bounds=oracle_bounds, targets=targets, deadline=deadline)
    arms["M3f"] = train_arm(pipe, DilatedHead, supervision="field",
                            oracle_bounds=oracle_bounds, targets=targets, deadline=deadline)
    arms["M3t"] = train_arm(pipe, DilatedHead, supervision="target",
                            oracle_bounds=oracle_bounds, targets=targets, deadline=deadline)

    verdict = classify(arms, r4e10_leg_a)

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
        "oracle_bounds": {k: {"acceptance_convention_gain": v["acceptance_convention_gain"]}
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
    os.replace(temp, RESULT_DIR / "terminal.json")
    print(json.dumps({"status": "success", "elapsed_s": round(elapsed, 1),
                      "verdict": {k: v for k, v in verdict.items() if k != "rule"}},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
