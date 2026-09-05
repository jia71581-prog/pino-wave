#!/usr/bin/env python3
"""r4e10: two-leg diagnostic probe behind the v15 smoke fail_gate.

Frozen spec: results/r4e10_optimization_gap_spec_20260826.md

Leg A (convergence / lr sweep) asks whether the v15 ansatz can approach its own
per-record confined oracle bound when the optimization budget and learning rate
stop being the smoke defaults.  Leg B diagnoses the uniform in-mask degradation
mechanism (direction mismatch / weak gradient / lr overshoot / tanh-scale
saturation).

This is a train-only diagnostic.  It is not a stage, produces no promotable
evidence, writes no model checkpoint, moves no threshold, and never touches
validation or test_id.  Deviation from the smoke path, declared in the spec:
the probe runs fp32 throughout (the smoke coefficient head ran bf16 autocast).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

for _value in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _value not in sys.path:
        sys.path.insert(0, _value)

from saved_time_phase_operator_v4.eikonal import grid_eikonal_travel_time
from saved_time_phase_operator_v4.instance_adaptation.data_guard import GuardedOnsetDataset
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp import (
    CONDITION_MAXIMUM,
    R16DSCP,
    c1_causal_mask,
    deployment_features,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v3 import relative_l2
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_engine_v4 import (
    V4GuardedOnsetLoader,
    canonical_public,
    canonical_travel,
    deployment_args,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v2 import (
    configure_determinism,
    sha256_file,
)
from saved_time_phase_operator_v4.instance_adaptation.r16_dscp_training_v3 import (
    TAU,
    apply_confined_correction,
    confined_oracle_upper_bound,
    energy_keep_mask,
    frame_energy,
    full_time_keep_mask,
    masked_confined_loss,
    parent_energy_keep_mask,
    weighted_coefficient_map,
)
from scripts import benchmark_r4_parent_e2e_trainonly as parent_runtime
from scripts import probe_r4e8_masked_oracle as r4e8


CANDIDATE = "r4e10_optimization_gap"
SPEC_PATH = PROJECT_ROOT / "results/r4e10_optimization_gap_spec_20260826.md"
SPEC_SHA256 = "0c2739a561da8595cbdeb5c020abbf84adcaafe80ed9ba4ad4ed17618824f6ff"
RESULT_ROOT = PROJECT_ROOT / "results/r4e10_optimization_gap_20260826"

BASIS = PROJECT_ROOT / "results/r16_dscp_v1/basis_rank16.pt"
PANELS = PROJECT_ROOT / "results/r16_dscp_v1/panels.json"
SMOKE_TERMINAL = PROJECT_ROOT / "results/r16_dscp_v15/smoke/terminal.json"
SMOKE_LAST = PROJECT_ROOT / "results/r16_dscp_v15/smoke/last.pt"

# Frozen bindings, transcribed from the spec table.  Drift is a refusal.
FROZEN = {
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp.py": "f906a422c26c86a736e21f77842ee83892ce7bd48364e2ca665a261451070201",
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v2.py": "959619b032a77af8f6095ba927e46fa30502df338f8067179ea4a94cfd4945a9",
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_training_v3.py": "8e0ae9294ff615633c07a1c686577325ed6785bf46ffb07d98e3e393963ba517",
    "saved_time_phase_operator_v4/instance_adaptation/r16_dscp_engine_v15.py": "4a3ca6fb5792347055cb1d7af869713e430ef9a2b59a7921b667fdd0915602ce",
    "results/r16_dscp_v1/basis_rank16.pt": "8cab01344fc0a88f43fa232458127e87d2f8bbb63523b9389b927b39389c4786",
    "results/r16_dscp_v1/panels.json": "6d2f2facd86b449c058f9f824895f77977a23fbecf954c0de69144ec50162464",
    "results/r16_dscp_v15/smoke/terminal.json": "0947af12a248bb92e77f417a3dcd6b41dc922118afb68d76d94985ce15dccf35",
    "results/r16_dscp_v15/smoke/last.pt": "c8e54e0335f53185d45a6fb3c8f4b56de09d9936c49602ff7d58228628a9857c",
}

RECORDS = (
    ("uniform", "train_uniform_00321"),
    ("layered", "train_layered_00564"),
    ("marmousi", "train_marmousi_00385"),
)

# Frozen probe parameters (spec, leg A / leg B sections)
MODEL_ARM_LRS = (3e-4, 1e-3, 3e-3, 1e-2)
MODEL_ARM_UPDATES = 768
EVAL_EVERY = 48
DIRECT_LR = 3e-2
DIRECT_STEPS = 512
SATURATION_EDGE = 0.99
ONE_STEP_LRS = (3e-5, 3e-4, 3e-3, 3e-2)
GPU_SECONDS_MAXIMUM = 1800.0
PEAK_RESERVED_MAXIMUM = int(20 * 1024**3)
MIN_FREE_BYTES = 2 * 1024**3
FIDELITY_REL_TOL_LOSS = 1e-3
FIDELITY_REL_TOL_PARENT = 1e-3
FIDELITY_REL_TOL_ORACLE = 1e-6

# Pre-registered readings (spec), transcribed so the terminal carries them.
LEG_A_RULE = {
    "A1": "best-lr arm reaches f>=0.5 of oracle on >=2/3 records -> H-A (budget-bound)",
    "A2": "all model arms f<0.25 everywhere while direct arm f>=0.5 -> H-B (feature head)",
    "A3": "direct arm also f<0.5 and saturation>0.10 -> H-B (tanh-scale parameterization)",
    "A4": "otherwise report per record, no single verdict",
}
LEG_B_RULE = {
    "B1": "in-mask cos(correction, error) < 0 at smoke-final -> direction mismatch",
    "B2": "uniform grad norm < 0.1 x median of other two at init -> weak gradient signal",
    "B3": "one-step dloss(3e-3) > 0 and dloss(3e-4) < 0 -> lr overshoot",
    "B4": "uniform oracle saturation fraction > 0.10 -> parameterization cap",
}


class ProbeRefusal(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def verify_bindings() -> dict[str, Any]:
    checked = {}
    if sha256_file(SPEC_PATH) != SPEC_SHA256:
        raise ProbeRefusal("spec drifted after freeze")
    for rel, expected in FROZEN.items():
        observed = sha256_file(PROJECT_ROOT / rel)
        checked[rel] = {"expected": expected, "observed": observed}
        if observed != expected:
            raise ProbeRefusal(f"binding drift: {rel}")
    parent_sha = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if parent_sha != parent_runtime.CHECKPOINT_SHA256:
        raise ProbeRefusal("parent checkpoint drifted")
    checked["parent_checkpoint"] = {"observed": parent_sha}
    return checked


def atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    if not str(path.resolve()).startswith(str(RESULT_ROOT.resolve())):
        raise ProbeRefusal(f"write outside the whitelisted result root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(temp, path)


# ---------------------------------------------------------------------------
# shared pipeline (replicates the v10 factory + v15 engine paths, fp32)
# ---------------------------------------------------------------------------


class RecordBundle:
    def __init__(self, family, sample_id, args, features, route_index, condition,
                 k1, truth_future, keep_full, keep_future):
        self.family = family
        self.sample_id = sample_id
        self.args = args
        self.features = features
        self.route_index = route_index
        self.condition = condition
        self.k1 = k1
        self.truth_future = truth_future
        self.keep_full = keep_full
        self.keep_future = keep_future


def build_pipeline(device: torch.device) -> dict[str, Any]:
    configure_determinism(372)
    artifact = torch.load(BASIS, map_location="cpu", weights_only=False)
    parent_model, normalizer, manifest_payload, _ = parent_runtime.load_model_context(device)
    manifest = parent_runtime.manifest_object(manifest_payload)
    sample_ids = [sid for _f, sid in RECORDS]
    loader = V4GuardedOnsetLoader(
        GuardedOnsetDataset(parent_runtime.SOURCE_H5_PATH, manifest, split="train",
                            sample_ids=sample_ids)
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

    bundles = []
    order = {sid: i for i, (_f, sid) in enumerate(RECORDS)}
    loaded = {}
    for index in range(len(loader)):
        public = canonical_public(loader[index])
        loaded[public.sample_id] = (index, public)
    for family, sample_id in RECORDS:
        _index, public = loaded[sample_id]
        parent = torch.as_tensor(parent_predictor(public)).detach().cpu().float()
        travel = canonical_travel(public, travel_builder).float()
        args = deployment_args(public, parent, travel, device)
        features, decisions, conditions = deployment_features(
            *args[:-2], bases.to(device).float(), args[-2], args[-1]
        )
        decision = decisions[0]
        if decision.abstain or float(conditions[0]) > CONDITION_MAXIMUM:
            raise ProbeRefusal(f"record abstained, probe undefined: {sample_id}")
        record = r4e8.record_from_manifest(manifest_payload, sample_id)
        truth, _truth_sha = r4e8.load_authorized_train_truth(record, device=torch.device("cpu"))
        k1 = int(public.observed_indices[1])
        truth_future = truth[k1 + 1:].float().to(device)
        parent_field = args[7][0]
        keep_full, _floor = full_time_keep_mask(parent_field.detach().float(), k1=k1, tau=TAU)
        bundles.append(RecordBundle(
            family, sample_id, args, features.detach().float(), int(decision.index),
            float(conditions[0]), k1, truth_future,
            keep_full.to(device), keep_full[k1 + 1:].to(device),
        ))
    bundles.sort(key=lambda b: order[b.sample_id])
    return {
        "artifact": artifact, "bases": bases, "scales": scales,
        "bundles": bundles, "device": device,
    }


def fresh_model(pipe) -> R16DSCP:
    configure_determinism(372)
    model = R16DSCP(pipe["artifact"]["basis"], pipe["artifact"]["coefficient_scales"])
    return model.to(pipe["device"]).float()


def materialize(pipe, bundle: RecordBundle, coefficient: torch.Tensor) -> torch.Tensor:
    """The v15 confined correction path (fp32), returning the corrected batch."""
    parent = bundle.args[7]
    corrected = parent.float().clone()
    basis = pipe["bases"].to(parent.device)[bundle.route_index].float()
    correction = torch.einsum("tr,rhw->thw", basis, coefficient[0])
    ramp = c1_causal_mask(parent.shape[1], bundle.k1, device=parent.device, dtype=torch.float32)
    correction = correction * ramp[:, None, None]
    correction = torch.cat(
        (torch.zeros_like(correction[:, :1]), correction[:, 1:]), dim=1
    )
    confined_field = apply_confined_correction(parent[0].float(), correction, bundle.keep_full)
    return torch.cat((confined_field[None], corrected[1:]), dim=0)


def model_coefficient(pipe, model: R16DSCP, bundle: RecordBundle) -> torch.Tensor:
    unit = model.coefficient_head(bundle.features)
    coefficient = torch.zeros_like(unit, dtype=torch.float32)
    coefficient[0] = unit[0].float() * model.coefficient_scales[bundle.route_index, :, None, None].float()
    return coefficient


def record_loss(pipe, bundle: RecordBundle, coefficient: torch.Tensor):
    adapted = materialize(pipe, bundle, coefficient)
    parent = bundle.args[7]
    return masked_confined_loss(
        adapted[:, bundle.k1 + 1:].float(),
        bundle.truth_future[None],
        coefficient,
        parent[:, bundle.k1 + 1:].float(),
        bundle.keep_future,
        tau=TAU,
    ), adapted


def achieved_gain(pipe, bundle: RecordBundle, adapted: torch.Tensor) -> dict[str, float]:
    candidate_future = adapted[0, bundle.k1 + 1:].float()
    parent_future = bundle.args[7][0, bundle.k1 + 1:].float()
    cand = relative_l2(candidate_future, bundle.truth_future)
    par = relative_l2(parent_future, bundle.truth_future)
    return {
        "aggregate_rel_l2": float(cand),
        "parent_rel_l2": float(par),
        "gain_iii": float((par - cand) / max(abs(par), 1e-30)),
    }


def finite_backward(model, optimizer, total: torch.Tensor, *, step: bool) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise ProbeRefusal("nonfinite or absent gradient")
    unclipped = math.sqrt(sum(float(g.detach().double().square().sum()) for g in grads))
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if step:
        optimizer.step()
    return {"unclipped_grad_norm": unclipped}


def oracle_for(pipe, bundle: RecordBundle) -> dict[str, Any]:
    parent_future = bundle.args[7][0, bundle.k1 + 1:].detach().cpu()
    truth_future = bundle.truth_future.detach().cpu()
    basis_future = pipe["bases"][bundle.route_index].detach().cpu()[bundle.k1 + 1:]
    return confined_oracle_upper_bound(parent_future, truth_future, basis_future, tau=TAU)


def oracle_coefficient_saturation(pipe, bundle: RecordBundle) -> dict[str, Any]:
    """Per-pixel oracle coefficients versus the tanh-scale representable bound."""
    parent_future = bundle.args[7][0, bundle.k1 + 1:].detach().cpu().double()
    truth_future = bundle.truth_future.detach().cpu().double()
    frames = parent_future.shape[0]
    points = parent_future.shape[1] * parent_future.shape[2]
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
    coefficients = mapping @ residual[keep_index]          # [rank, points]
    scale = pipe["scales"][bundle.route_index].double()[:, None]
    saturated = (coefficients.abs() > SATURATION_EDGE * scale)
    return {
        "saturation_edge": SATURATION_EDGE,
        "saturation_fraction": float(saturated.double().mean()),
        "per_rank_saturation": [float(v) for v in saturated.double().mean(dim=1)],
        "max_abs_over_scale": float((coefficients.abs() / scale).max()),
        "mean_abs_over_scale": float((coefficients.abs() / scale).mean()),
    }


def fidelity_gate(pipe, smoke: Mapping[str, Any]) -> dict[str, Any]:
    """Pipeline fidelity versus the stored smoke terminal.  Fail = invalid probe."""
    rows = {r["sample_id"]: r for r in smoke["metrics"]["records"]}
    smoke_oracle = smoke["metrics"]["oracle_by_record"]
    out = {"records": {}, "passed": True}
    for bundle in pipe["bundles"]:
        parent_future = bundle.args[7][0, bundle.k1 + 1:].float()
        par = float(relative_l2(parent_future, bundle.truth_future))
        want_par = float(rows[bundle.sample_id]["parent_rel_l2"])
        parent_rel_diff = abs(par - want_par) / max(abs(want_par), 1e-30)
        bound = oracle_for(pipe, bundle)
        want_bound = float(smoke_oracle[bundle.sample_id]["acceptance_convention_gain"])
        got_bound = float(bound["acceptance_convention_gain"])
        oracle_rel_diff = abs(got_bound - want_bound) / max(abs(want_bound), 1e-30)
        entry = {
            "parent_rel_l2": {"probe": par, "smoke": want_par, "rel_diff": parent_rel_diff,
                              "passed": parent_rel_diff <= FIDELITY_REL_TOL_PARENT},
            "oracle_gain_iii": {"probe": got_bound, "smoke": want_bound,
                                "rel_diff": oracle_rel_diff,
                                "passed": oracle_rel_diff <= FIDELITY_REL_TOL_ORACLE},
        }
        out["records"][bundle.sample_id] = entry
        out["passed"] = out["passed"] and all(v["passed"] for v in entry.values())
    return out


def initial_loss_fidelity(pipe, smoke: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the first smoke round (lr 3e-3) and compare first-seen losses."""
    per = smoke["per_record_loss"]
    model = fresh_model(pipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99),
                                  eps=1e-8, weight_decay=1e-4)
    out = {"records": {}, "passed": True}
    for bundle in pipe["bundles"]:
        losses, _ = record_loss(pipe, bundle, model_coefficient(pipe, model, bundle))
        observed = float(losses["total"])
        finite_backward(model, optimizer, losses["total"], step=True)
        want = float(per[bundle.sample_id]["initial_loss"])
        rel = abs(observed - want) / max(abs(want), 1e-30)
        out["records"][bundle.sample_id] = {
            "probe_first_loss": observed, "smoke_initial_loss": want,
            "rel_diff": rel, "passed": rel <= FIDELITY_REL_TOL_LOSS,
        }
        out["passed"] = out["passed"] and out["records"][bundle.sample_id]["passed"]
    del model, optimizer
    return out


# ---------------------------------------------------------------------------
# leg A
# ---------------------------------------------------------------------------


def run_leg_a(pipe, oracle_bounds, deadline: float) -> dict[str, Any]:
    bundles = pipe["bundles"]
    arms: dict[str, Any] = {}
    for lr in MODEL_ARM_LRS:
        if time.monotonic() > deadline:
            arms[f"lr_{lr:g}"] = {"status": "skipped_budget"}
            continue
        model = fresh_model(pipe)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99),
                                      eps=1e-8, weight_decay=1e-4)
        trajectory = []
        status = "completed"
        for update in range(MODEL_ARM_UPDATES):
            if time.monotonic() > deadline:
                status = "partial_budget"
                break
            bundle = bundles[update % len(bundles)]
            losses, _ = record_loss(pipe, bundle, model_coefficient(pipe, model, bundle))
            finite_backward(model, optimizer, losses["total"], step=True)
            if (update + 1) % EVAL_EVERY == 0:
                snap = {"update": update + 1, "records": {}}
                with torch.no_grad():
                    for b in bundles:
                        l2, adapted = record_loss(pipe, b, model_coefficient(pipe, model, b))
                        snap["records"][b.sample_id] = {
                            "loss": float(l2["total"]),
                            **achieved_gain(pipe, b, adapted),
                        }
                trajectory.append(snap)
        final = {}
        with torch.no_grad():
            for b in bundles:
                l2, adapted = record_loss(pipe, b, model_coefficient(pipe, model, b))
                gain = achieved_gain(pipe, b, adapted)
                bound = float(oracle_bounds[b.sample_id]["acceptance_convention_gain"])
                final[b.sample_id] = {
                    "loss": float(l2["total"]), **gain,
                    "oracle_bound_iii": bound,
                    "oracle_fraction": gain["gain_iii"] / max(abs(bound), 1e-30),
                }
        arms[f"lr_{lr:g}"] = {"status": status, "learning_rate": lr,
                              "trajectory": trajectory, "final": final}
        del model, optimizer
        torch.cuda.empty_cache()

    direct = {}
    for b in bundles:
        if time.monotonic() > deadline:
            direct[b.sample_id] = {"status": "skipped_budget"}
            continue
        raw = torch.zeros(16, *b.args[7].shape[2:], device=pipe["device"],
                          dtype=torch.float32, requires_grad=True)
        scale = pipe["scales"].to(pipe["device"])[b.route_index][:, None, None].float()
        adam = torch.optim.Adam([raw], lr=DIRECT_LR)
        track = []
        status = "completed"
        for step_index in range(DIRECT_STEPS):
            if time.monotonic() > deadline:
                status = "partial_budget"
                break
            coefficient = (torch.tanh(raw) * scale)[None]
            losses, _ = record_loss(pipe, b, coefficient)
            adam.zero_grad(set_to_none=True)
            losses["total"].backward()
            if not torch.isfinite(raw.grad).all():
                raise ProbeRefusal("direct arm gradient is nonfinite")
            adam.step()
            if (step_index + 1) % 128 == 0:
                track.append({"step": step_index + 1, "loss": float(losses["total"])})
        with torch.no_grad():
            coefficient = (torch.tanh(raw) * scale)[None]
            l2, adapted = record_loss(pipe, b, coefficient)
            gain = achieved_gain(pipe, b, adapted)
            bound = float(oracle_bounds[b.sample_id]["acceptance_convention_gain"])
        direct[b.sample_id] = {
            "status": status, "steps": DIRECT_STEPS, "learning_rate": DIRECT_LR,
            "trajectory": track, "loss": float(l2["total"]), **gain,
            "oracle_bound_iii": bound,
            "oracle_fraction": gain["gain_iii"] / max(abs(bound), 1e-30),
        }
        del raw, adam
        torch.cuda.empty_cache()

    saturation = {b.sample_id: oracle_coefficient_saturation(pipe, b) for b in bundles}

    # pre-registered classification
    completed_arms = {k: v for k, v in arms.items() if v.get("status") == "completed"}
    verdict: dict[str, Any] = {"rule": LEG_A_RULE}
    if completed_arms:
        def arm_fractions(arm):
            return [arm["final"][sid]["oracle_fraction"] for _f, sid in RECORDS]
        best_name, best_arm = max(
            completed_arms.items(),
            key=lambda kv: sum(arm_fractions(kv[1])),
        )
        best_fractions = arm_fractions(best_arm)
        a1 = sum(f >= 0.5 for f in best_fractions) >= 2
        all_below_quarter = all(
            f < 0.25 for arm in completed_arms.values() for f in arm_fractions(arm)
        )
        direct_ok = [v.get("oracle_fraction", float("nan")) for v in direct.values()
                     if v.get("status") in ("completed", "partial_budget")]
        direct_ge_half = bool(direct_ok) and all(f >= 0.5 for f in direct_ok)
        direct_lt_half = bool(direct_ok) and any(f < 0.5 for f in direct_ok)
        high_saturation = any(s["saturation_fraction"] > 0.10 for s in saturation.values())
        if a1:
            verdict["branch"] = "A1_budget_bound"
        elif all_below_quarter and direct_ge_half:
            verdict["branch"] = "A2_feature_head_bound"
        elif direct_lt_half and high_saturation:
            verdict["branch"] = "A3_parameterization_bound"
        else:
            verdict["branch"] = "A4_mixed_report_per_record"
        verdict["best_arm"] = best_name
        verdict["best_arm_oracle_fractions"] = dict(zip([s for _f, s in RECORDS], best_fractions))
    else:
        verdict["branch"] = "no_completed_arm_budget"
    return {"arms": arms, "direct_fit": direct, "oracle_saturation": saturation,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# leg B
# ---------------------------------------------------------------------------


def load_smoke_final(pipe) -> R16DSCP:
    payload = torch.load(SMOKE_LAST, map_location="cpu", weights_only=False)
    if payload["basis"]["file_sha256"] != FROZEN["results/r16_dscp_v1/basis_rank16.pt"]:
        raise ProbeRefusal("smoke checkpoint bound to a different basis artifact")
    model = fresh_model(pipe)
    missing, unexpected = model.load_state_dict(payload["predictor_parameters"], strict=False)
    if unexpected:
        raise ProbeRefusal(f"unexpected checkpoint keys: {unexpected}")
    surviving = [k for k in missing if not k.startswith(("bases", "coefficient_scales"))]
    if surviving:
        raise ProbeRefusal(f"missing predictor parameters: {surviving}")
    return model


def direction_diagnostics(pipe, bundle: RecordBundle, model: R16DSCP) -> dict[str, Any]:
    with torch.no_grad():
        coefficient = model_coefficient(pipe, model, bundle)
        adapted = materialize(pipe, bundle, coefficient)
        parent_future = bundle.args[7][0, bundle.k1 + 1:].double()
        candidate_future = adapted[0, bundle.k1 + 1:].double()
        truth_future = bundle.truth_future.double()
        correction = candidate_future - parent_future
        error = truth_future - parent_future
        keep = bundle.keep_future
        c_in = correction[keep]
        e_in = error[keep]
        c_norm = float(c_in.square().sum().sqrt())
        e_norm = float(e_in.square().sum().sqrt())
        cos_global = float((c_in * e_in).sum()) / max(c_norm * e_norm, 1e-300)
        per_frame_cos = []
        per_frame_gain = []
        parent_err_sq = (parent_future - truth_future).square().sum(dim=(1, 2))
        cand_err_sq = (candidate_future - truth_future).square().sum(dim=(1, 2))
        for t in range(parent_future.shape[0]):
            if not bool(keep[t]):
                continue
            ct, et = correction[t], error[t]
            denom = float(ct.square().sum().sqrt() * et.square().sum().sqrt())
            per_frame_cos.append({
                "frame": t,
                "cos": float((ct * et).sum()) / max(denom, 1e-300),
                "parent_err_sq": float(parent_err_sq[t]),
                "delta_err_sq": float(cand_err_sq[t] - parent_err_sq[t]),
            })
        worst = sorted(per_frame_cos, key=lambda r: r["delta_err_sq"], reverse=True)[:10]
        gain = achieved_gain(pipe, bundle, adapted)
        correction_energy_ratio = float(
            correction.square().sum() / parent_future.square().sum().clamp_min(1e-300)
        )
    return {
        "cos_in_mask_global": cos_global,
        "correction_norm_in_mask": c_norm,
        "error_norm_in_mask": e_norm,
        "correction_energy_ratio": correction_energy_ratio,
        "kept_frames": int(keep.sum()),
        "frames_cos_negative": sum(1 for r in per_frame_cos if r["cos"] < 0),
        "frames_delta_positive": sum(1 for r in per_frame_cos if r["delta_err_sq"] > 0),
        "worst_frames_by_delta_err": worst,
        **gain,
    }


def run_leg_b(pipe, deadline: float) -> dict[str, Any]:
    bundles = {b.family: b for b in pipe["bundles"]}
    smoke_model = load_smoke_final(pipe)
    direction = {family: direction_diagnostics(pipe, bundle, smoke_model)
                 for family, bundle in bundles.items()}
    del smoke_model
    torch.cuda.empty_cache()

    saturation = {b.sample_id: oracle_coefficient_saturation(pipe, b)
                  for b in pipe["bundles"]}

    gradient = {}
    for family, bundle in bundles.items():
        model = fresh_model(pipe)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.99),
                                      eps=1e-8, weight_decay=1e-4)
        losses, _ = record_loss(pipe, bundle, model_coefficient(pipe, model, bundle))
        norms = finite_backward(model, optimizer, losses["total"], step=False)
        per_layer = {name: float(p.grad.detach().double().square().sum().sqrt())
                     for name, p in model.named_parameters() if p.grad is not None}
        gradient[family] = {"loss_at_init": float(losses["total"]),
                            **norms, "per_layer_grad_norm": per_layer}
        del model, optimizer
        torch.cuda.empty_cache()

    one_step = {}
    uniform = bundles["uniform"]
    for lr in ONE_STEP_LRS:
        if time.monotonic() > deadline:
            one_step[f"lr_{lr:g}"] = {"status": "skipped_budget"}
            continue
        model = fresh_model(pipe)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99),
                                      eps=1e-8, weight_decay=1e-4)
        losses, _ = record_loss(pipe, uniform, model_coefficient(pipe, model, uniform))
        before = float(losses["total"])
        finite_backward(model, optimizer, losses["total"], step=True)
        with torch.no_grad():
            after_losses, adapted = record_loss(
                pipe, uniform, model_coefficient(pipe, model, uniform)
            )
            gain = achieved_gain(pipe, uniform, adapted)
        one_step[f"lr_{lr:g}"] = {
            "learning_rate": lr, "loss_before": before,
            "loss_after": float(after_losses["total"]),
            "delta_loss": float(after_losses["total"]) - before,
            **gain,
        }
        del model, optimizer
        torch.cuda.empty_cache()

    verdict: dict[str, Any] = {"rule": LEG_B_RULE}
    verdict["B1_direction_mismatch"] = bool(direction["uniform"]["cos_in_mask_global"] < 0)
    other = sorted(gradient[f]["unclipped_grad_norm"] for f in ("layered", "marmousi"))
    median_other = other[len(other) // 2] if len(other) % 2 else sum(other) / 2
    verdict["B2_weak_gradient"] = bool(
        gradient["uniform"]["unclipped_grad_norm"] < 0.1 * median_other
    )
    d33 = one_step.get("lr_0.003", {}).get("delta_loss")
    d34 = one_step.get("lr_0.0003", {}).get("delta_loss")
    verdict["B3_lr_overshoot"] = bool(
        d33 is not None and d34 is not None and d33 > 0 and d34 < 0
    )
    verdict["B4_saturation_cap"] = bool(
        saturation[uniform.sample_id]["saturation_fraction"] > 0.10
    )
    if not any(verdict[k] for k in ("B1_direction_mismatch", "B2_weak_gradient",
                                    "B3_lr_overshoot", "B4_saturation_cap")):
        verdict["unexplained"] = True
    return {"direction": direction, "oracle_saturation": saturation,
            "gradient_at_init": gradient, "one_step_lr_sweep": one_step,
            "verdict": verdict}


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg", choices=("a", "b"), required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    args = parser.parse_args(argv)

    started = time.monotonic()
    deadline = started + GPU_SECONDS_MAXIMUM
    out_dir = RESULT_ROOT / f"leg_{args.leg}"
    terminal_path = out_dir / "terminal.json"
    if terminal_path.exists():
        raise ProbeRefusal(f"terminal already exists: {terminal_path}")

    device = torch.device("cuda:0")
    bindings_before = verify_bindings()
    parent_runtime.require_free_disk(minimum=MIN_FREE_BYTES)
    gpu = parent_runtime.gpu_identity(args.physical_gpu_index, device)
    torch.cuda.reset_peak_memory_stats(device)

    smoke = json.loads(SMOKE_TERMINAL.read_text())
    pipe = build_pipeline(device)
    fidelity = fidelity_gate(pipe, smoke)
    loss_fidelity = initial_loss_fidelity(pipe, smoke)
    payload_base = {
        "schema": "r4e10_optimization_gap_result_v1",
        "candidate": CANDIDATE,
        "kind": "diagnostic_probe_not_a_promotion_candidate",
        "leg": args.leg,
        "spec_sha256": SPEC_SHA256,
        "started_utc": utc_now(),
        "gpu": gpu,
        "fp32_deviation_from_smoke_bf16": True,
        "pipeline_fidelity": fidelity,
        "initial_loss_fidelity": loss_fidelity,
        "claim_scope": (
            "offline_train_only_optimization_diagnostics_on_the_three_smoke_records_"
            "not_a_stage_not_promotable_not_validation_not_test_id"
        ),
        "thresholds_moved_toward_measured_values": False,
        "sealed_splits_opened": False,
    }
    if not (fidelity["passed"] and loss_fidelity["passed"]):
        payload = {**payload_base, "status": "invalid_pipeline",
                   "completed_utc": utc_now()}
        atomic_json(payload, terminal_path)
        print(json.dumps({"status": "invalid_pipeline"}, indent=1))
        return 1

    oracle_bounds = {b.sample_id: oracle_for(pipe, b) for b in pipe["bundles"]}
    if args.leg == "a":
        body = run_leg_a(pipe, oracle_bounds, deadline)
    else:
        body = run_leg_b(pipe, deadline)

    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_reserved(device))
    if peak > PEAK_RESERVED_MAXIMUM:
        raise ProbeRefusal("peak reserved memory exceeded the probe gate")
    parent_after = parent_runtime.sha256_file(parent_runtime.CHECKPOINT_PATH)
    if parent_after != parent_runtime.CHECKPOINT_SHA256:
        raise ProbeRefusal("parent checkpoint changed during the probe")

    payload = {
        **payload_base,
        "status": "success",
        "oracle_bounds": {k: {kk: v[kk] for kk in (
            "acceptance_convention_gain", "fit_frame_count", "correction_energy_ratio")}
            for k, v in oracle_bounds.items()},
        "result": body,
        "elapsed_s": elapsed,
        "peak_cuda_reserved_bytes": peak,
        "parent_checkpoint_before_after_match": True,
        "bindings": bindings_before,
        "completed_utc": utc_now(),
    }
    atomic_json(payload, terminal_path)
    print(json.dumps({"status": "success", "leg": args.leg,
                      "elapsed_s": round(elapsed, 1),
                      "verdict": body["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
